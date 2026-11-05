import time
import random
import heapq
import bisect
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140  # 10s safety margin

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used

    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True

    def _scope(rows):
        return max(max(r) for r in rows)

    # Bitmask-based representation for fast conflict checking
    # Max scope we'll consider: 200 (generous upper bound)
    MAX_SCOPE = 200
    N_WORDS = (MAX_SCOPE + 63) // 64  # number of uint64 words needed

    def _diffs_to_mask(diffs):
        """Convert a set of differences to a numpy uint64 array bitmask."""
        mask = np.zeros(N_WORDS, dtype=np.uint64)
        for d in diffs:
            if 0 < d <= MAX_SCOPE:
                word_idx = d // 64
                bit_idx = d % 64
                mask[word_idx] |= (1 << bit_idx)
        return mask

    def _mask_has_conflict(used_mask, new_diffs):
        """Check if any new_diff conflicts with used_mask. Returns True if conflict."""
        if not new_diffs:
            return False
        new_mask = np.zeros(N_WORDS, dtype=np.uint64)
        for d in new_diffs:
            if 0 < d <= MAX_SCOPE:
                word_idx = d // 64
                bit_idx = d % 64
                new_mask[word_idx] |= (1 << bit_idx)
        return bool(np.any(used_mask & new_mask))

    def _row_diffs_list(row):
        """Get list of all positive differences in a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def _column_wise_construct(seed_offset, max_scope, rng):
        """Column-wise construction: build all rows simultaneously, position by position.
        At each position, assign values to all rows that minimize conflicts."""
        rows = [[0] for _ in range(n_rows)]
        used_mask = np.zeros(N_WORDS, dtype=np.uint64)
        
        for pos in range(1, k + 1):
            # For each row, find the best value for position pos
            # Process rows in order of most constrained first (highest current max)
            row_order = sorted(range(n_rows), key=lambda r: rows[r][-1], reverse=True)
            
            assigned = [False] * n_rows
            values = [None] * n_rows
            
            for ri in row_order:
                row = rows[ri]
                last = row[-1]
                found = False
                
                # Try values from last+1 upward
                for x in range(last + 1, max_scope + 1):
                    # Check monotonicity
                    if pos < k:
                        # No constraint from future positions yet
                        pass
                    
                    # Compute new diffs
                    new_diffs = [x - a for a in row]
                    
                    # Check conflict with used
                    if _mask_has_conflict(used_mask, new_diffs):
                        continue
                    
                    # Also check: if this row already has values assigned at later positions
                    # (shouldn't happen in column-wise, but safety)
                    
                    values[ri] = x
                    assigned[ri] = True
                    found = True
                    break
                
                if not found:
                    return None
            
            # Commit all assignments
            for ri in range(n_rows):
                if values[ri] is None:
                    return None
                x = values[ri]
                row = rows[ri]
                new_diffs = [x - a for a in row]
                
                # Update mask
                for d in new_diffs:
                    if 0 < d <= MAX_SCOPE:
                        word_idx = d // 64
                        bit_idx = d % 64
                        used_mask[word_idx] |= (1 << bit_idx)
                
                row.append(x)
        
        return rows

    def _column_wise_randomized(seed_offset, max_scope, rng, diversify_prob=0.3):
        """Column-wise with randomization for diversity."""
        rows = [[0] for _ in range(n_rows)]
        used_mask = np.zeros(N_WORDS, dtype=np.uint64)
        
        for pos in range(1, k + 1):
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            # With some probability, process in constrained-first order
            if rng.random() < 0.5:
                row_order = sorted(range(n_rows), key=lambda r: rows[r][-1], reverse=True)
            
            values = [None] * n_rows
            
            for ri in row_order:
                row = rows[ri]
                last = row[-1]
                found = False
                
                # Collect candidates
                candidates = []
                for x in range(last + 1, min(last + 50, max_scope + 1)):
                    new_diffs = [x - a for a in row]
                    if _mask_has_conflict(used_mask, new_diffs):
                        continue
                    candidates.append(x)
                
                if not candidates:
                    # Fallback: try wider range
                    for x in range(last + 1, max_scope + 1):
                        new_diffs = [x - a for a in row]
                        if _mask_has_conflict(used_mask, new_diffs):
                            continue
                        candidates.append(x)
                        break
                
                if not candidates:
                    return None
                
                # Pick: best-fit (smallest) with some randomization
                if rng.random() < diversify_prob and len(candidates) > 1:
                    idx = rng.randint(0, min(3, len(candidates) - 1))
                    values[ri] = candidates[idx]
                else:
                    values[ri] = candidates[0]
            
            # Commit all assignments
            for ri in range(n_rows):
                if values[ri] is None:
                    return None
                x = values[ri]
                row = rows[ri]
                new_diffs = [x - a for a in row]
                
                for d in new_diffs:
                    if 0 < d <= MAX_SCOPE:
                        word_idx = d // 64
                        bit_idx = d % 64
                        used_mask[word_idx] |= (1 << bit_idx)
                
                row.append(x)
        
        return rows

    def _backtrack_row_bitmask(used_mask, max_val, row, depth, rng, deadline):
        """Build one row via backtracking using bitmask for fast conflict checks."""
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        candidates = list(range(start, max_val + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            new_diffs = [x - a for a in row]
            if _mask_has_conflict(used_mask, new_diffs):
                continue
            
            row.append(x)
            # Update mask
            mask_backup = used_mask.copy()
            for d in new_diffs:
                if 0 < d <= MAX_SCOPE:
                    word_idx = d // 64
                    bit_idx = d % 64
                    used_mask[word_idx] |= (1 << bit_idx)
            
            result = _backtrack_row_bitmask(used_mask, max_val, row, depth + 1, rng, deadline)
            if result is not None:
                return result
            
            row.pop()
            used_mask[:] = mask_backup
        
        return None

    def _backtrack_construct_bitmask(max_val, num_rows, rng, deadline):
        """Try to construct all rows within max_val using bitmask backtracking."""
        used_mask = np.zeros(N_WORDS, dtype=np.uint64)
        rows = []
        
        for i in range(num_rows):
            if time.time() > deadline:
                return None
            
            row = [0]
            result = _backtrack_row_bitmask(used_mask, max_val, row, 1, rng, deadline)
            if result is None:
                return None
            
            rows.append(result)
            # Update mask
            for d in _row_diffs_list(result):
                if 0 < d <= MAX_SCOPE:
                    word_idx = d // 64
                    bit_idx = d % 64
                    used_mask[word_idx] |= (1 << bit_idx)
        
        return rows

    def _local_search_bitmask(rows, deadline, rng, iterations=20000):
        """Local search using bitmask for fast conflict checking."""
        best = [list(r) for r in rows]
        best_scope = _scope(best)
        
        # Build initial mask
        used_mask = np.zeros(N_WORDS, dtype=np.uint64)
        for row in best:
            for d in _row_diffs_list(row):
                if 0 < d <= MAX_SCOPE:
                    word_idx = d // 64
                    bit_idx = d % 64
                    used_mask[word_idx] |= (1 << bit_idx)
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            ri = rng.randint(0, len(best) - 1)
            pi = rng.randint(1, k)
            
            row = best[ri]
            old_val = row[pi]
            
            if pi < k:
                lo = row[pi - 1] + 1
                hi = min(best_scope, row[pi + 1] - 1)
            else:
                lo = row[pi - 1] + 1
                hi = best_scope
            
            if lo > hi:
                continue
            
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            # Compute old and new diffs
            old_diffs = []
            for i in range(pi):
                old_diffs.append(old_val - row[i])
            for i in range(pi + 1, len(row)):
                old_diffs.append(row[i] - old_val)
            
            new_diffs = []
            for i in range(pi):
                new_diffs.append(new_val - row[i])
            for i in range(pi + 1, len(row)):
                new_diffs.append(row[i] - new_val)
            
            # Check monotonicity
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            if pi < k and new_val >= row[pi + 1]:
                continue
            
            # Remove old diffs from mask
            for d in old_diffs:
                if 0 < d <= MAX_SCOPE:
                    word_idx = d // 64
                    bit_idx = d % 64
                    used_mask[word_idx] &= ~(1 << bit_idx)
            
            # Check new diffs for conflict
            conflict = False
            for d in new_diffs:
                if 0 < d <= MAX_SCOPE:
                    word_idx = d // 64
                    bit_idx = d % 64
                    if used_mask[word_idx] & (1 << bit_idx):
                        conflict = True
                        break
            
            if conflict:
                # Restore old diffs
                for d in old_diffs:
                    if 0 < d <= MAX_SCOPE:
                        word_idx = d // 64
                        bit_idx = d % 64
                        used_mask[word_idx] |= (1 << bit_idx)
                continue
            
            # Accept move
            row[pi] = new_val
            for d in new_diffs:
                if 0 < d <= MAX_SCOPE:
                    word_idx = d // 64
                    bit_idx = d % 64
                    used_mask[word_idx] |= (1 << bit_idx)
            
            new_scope = max(max(r) for r in best)
            if new_scope < best_scope:
                best_scope = new_scope
        
        return best, best_scope

    # Phase 1: Column-wise construction with many random restarts
    best_rows = None
    best_scope = MAX_SCOPE
    
    # Try deterministic column-wise first
    rng = random.Random(42)
    rows = _column_wise_construct(0, MAX_SCOPE, rng)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
    
    # Try randomized column-wise with various seeds and scopes
    for seed in range(60):
        if time.time() > deadline - 30:
            break
        
        rng = random.Random(1000 + seed)
        # Try progressively tighter scopes
        for scope_attempt in range(best_scope - 1, max(60, best_scope - 15), -1):
            rows = _column_wise_randomized(seed, scope_attempt, rng)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_rows = rows
                    best_scope = scope
                break  # Found a valid solution at this scope, try next seed
    
    # Phase 2: Local search to polish
    if best_rows is not None and time.time() < deadline - 10:
        rng = random.Random(777)
        polished, polished_scope = _local_search_bitmask(best_rows, deadline - 5, rng, iterations=30000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Phase 3: Backtracking at lower scopes
    if time.time() < deadline - 8:
        current_max = best_scope
        lower_bound = max(n_rows * k * (k + 1) // 2, 50)
        
        while time.time() < deadline - 5:
            target_scope = current_max - 1
            if target_scope < lower_bound or target_scope > MAX_SCOPE:
                break
            
            rng = random.Random(int(time.time() * 1000) % 100000)
            result = _backtrack_construct_bitmask(target_scope, n_rows, rng, deadline - 5)
            if result is not None:
                best_rows = result
                best_scope = target_scope
                current_max = target_scope
            else:
                current_max = target_scope
    
    # Phase 4: Final polish
    if best_rows is not None and time.time() < deadline - 3:
        rng = random.Random(999)
        polished, polished_scope = _local_search_bitmask(best_rows, deadline - 2, rng, iterations=10000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    if best_rows is None:
        # Emergency fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
    
    return best_rows
