"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    # Number of distinct positive differences required per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # ---------- helpers ----------
    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs(row):
        out = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                out.append(row[j] - row[i])
        return out

    def pure_greedy():
        """Fast baseline: greedy construction, always valid."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def backtrack_row(row_idx, max_scope, used_mask, max_diff, deadline_check):
        """
        Backtrack to construct a single row given the current used difference mask.
        Returns the completed row or None if no solution within max_scope.
        used_mask: bit i is set if difference i is already used.
        max_diff: maximum difference value we care about (max_scope)
        """
        if row_idx >= n_rows:
            return [0] * n_rows  # all rows done (placeholder)
        
        row = [0]
        max_len = k + 1
        
        def _build(pos, last_val):
            if time.time() > deadline_check:
                return None
            if pos == max_len:
                return row[:]
            
            # Try candidates from last_val+1 up to max_scope
            # Use randomized order for diversity
            start = last_val + 1
            end = max_scope
            if start > end:
                return None
            
            # Generate candidates in random order for exploration
            cands = list(range(start, end + 1))
            random.shuffle(cands)
            
            # Limit candidates to improve speed
            if len(cands) > 200:
                cands = cands[:200]
            
            for cand in cands:
                # Compute new differences
                new_diffs = [cand - row[i] for i in range(pos)]
                
                # Check internal uniqueness and collision with used_mask
                local_mask = 0
                valid = True
                for d in new_diffs:
                    if d < 1 or d > max_diff:
                        valid = False
                        break
                    bit = 1 << d
                    if local_mask & bit:
                        valid = False
                        break
                    local_mask |= bit
                
                if not valid:
                    continue
                
                # Check collision with global used_mask
                if local_mask & used_mask:
                    continue
                
                # Accept
                row.append(cand)
                new_used = used_mask | local_mask
                
                result = _build(pos + 1, cand)
                if result is not None:
                    return result
                
                row.pop()
            
            return None
        
        return _build(1, 0)

    def construct_all_rows(max_scope, deadline_check):
        """
        Try to construct all n_rows with max entry <= max_scope.
        Returns list of rows or None.
        """
        max_diff = max_scope
        # Use a large enough bit representation
        # We'll use a set-based approach for clarity, with bitmask for speed
        used = set()
        rows = []
        
        def _build_row(row_idx):
            if time.time() > deadline_check:
                return None
            if row_idx == n_rows:
                return rows[:]
            
            row = [0]
            max_len = k + 1
            
            def _fill(pos, last_val):
                if time.time() > deadline_check:
                    return None
                if pos == max_len:
                    rows.append(row[:])
                    result = _build_row(row_idx + 1)
                    if result is not None:
                        return result
                    rows.pop()
                    return None
                
                start = last_val + 1
                end = max_scope
                if start > end:
                    return None
                
                # Randomized candidates
                cands = list(range(start, end + 1))
                random.shuffle(cands)
                if len(cands) > 150:
                    cands = cands[:150]
                
                for cand in cands:
                    new_diffs = [cand - row[i] for i in range(pos)]
                    
                    # Check internal uniqueness
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    
                    # Check collision with used
                    if any(d in used for d in new_diffs):
                        continue
                    
                    # Accept
                    row.append(cand)
                    for d in new_diffs:
                        used.add(d)
                    
                    result = _fill(pos + 1, cand)
                    if result is not None:
                        return result
                    
                    row.pop()
                    for d in new_diffs:
                        used.discard(d)
                
                return None
            
            return _fill(1, 0)
        
        return _build_row(0)

    def local_search(rows, max_iters, time_limit):
        """Perturb single entries to reduce scope while maintaining validity."""
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])
        
        def build_used_set(rows):
            used = set()
            for row in rows:
                for d in row_diffs(row):
                    used.add(d)
            return used
        
        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            ri = random.randrange(n)
            if L <= 1:
                break
            ci = random.randrange(1, L)
            
            old_val = rows[ri][ci]
            old_row = rows[ri]
            
            # Candidate values
            candidates = set()
            for d in range(-8, 9):
                nv = old_val + d
                if nv >= 1 and nv <= best_scope:
                    candidates.add(nv)
            for _ in range(6):
                nv = random.randint(1, best_scope)
                candidates.add(nv)
            
            lower = old_row[ci - 1] + 1
            upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope
            candidates = [c for c in candidates if lower <= c <= upper and c != old_val]
            if not candidates:
                continue
            
            random.shuffle(candidates)
            for nv in candidates:
                new_row = old_row[:]
                new_row[ci] = nv
                new_diffs = row_diffs(new_row)
                
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Check collisions with other rows
                collides = False
                for oi in range(n):
                    if oi == ri:
                        continue
                    od = set(row_diffs(rows[oi]))
                    if any(d in od for d in new_diffs):
                        collides = True
                        break
                if collides:
                    continue
                
                new_scope = max(max(new_row), max(max(rows[i]) for i in range(n) if i != ri))
                if new_scope < best_scope:
                    rows[ri] = new_row
                    best = [r[:] for r in rows]
                    best_scope = new_scope
                break
        
        return best, best_scope

    # ---------- main search ----------
    # Phase 0: Get a valid baseline
    base_rows = pure_greedy()
    best_rows = base_rows
    best_scope = scope_of(base_rows)
    
    # Phase 1: Try backtracking with decreasing max_scope
    # Start from a reasonable upper bound
    upper = best_scope
    lower = 112  # counting bound
    
    # Binary search on scope
    while time.time() < deadline - 20.0:
        mid = (lower + upper + 1) // 2
        if mid <= lower:
            break
        
        # Try to construct with max_scope = mid
        result = construct_all_rows(mid, deadline_check=deadline - 20.0)
        if result is not None:
            sc = scope_of(result)
            if sc < best_scope:
                best_rows = result
                best_scope = sc
            upper = mid - 1
        else:
            lower = mid + 1
        
        if upper < lower:
            break
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 5.0:
        improved, sc = local_search(best_rows, max_iters=3000, time_limit=deadline - 5.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc
    
    return best_rows
# EVOLVE-BLOCK-END
