import time
import random
import copy

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            found = False
            for x in range(base + 1, 500):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _solve_with_backtracking(n_rows, k, max_entry, rng, deadline):
    """
    Backtracking solver with forward checking and dynamic variable ordering.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = [None] * n_rows
    
    def get_candidates(row, pos):
        """Get valid candidates for position pos in the given row."""
        if pos == 0:
            return [0]
        
        lo = row[pos - 1] + 1
        hi = max_entry
        candidates = []
        
        for val in range(lo, hi + 1):
            # Check strict increase
            if pos > 0 and val <= row[pos - 1]:
                continue
            if pos < k and row[pos + 1] is not None and val >= row[pos + 1]:
                continue
            
            # Check differences with existing elements
            new_diffs = set()
            valid = True
            for i in range(pos):
                d = val - row[i]
                if d <= 0:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Check against used_diffs
            if new_diffs & used_diffs:
                continue
            
            # Forward checking: ensure remaining positions can be filled
            if pos < k:
                remaining = k - pos
                # Need 'remaining' more values, each at least 1 apart
                if max_entry - val < remaining:
                    continue
                # Check if at least one valid next value exists
                has_next = False
                for next_val in range(val + 1, min(val + 1 + remaining, max_entry + 1)):
                    next_diffs = set()
                    next_valid = True
                    for i in range(pos + 1):
                        d = next_val - row[i] if i < pos else next_val - val
                        if d <= 0:
                            next_valid = False
                            break
                        if d in next_diffs:
                            next_valid = False
                            break
                        next_diffs.add(d)
                    if next_valid and not (next_diffs & (used_diffs | new_diffs)):
                        has_next = True
                        break
                if not has_next:
                    continue
            
            candidates.append(val)
        
        return candidates
    
    def solve(row_idx, pos):
        """Recursively solve row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        if pos > k:
            # Row complete
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 0)
        
        if pos == 0:
            # Start new row with 0
            rows[row_idx] = [0] + [None] * k
            used_diffs.clear()  # This is wrong, we need to track per-row
            # Actually, let's restructure
            
            # For simplicity, let's build rows sequentially
            row = [0] + [None] * k
            rows[row_idx] = row
            return solve(row_idx, 1)
        
        row = rows[row_idx]
        candidates = get_candidates(row, pos)
        
        if not candidates:
            return False
        
        # Order candidates: prefer smaller values
        candidates.sort()
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            row[pos] = val
            
            # Compute new diffs
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(val - row[i])
            
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1):
                return True
            
            row[pos] = None
            used_diffs.difference_update(new_diffs)
        
        return False
    
    # Build rows one by one
    for ri in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = [0] + [None] * k
        rows[ri] = row
        
        # Clear used_diffs for this row? No, it's global
        # Actually, we need to track all used diffs globally
        
        if not solve(ri, 1):
            return None
    
    # Verify the solution
    for row in rows:
        if None in row:
            return None
    
    return rows

def _local_search(rows, n_rows, k, rng, deadline, max_iter=5000):
    """
    Local search to reduce max entry.
    """
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(rows)
    
    # Precompute diffs for each row
    row_diffs = []
    for row in current_rows:
        dset = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                dset.add(row[j] - row[i])
        row_diffs.append(dset)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random row and position
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        
        # Bias toward positions with high values
        if rng.random() < 0.5:
            pi = rng.randrange(k // 2, k + 1)
        else:
            pi = rng.randrange(1, k + 1)
        
        old_val = row[pi]
        
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else best_score
        
        if lo > hi:
            continue
        
        # Try to reduce the value
        candidates = list(range(lo, min(hi + 1, old_val)))
        rng.shuffle(candidates)
        
        for new_val in candidates[:20]:
            if new_val == old_val:
                continue
            
            new_row = row[:]
            new_row[pi] = new_val
            
            # Check strict increase
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            
            # Compute new diffs
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            expected = (len(new_row) * (len(new_row) - 1)) // 2
            if len(new_diffs) != expected:
                continue
            
            # Check conflicts with other rows
            added = new_diffs - row_diffs[ri]
            conflict = False
            for oi in range(n_rows):
                if oi == ri:
                    continue
                if added & row_diffs[oi]:
                    conflict = True
                    break
            
            if not conflict:
                # Accept
                current_rows[ri] = new_row
                row_diffs[ri] = new_diffs
                
                new_score = _max_entry(current_rows)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = copy.deepcopy(current_rows)
                break
    
    return best_rows, best_score

def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin from 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get initial solution with greedy
    best_rows = _greedy_all(n_rows, k)
    if best_rows is None or not _validate(best_rows):
        # Fallback: very simple construction
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
    
    best_score = _max_entry(best_rows)
    
    # Phase 2: Try randomized greedy with different strategies
    for attempt in range(30):
        if time.time() > deadline - 60:
            break
        
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                base = row[-1]
                candidates = []
                for x in range(base + 1, min(base + 50, best_score + 1)):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                
                if not candidates:
                    for x in range(base + 1, best_score + 1):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            candidates.append(x)
                            break
                    if not candidates:
                        success = False
                        break
                
                pick = rng.choice(candidates[:min(8, len(candidates))])
                row.append(pick)
                used |= {pick - a for a in row[:-1]}
            
            if not success:
                break
            rows.append(row)
        
        if success and len(rows) == n_rows and _validate(rows):
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
    
    # Phase 3: Local search polish
    if time.time() < deadline - 10:
        rows, score = _local_search(
            best_rows, n_rows, k, rng, deadline - 5, max_iter=5000
        )
        if score < best_score and _validate(rows):
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Final validation
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k)
        if fresh_rows is not None and _validate(fresh_rows):
            best_rows = fresh_rows
        else:
            # Emergency fallback
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    for x in range(row[-1] + 1, 10000):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                best_rows.append(row)
    
    return best_rows
