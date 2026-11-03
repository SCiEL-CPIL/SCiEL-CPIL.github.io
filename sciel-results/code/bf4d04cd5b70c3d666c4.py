import time
import random
import math

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _row_diff_mask(row):
    """Return a bitmask of all differences in the row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _build_row_greedy(k, used_mask, max_val, rng, spread=0):
    """
    Build a row [0, a1, ..., ak] greedily, ensuring no diff conflicts with used_mask.
    Uses bitmask for O(1) collision checks.
    """
    row = [0]
    for j in range(1, k + 1):
        best_val = None
        min_v = row[-1] + 1
        max_v = max_val - (k - j)
        
        if max_v < min_v:
            return None
        
        # Try values, preferring smaller ones for compactness
        if spread > 0 and j < k:
            # Add some spread
            start = min_v + rng.randint(0, spread)
        else:
            start = min_v
        
        found = False
        for v in range(start, max_v + 1):
            # Check if adding v creates any collision
            conflict = 0
            for a in row:
                d = v - a
                if d <= 0:
                    break
                conflict |= (1 << d)
            
            if (conflict & used_mask) == 0:
                row.append(v)
                found = True
                break
        
        if not found:
            return None
    
    # Final validation
    rm = _row_diff_mask(row)
    if rm & used_mask:
        return None
    
    return row

def _build_row_csp(k, used_mask, max_val, rng, time_limit=0.05):
    """
    Build a row using CSP with backtracking.
    """
    start = time.time()
    
    def backtrack(row, local_mask):
        if time.time() - start > time_limit:
            return None
        if len(row) == k + 1:
            return list(row)
        
        j = len(row)
        min_v = row[-1] + 1
        max_v = max_val - (k - j)
        
        if max_v < min_v:
            return None
        
        # Generate candidates
        if j < k and rng.random() < 0.3:
            # Sometimes try a random offset
            offset = rng.randint(0, max(0, max_v - min_v))
            candidates = list(range(min_v + offset, max_v + 1)) + list(range(min_v, min_v + offset))
            rng.shuffle(candidates)
        else:
            candidates = range(min_v, max_v + 1)
        
        for v in candidates:
            conflict = 0
            for a in row:
                d = v - a
                if d > 0:
                    conflict |= (1 << d)
            
            if (conflict & used_mask) == 0 and (conflict & local_mask) == 0:
                row.append(v)
                result = backtrack(row, local_mask | conflict)
                if result is not None:
                    return result
                row.pop()
        
        return None
    
    return backtrack([0], 0)

def _sa_optimize(rows, k, deadline, rng, T_start=5.0, cooling=0.9998, min_T=0.05):
    """
    Simulated annealing with bitmask collision tracking.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    best = [list(r) for r in current]
    best_scope = _scope(current)
    
    # Build global used mask
    used_mask = 0
    row_masks = []
    for r in current:
        m = _row_diff_mask(r)
        row_masks.append(m)
        used_mask |= m
    
    T = T_start
    it = 0
    
    while time.time() < deadline:
        it += 1
        if it % 200 == 0:
            T = max(T * cooling, min_T)
        
        move_type = rng.random()
        
        if move_type < 0.25:
            # Single element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current[i][j]
            
            lo = current[i][j-1] + 1
            hi = current[i][j+1] - 1 if j < k else best_scope
            
            if hi <= lo:
                continue
            
            # Bias toward decreasing
            if old_val > lo and rng.random() < 0.7:
                new_val = rng.randint(lo, old_val - 1)
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val:
                continue
            
            # Remove old row's contribution
            old_mask = row_masks[i]
            used_mask ^= old_mask  # Remove (XOR since we're toggling)
            # Actually need to recompute: used_mask without this row
            # Better: recompute used_mask from all other rows
            used_mask = 0
            for ri in range(n_rows):
                if ri != i:
                    used_mask |= row_masks[ri]
            
            # Try new value
            new_row = list(current[i])
            new_row[j] = new_val
            new_mask = _row_diff_mask(new_row)
            
            if (new_mask & used_mask) == 0:
                # Valid move
                row_masks[i] = new_mask
                used_mask |= new_mask
                current[i] = new_row
                new_scope = _scope(current)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
            else:
                # Collision - accept with probability
                # Count collisions
                colliding = new_mask & used_mask
                n_coll = bin(colliding).count('1')
                penalty = n_coll * 10
                new_scope = max(_scope(current), new_val)
                
                if rng.random() < math.exp(-penalty / max(T, min_T)):
                    row_masks[i] = new_mask
                    used_mask = 0
                    for ri in range(n_rows):
                        used_mask |= row_masks[ri]
                    current[i] = new_row
                    new_scope = _scope(current)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [list(r) for r in current]
                else:
                    # Revert - nothing to do since we didn't change
                    pass
        
        elif move_type < 0.45:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = list(current[i])
            old_mask = row_masks[i]
            
            # Remove row from used_mask
            used_mask = 0
            for ri in range(n_rows):
                if ri != i:
                    used_mask |= row_masks[ri]
            
            # Try to rebuild
            new_row = _build_row_csp(k, used_mask, best_scope, rng, 0.03)
            if new_row is None:
                new_row = _build_row_greedy(k, used_mask, best_scope, rng)
            
            if new_row is not None:
                new_mask = _row_diff_mask(new_row)
                if (new_mask & used_mask) == 0:
                    row_masks[i] = new_mask
                    used_mask |= new_mask
                    current[i] = new_row
                    new_scope = _scope(current)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [list(r) for r in current]
        
        elif move_type < 0.6:
            # Target the maximum element directly
            max_val = best_scope
            max_ri = -1
            max_rj = -1
            for ri in range(n_rows):
                for rj in range(k, 0, -1):
                    if current[ri][rj] == max_val:
                        max_ri = ri
                        max_rj = rj
                        break
                if max_ri >= 0:
                    break
            
            if max_ri < 0:
                continue
            
            i = max_ri
            j = max_rj
            old_val = current[i][j]
            prev_val = current[i][j-1]
            
            if old_val <= prev_val + 1:
                continue
            
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            used_mask = 0
            for ri in range(n_rows):
                if ri != i:
                    used_mask |= row_masks[ri]
            
            new_row = list(current[i])
            new_row[j] = new_val
            new_mask = _row_diff_mask(new_row)
            
            if (new_mask & used_mask) == 0:
                row_masks[i] = new_mask
                used_mask |= new_mask
                current[i] = new_row
                new_scope = _scope(current)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
        
        else:
            # Swap two elements in same row (maintaining order)
            i = rng.randint(0, n_rows - 1)
            j1 = rng.randint(1, k)
            j2 = rng.randint(1, k)
            if j1 == j2:
                continue
            if j1 > j2:
                j1, j2 = j2, j1
            
            # Try to adjust element at j2 downward
            old_val = current[i][j2]
            lo = current[i][j2-1] + 1
            hi = current[i][j2+1] - 1 if j2 < k else best_scope
            
            if hi <= lo:
                continue
            
            new_val = rng.randint(lo, min(old_val, hi))
            if new_val == old_val:
                continue
            
            used_mask = 0
            for ri in range(n_rows):
                if ri != i:
                    used_mask |= row_masks[ri]
            
            new_row = list(current[i])
            new_row[j2] = new_val
            new_mask = _row_diff_mask(new_row)
            
            if (new_mask & used_mask) == 0:
                row_masks[i] = new_mask
                used_mask |= new_mask
                current[i] = new_row
                new_scope = _scope(current)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _fast_greedy(k, n_rows, rng, spread_factor=0):
    """Fast greedy initialization using bitmask."""
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        for j in range(1, k + 1):
            x = row[-1] + 1
            while True:
                conflict = 0
                for a in row:
                    d = x - a
                    if d > 0:
                        conflict |= (1 << d)
                if (conflict & used_mask) == 0:
                    row.append(x)
                    break
                x += 1
        used_mask |= _row_diff_mask(row)
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple diverse initializations
    init_deadline = start_time + 8.0
    candidates = []
    
    for trial in range(40):
        if time.time() > init_deadline:
            break
        
        spread = rng.randint(0, 5)
        rows = _fast_greedy(k, n_rows, rng, spread)
        
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    # Also try with different seeds
    for seed in [123, 456, 789, 101]:
        if time.time() > init_deadline:
            break
        rng2 = random.Random(seed)
        rows = _fast_greedy(k, n_rows, rng2)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if not candidates:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: CSP-based row reconstruction to tighten
    if best_rows and time.time() < deadline - 40:
        csp_rows = [list(r) for r in best_rows]
        csp_scope = _scope(csp_rows)
        
        improved = True
        rounds = 0
        while improved and time.time() < deadline - 40 and rounds < 3:
            improved = False
            rounds += 1
            for i in range(n_rows):
                if time.time() > deadline - 40:
                    break
                
                # Build used_mask from all other rows
                used_mask = 0
                for ri in range(n_rows):
                    if ri != i:
                        used_mask |= _row_diff_mask(csp_rows[ri])
                
                old_row = list(csp_rows[i])
                
                # Try to build a tighter row
                new_row = _build_row_csp(k, used_mask, csp_scope, rng, 0.1)
                if new_row is None:
                    new_row = _build_row_greedy(k, used_mask, csp_scope, rng)
                
                if new_row is not None:
                    new_mask = _row_diff_mask(new_row)
                    if (new_mask & used_mask) == 0:
                        csp_rows[i] = new_row
                        new_scope = _scope(csp_rows)
                        if new_scope < csp_scope:
                            csp_scope = new_scope
                            improved = True
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in csp_rows]
        
        if _valid(csp_rows, k):
            actual = _scope(csp_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = [list(r) for r in csp_rows]
    
    # Phase 3: SA optimization on top candidates
    remaining = deadline - time.time()
    top_n = min(3, len(candidates))
    
    for idx in range(top_n):
        if time.time() > deadline - 15:
            break
        
        s, rows = candidates[idx]
        sa_deadline = time.time() + min(remaining / max(1, top_n) * 0.85, 30)
        
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng, T_start=4.0, cooling=0.9997, min_T=0.03)
        
        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 4: Final local polish
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(12, deadline - time.time() - 5)
        polish_scope, polish_rows = _sa_optimize(
            best_rows, k, polish_deadline, rng,
            T_start=1.5, cooling=0.9999, min_T=0.01
        )
        
        if polish_rows is not None and _valid(polish_rows, k):
            actual = _scope(polish_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = polish_rows
    
    # Final validation
    if best_rows is not None:
        if not _valid(best_rows, k):
            # Fallback to first valid candidate
            for s, rows in candidates:
                if _valid(rows, k):
                    return rows
            # Last resort
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    x = len(used) + j + 1
                    while any((x - row[i]) in used for i in range(len(row))):
                        x += 1
                    for i in range(len(row)):
                        used.add(x - row[i])
                    row.append(x)
                rows.append(row)
            return rows
    
    return best_rows
