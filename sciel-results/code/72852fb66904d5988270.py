import time
import random
import sys

def _generate_valid_rows(k, M):
    """
    Generate all Golomb rulers of length k+1 with span <= M.
    Returns a list of (row_tuple, diff_set).
    """
    L = k + 1
    rows = []
    used_diffs = set()
    current_row = [0] * L
    
    def backtrack(pos, last_val):
        if pos == L:
            rows.append((tuple(current_row), set(used_diffs)))
            return
        
        # Heuristic: if the remaining positions cannot be filled within M, prune
        # Minimal increment is 1.
        # So we need last_val + (L - pos) <= M
        if last_val + (L - pos) > M:
            return
            
        for v in range(last_val + 1, M + 1):
            # Check differences with all previous elements
            new_diffs = []
            valid = True
            for i in range(pos):
                d = v - current_row[i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                current_row[pos] = v
                used_diffs.update(new_diffs)
                backtrack(pos + 1, v)
                used_diffs.difference_update(new_diffs)

    backtrack(1, 0)
    return rows

def _find_clique(rows, K):
    """
    Find a clique of size K in the compatibility graph of rows.
    rows: list of (row_tuple, diff_set)
    Returns list of indices or None.
    """
    N = len(rows)
    if N < K:
        return None
    
    # Precompute compatibility bitsets
    compat_bits = [0] * N
    for i in range(N):
        di = rows[i][1]
        for j in range(i + 1, N):
            if not (di & rows[j][1]):
                compat_bits[i] |= (1 << j)
                compat_bits[j] |= (1 << i)
    
    # Search for clique of size K
    # For K=4, we can use nested loops with bitsets
    if K == 4:
        for i in range(N):
            ci = compat_bits[i]
            # Iterate j in ci
            j_mask = ci
            while j_mask:
                j_bit = j_mask & -j_mask
                jj = j_bit.bit_length() - 1
                j_mask -= j_bit
                if jj <= i:
                    continue
                
                common = ci & compat_bits[jj]
                # Remove i and jj
                common &= ~(1 << i)
                common &= ~(1 << jj)
                
                # Iterate k in common
                k_mask = common
                while k_mask:
                    k_bit = k_mask & -k_mask
                    kk = k_bit.bit_length() - 1
                    k_mask -= k_bit
                    
                    # Need l in common & compat[kk]
                    candidates_l = common & compat_bits[kk]
                    candidates_l &= ~(1 << i)
                    candidates_l &= ~(1 << jj)
                    candidates_l &= ~(1 << kk)
                    
                    if candidates_l:
                        ll = candidates_l.bit_length() - 1
                        return [i, jj, kk, ll]
                        
    # Generic backtracking for other K (not expected here)
    # ...
    
    return None

def _greedy_fallback(n_rows, k):
    """
    Simple greedy construction if clique search fails or times out.
    """
    rng = random.Random(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.05:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145 # Safety margin for 150s kill
    
    # Lower bound: n_rows * C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * diffs_per_row
    
    # Upper bound: Start with a safe guess, maybe from greedy
    # Greedy usually gives a decent upper bound
    greedy_rows = _greedy_fallback(n_rows, k)
    upper_bound = max(max(r) for r in greedy_rows)
    
    # We search for M in [lower_bound, upper_bound]
    # But we want to minimize M.
    # If we find a solution at M, it's optimal if M == lower_bound.
    # Otherwise, we might find a better one than greedy.
    
    best_solution = greedy_rows
    best_max = max(max(r) for r in greedy_rows)
    
    # Iterate M from lower_bound to best_max - 1
    # If we find a solution at M, update best.
    # If M reaches best_max, we can stop (greedy is already that good).
    
    for M in range(lower_bound, best_max):
        if time.time() > time_limit:
            break
        
        # Generate valid rows for span M
        # Note: This might be slow for large M or many rows.
        # We can add a timeout per M step.
        step_start = time.time()
        
        try:
            valid_rows = _generate_valid_rows(k, M)
        except Exception:
            break
            
        if time.time() - step_start > 10: # If generation takes too long, skip?
            # Actually, if it takes >10s, it's likely huge.
            # But we need the rows to check compatibility.
            pass
        
        if len(valid_rows) < n_rows:
            continue
        
        # Find clique
        clique = _find_clique(valid_rows, n_rows)
        
        if clique:
            # Extract the actual rows
            solution = [list(valid_rows[i][0]) for i in clique]
            current_max = max(max(r) for r in solution)
            if current_max < best_max:
                best_max = current_max
                best_solution = solution
                # If we hit the lower bound, we can't do better
                if current_max == lower_bound:
                    return best_solution
                    
    return best_solution
