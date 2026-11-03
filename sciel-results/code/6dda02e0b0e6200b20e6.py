"""Difference triangle set (4,7): Optimized CSP solver with forward-checking and better heuristics."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver with forward-checking and strong pruning.
    """
    # Greedy baseline
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    def solve_scope(scope, seed, max_nodes=500000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
        # Precompute: for a row of length k+1, the number of differences is k*(k+1)/2
        num_diffs_per_row = k * (k + 1) // 2
        total_diffs_needed = n_rows * num_diffs_per_row
        
        def build_row(idx):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            # Symmetry breaking: 
            # We assume rows are sorted by their second element (index 1).
            # This is a valid symmetry breaking for DTS.
            if idx == 0:
                min_start = 1
            else:
                min_start = rows[idx - 1][1] + 1
            
            # Forward-checking: count how many differences are already used
            used_count = bin(used_bits).count('1')
            remaining_rows = n_rows - idx - 1
            remaining_diffs_needed = total_diffs_needed - used_count - num_diffs_per_row
            
            # If remaining differences needed exceed available values, prune
            available_values = scope - used_count
            if remaining_diffs_needed > available_values:
                return False
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    # Check conflict
                    if used_bits & row_diff_bits:
                        return False
                    
                    used_bits |= row_diff_bits
                    rows.append(row)
                    if build_row(idx + 1):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

                remaining = (k + 1) - len(row)
                max_x = scope - (remaining - 1)
                
                if max_x < min_x:
                    return False

                blocked = used_bits | row_diff_bits
                candidates = []
                
                # Generate candidates with forward-checking
                for x in range(min_x, max_x + 1):
                    ok = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (blocked >> d) & 1:
                            ok = False
                            break
                        new_diffs |= (1 << d)
                    if ok:
                        # Additional forward-check: ensure we don't block too many future values
                        # Count how many new differences this would use
                        candidates.append((x, new_diffs, bin(new_diffs).count('1')))
                
                if not candidates:
                    return False
                
                # Heuristic ordering: prefer candidates that:
                # 1. Use fewer "small" differences (to leave room for other rows)
                # 2. Leave more flexibility
                # 3. Are smaller (to keep scope low)
                
                def score_candidate(cand):
                    x, new_diffs, diff_count = cand
                    # Score based on how "central" the differences are
                    # Prefer candidates whose differences are spread out
                    # and don't cluster in small values
                    small_diffs = 0
                    for d in range(1, min(20, scope + 1)):
                        if (new_diffs >> d) & 1:
                            small_diffs += 1
                    # Prefer fewer small differences
                    return (small_diffs, x)
                
                candidates.sort(key=score_candidate)
                
                # Try candidates in order, with some randomization based on seed
                if seed % 5 == 0:
                    # Greedy: try first few
                    try_list = candidates[:min(5, len(candidates))]
                elif seed % 5 == 1:
                    # Middle range
                    mid = len(candidates) // 2
                    try_list = candidates[max(0, mid-2):mid+3]
                elif seed % 5 == 2:
                    # Larger values
                    try_list = candidates[-min(5, len(candidates)):]
                else:
                    # Random selection
                    try_list = [candidates[rng.randint(0, len(candidates)-1)]]
                
                for x, new_diffs, diff_count in try_list:
                    if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                        return True
                
                return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    # Lower bound from counting: for DTS(n,k), m >= n*k*(k+1)/2
    # For (4,7): 4*7*8/2 = 112
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Start from best_scope - 1 and go down to lower_bound
    # Use adaptive time allocation: give more time to promising scopes
    scopes_to_check = list(range(min(best_scope - 1, lower_bound + 25), lower_bound - 1, -1))
    
    # If we have a wide range, prioritize middle values first (likely to be feasible)
    if len(scopes_to_check) > 15:
        mid = len(scopes_to_check) // 2
        # Interleave: try middle, then expand outward
        ordered_scopes = []
        for offset in range(mid + 1):
            if mid - offset >= 0:
                ordered_scopes.append(scopes_to_check[mid - offset])
            if mid + offset < len(scopes_to_check) and mid + offset != mid - offset:
                ordered_scopes.append(scopes_to_check[mid + offset])
        scopes_to_check = ordered_scopes
    
    for scope in scopes_to_check:
        if time.time() - t0 > time_limit:
            break
        
        # Time slice for this scope
        scope_start_time = time.time()
        remaining_total = time_limit - (time.time() - t0)
        
        # Allocate more time to scopes closer to the lower bound (more promising)
        distance_from_lb = scope - lower_bound
        if distance_from_lb <= 5:
            slice_time = min(6.0, remaining_total)
        elif distance_from_lb <= 10:
            slice_time = min(4.0, remaining_total)
        else:
            slice_time = min(2.5, remaining_total)
        
        if slice_time < 0.3:
            break
            
        found = None
        seed = 0
        while time.time() - scope_start_time < slice_time:
            # Use a reasonable node budget
            budget = 300000
            
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
            seed += 1
        
        if found:
            best_found = found
            best_scope_found = scope
        else:
            pass

    return best_found
