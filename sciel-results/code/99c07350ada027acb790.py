"""Difference triangle set (4,7): Optimized CSP solver using bitsets for difference tracking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver with strong pruning to minimize the maximum element (scope).
    """
    
    # Known best for (4,7) is 134. Counting bound is 112.
    # We will search downwards from 134.
    # We also generate a greedy solution as a fallback.
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
    
    # If greedy is better than 134, use it as the upper bound.
    # Otherwise, start from 134.
    start_scope = min(best_scope, 134)
    
    def solve_scope(scope, seed, max_nodes=1000000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
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
                
                # Generate candidates
                for x in range(min_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if (blocked >> d) & 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Heuristic: 
                # Try to pick candidates that leave the most "space" for future rows?
                # Or just fail-first.
                # Let's try a mix of greedy and random.
                if len(candidates) < 10:
                    # If few candidates, try them in a specific order
                    # Sort by "constraint tightness"? 
                    # For now, just pick one.
                    # To improve diversity, we can shuffle if seed is random.
                    if seed % 2 == 0:
                        x = candidates[0]
                    else:
                        x = rng.choice(candidates)
                else:
                    # Pick from a subset
                    # Bias towards smaller values to keep scope low
                    subset = candidates[:min(15, len(candidates))]
                    if seed % 3 == 0:
                        x = subset[0]
                    elif seed % 3 == 1:
                        x = subset[len(subset)//2]
                    else:
                        x = rng.choice(subset)
                
                new_diffs = 0
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                
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
    
    # Search downwards from start_scope to 112
    scopes_to_check = list(range(start_scope, 111, -1))
    
    for scope in scopes_to_check:
        if time.time() - t0 > time_limit:
            break
        
        # Time slice for this scope
        scope_start_time = time.time()
        # Allow up to 5 seconds per scope, or less if we are running out of total time
        remaining_total = time_limit - (time.time() - t0)
        # Reserve 5s for the final attempt or just use a fixed slice
        slice_time = min(4.0, remaining_total)
        
        if slice_time < 0.5:
            break
            
        found = None
        # Try seeds until time slice is up
        seed = 0
        while time.time() - scope_start_time < slice_time:
            # Dynamic budget based on time remaining in slice
            time_in_slice = time.time() - scope_start_time
            # Use a generous node budget, but the time check will stop it
            budget = 5000000 # High budget, time is the limiter
            
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
            seed += 1
            
            # If we just failed, try a different heuristic?
            # The current solve_scope uses seed to vary heuristics.
        
        if found:
            best_found = found
            best_scope_found = scope
            # We can continue to search for even smaller scopes
        else:
            # No solution found for this scope within the time slice.
            # We assume it might be infeasible or just hard.
            # We continue to the next smaller scope.
            pass

    return best_found
