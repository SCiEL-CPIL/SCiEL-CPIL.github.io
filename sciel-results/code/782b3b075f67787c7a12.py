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
    
    def solve_scope(scope, seed, max_nodes=1000000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        # Precompute mask for valid values 1..scope
        # Bit i is 1 if i is a valid value
        valid_values_mask = (1 << (scope + 1)) - 1
        valid_values_mask ^= 1 # Remove 0
        
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
                
                # Calculate bad candidates using bitset shifts
                # A candidate x is bad if (x - a) is in blocked for any a in row
                # x is bad if x is in (blocked + a) for any a in row
                bad_mask = 0
                for a in row:
                    bad_mask |= (blocked << a)
                
                # Candidates must be in [min_x, max_x]
                # Mask for range [min_x, max_x]
                range_mask = ((1 << (max_x + 1)) - 1) ^ ((1 << min_x) - 1)
                
                candidates_mask = valid_values_mask & range_mask & ~bad_mask
                
                if candidates_mask == 0:
                    return False

                # Extract candidates from mask
                # We want to try them in a specific order.
                # Heuristic: Smallest first to minimize scope.
                # Or, if seed varies, we can shuffle.
                
                # Get list of candidates
                candidates = []
                m = candidates_mask
                while m:
                    # Get lowest set bit
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    candidates.append(x)
                    m ^= lsb
                
                if not candidates:
                    return False

                # Heuristic: 
                # If seed is even, pick smallest (greedy for scope)
                # If seed is odd, pick random from first few
                if seed % 2 == 0:
                    x = candidates[0]
                else:
                    # Pick from a small subset to add diversity
                    subset_size = min(5, len(candidates))
                    x = rng.choice(candidates[:subset_size])
                
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
    
    # Improved search strategy:
    # Scan downwards from best_scope - 1.
    # Use time slicing.
    
    scopes_to_check = list(range(best_scope - 1, 111, -1))
    
    for scope in scopes_to_check:
        if time.time() - t0 > time_limit:
            break
        
        # Time slice for this scope
        scope_start_time = time.time()
        remaining_total = time_limit - (time.time() - t0)
        # Allow up to 3 seconds per scope to try more scopes
        slice_time = min(3.0, remaining_total)
        
        if slice_time < 0.5:
            break
            
        found = None
        seed = 0
        while time.time() - scope_start_time < slice_time:
            budget = 5000000 # High budget, time is the limiter
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
