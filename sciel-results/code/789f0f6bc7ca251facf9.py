"""Difference triangle set (4,7): Optimized CSP solver with strong symmetry breaking and pruning."""
import random
import sys
import time
import numpy as np

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a backtracking solver with:
    1. Strong symmetry breaking (rows sorted lexicographically).
    2. Counting bound pruning (available differences vs needed differences).
    """
    
    # Greedy baseline
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    diffs_per_row = k * (k + 1) // 2

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
            
            # Pruning: Counting bound
            used_count = bin(used_bits).count('1')
            needed = (n_rows - idx) * diffs_per_row
            available = scope - used_count
            if available < needed:
                return False

            row = [0]
            row_diff_bits = 0
            
            # Symmetry breaking: 
            # Rows are sorted lexicographically.
            # We enforce this step-by-step.
            
            min_start = 1
            prev_row = None
            if idx > 0:
                prev_row = rows[idx - 1]
                min_start = prev_row[1]
            
            def rec(row, row_diff_bits, min_x, lexi_prefix_equal):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    # Check conflict
                    if used_bits & row_diff_bits:
                        return False
                    
                    # Final symmetry check: if prefix was equal, row must be > prev_row
                    # But we enforced > at the first differing step, so if we reach here,
                    # either it was already >, or it is equal (which is invalid if idx>0 and we didn't break).
                    # Actually, if lexi_prefix_equal is True at the end, it means row == prev_row, which is invalid.
                    if idx > 0 and lexi_prefix_equal:
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
                
                # Heuristic ordering
                if len(candidates) < 10:
                    if seed % 2 == 0:
                        x = candidates[0]
                    else:
                        x = rng.choice(candidates)
                else:
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
                
                # Update lexicographic state
                new_lexi_equal = lexi_prefix_equal
                if idx > 0 and lexi_prefix_equal:
                    if x < prev_row[len(row)]:
                        return False # Must be >=
                    elif x > prev_row[len(row)]:
                        new_lexi_equal = False # Now strictly greater
                    else:
                        new_lexi_equal = True # Still equal

                if rec(row + [x], row_diff_bits | new_diffs, x + 1, new_lexi_equal):
                    return True
                return False
            
            return rec(row, row_diff_bits, min_start, True)
        
        success = build_row(0)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    scopes_to_check = list(range(best_scope - 1, 111, -1))
    
    for scope in scopes_to_check:
        if time.time() - t0 > time_limit:
            break
        
        scope_start_time = time.time()
        remaining_total = time_limit - (time.time() - t0)
        slice_time = min(4.0, remaining_total)
        
        if slice_time < 0.5:
            break
            
        found = None
        seed = 0
        while time.time() - scope_start_time < slice_time:
            budget = 5000000
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
            seed += 1
            
        if found:
            best_found = found
            best_scope_found = scope

    return best_found
