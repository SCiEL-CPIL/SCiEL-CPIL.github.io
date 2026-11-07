import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    start_time = time.time()
    timeout = 50  # Safety margin for 60s limit
    
    # Heuristic initial solution: the divisors themselves
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    # Span: 39 - 6 = 33
    best_sol = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    best_span = 33

    # We search for smaller spans
    # The span must be at least 1 (distinct integers)
    # We iterate span L from 1 to best_span - 1
    for L in range(1, best_span):
        if time.time() - start_time > timeout:
            break
        
        # Iterate possible minimum m
        # We can bound m. The values are <= 1000000.
        # If m is very large, the density of multiples is the same.
        # However, due to periodicity of the pattern of available slots,
        # checking m up to a certain limit (like LCM of divisors or a multiple thereof)
        # might be sufficient to find the optimal pattern, but the absolute value
        # of m doesn't change the relative positions modulo LCM.
        # Actually, the set of available residues for each d_i modulo LCM is fixed.
        # We can just check m from 1 to some reasonable bound, e.g., 1000.
        # If the optimal span is small, the pattern repeats every LCM(6,10,14,15,21,22,26,33,35,39).
        # LCM is large, but the structure of "gaps" repeats.
        # Let's just check m from 1 to 500. If no solution found, try next L.
        
        found = False
        for m in range(1, 500):
            if time.time() - start_time > timeout:
                break
            
            # Check if a valid assignment exists in [m, m+L]
            # Precompute options for each divisor i
            options = []
            for i, d in enumerate(divisors):
                # Find multiples of d in [m, m+L]
                # First multiple >= m
                first = ((m + d - 1) // d) * d
                opts = []
                val = first
                while val <= m + L:
                    opts.append(val)
                    val += d
                options.append(opts)
            
            # Sort divisors by number of options (ascending) to prune faster
            # We need to keep track of original indices
            indices = sorted(range(len(divisors)), key=lambda i: len(options[i]))
            
            used_values = set()
            assignment = [0] * len(divisors)
            
            # Backtracking
            def backtrack(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for val in options[i]:
                    if val not in used_values:
                        used_values.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(val)
                return False
            
            if backtrack(0):
                # Found a solution with span L
                best_sol = assignment[:]
                best_span = L
                found = True
                break # Found for this L, so we can stop searching for this L
        
        if found:
            # Since we iterate L from 1 upwards, the first found is optimal
            # (assuming we check all m sufficiently)
            # We should break the outer loop as well
            break
            
    # Return the best solution found
    # Ensure types are Python int
    return [int(x) for x in best_sol]
