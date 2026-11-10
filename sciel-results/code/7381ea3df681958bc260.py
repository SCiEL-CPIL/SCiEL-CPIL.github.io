import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit

    # We iterate W from 1 upwards.
    # For each W, we iterate L from 1 to LCM.
    # We check if there is a valid assignment of distinct multiples.
    
    # To speed up, we can break early if we find a solution for a smaller W.
    # Since we iterate W in increasing order, the first solution found is optimal.
    
    for W in range(1, 100):
        if time.time() - start_time > deadline:
            # Fallback to a valid solution (the divisors themselves)
            return [int(d) for d in divisors]
        
        for L in range(1, LCM + 1):
            if time.time() - start_time > deadline:
                return [int(d) for d in divisors]
            
            # Build options for each divisor in [L, L+W]
            options = []
            valid_window = True
            for d in divisors:
                # Find first multiple of d >= L
                k = (L + d - 1) // d
                first = k * d
                if first > L + W:
                    valid_window = False
                    break
                opts = []
                x = first
                while x <= L + W:
                    opts.append(x)
                    x += d
                options.append(opts)
            
            if not valid_window:
                continue
            
            # Check if any divisor has no options
            if any(len(opts) == 0 for opts in options):
                continue
            
            # Sort indices by number of options (most constrained first)
            # This helps backtracking prune faster
            order = sorted(range(n), key=lambda i: len(options[i]))
            
            # Backtracking search
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                for val in options[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        
                        # Forward checking: ensure all remaining unassigned divisors
                        # still have at least one available option
                        feasible = True
                        for j in range(idx + 1, n):
                            k_idx = order[j]
                            has_option = False
                            for v in options[k_idx]:
                                if v not in used:
                                    has_option = True
                                    break
                            if not has_option:
                                feasible = False
                                break
                        
                        if feasible and backtrack(idx + 1):
                            return True
                        
                        used.remove(val)
                        assignment[i] = None
            
            if backtrack(0):
                # Verify the solution
                # 1. Distinct
                if len(set(assignment)) == n:
                    # 2. Positive and <= 1000000
                    if all(0 < x <= 1000000 for x in assignment):
                        # 3. Divisibility
                        if all(assignment[i] % divisors[i] == 0 for i in range(n)):
                            return [int(x) for x in assignment]
    
    # Fallback: return divisors (valid but not optimal)
    return [int(d) for d in divisors]
