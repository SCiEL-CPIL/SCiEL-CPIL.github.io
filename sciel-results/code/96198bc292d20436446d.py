import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Search for minimum range R
    for R in range(1, 200):
        # Search for starting position M
        for M in range(1, 1001):
            domains = []
            valid_window = True
            for d in divisors:
                k_start = (M + d - 1) // d
                k_end = (M + R) // d
                mults = []
                for k in range(k_start, k_end + 1):
                    val = k * d
                    if M <= val <= M + R:
                        mults.append(val)
                if not mults:
                    valid_window = False
                    break
                domains.append(mults)
            
            if not valid_window:
                continue
            
            # Custom CSP Solver with Forward Checking and MRV
            def backtrack(domains, assignment):
                if len(assignment) == n:
                    return assignment
                
                # MRV Heuristic: Select unassigned variable with smallest domain
                min_size = float('inf')
                var_idx = -1
                for i in range(n):
                    if i not in assignment:
                        if len(domains[i]) < min_size:
                            min_size = len(domains[i])
                            var_idx = i
                
                if var_idx == -1:
                    return None
                if min_size == 0:
                    return None
                
                # Try each value in the selected variable's domain
                for val in domains[var_idx]:
                    if val in assignment.values():
                        continue
                    
                    new_assignment = dict(assignment)
                    new_assignment[var_idx] = val
                    
                    # Forward Checking
                    valid = True
                    new_domains = [None] * n
                    for i in range(n):
                        if i == var_idx:
                            new_domains[i] = domains[i]
                        elif i in new_assignment:
                            new_domains[i] = domains[i]
                        else:
                            if val in domains[i]:
                                new_d = [x for x in domains[i] if x != val]
                                if not new_d:
                                    valid = False
                                    break
                                new_domains[i] = new_d
                            else:
                                new_domains[i] = domains[i]
                    
                    if valid:
                        result = backtrack(new_domains, new_assignment)
                        if result:
                            return result
                return None

            sol = backtrack(domains, {})
            if sol:
                final_sol = [0] * n
                for i, val in sol.items():
                    final_sol[i] = val
                if len(set(final_sol)) == n and all(1 <= x <= 1000000 for x in final_sol):
                    return final_sol[:]

    return divisors[:]
