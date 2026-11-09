import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # Run 5 iterations of the Greedy Dominating Set heuristic
    for run in range(5):
        S = []
        sums = set()
        available = list(range(1, N + 1))
        
        while available:
            best_x = -1
            best_cov = -1
            best_tie = 0
            
            # Evaluate candidates
            # To speed up, we can skip candidates that are "obviously" bad?
            # No, just iterate.
            
            for x in available:
                # Check if x compatible with S
                # New sums: 2x, x+s for s in S
                # Check uniqueness and against sums
                valid = True
                new_vals = [2*x]
                for s in S:
                    val = x + s
                    if val in sums or val in new_vals:
                        valid = False
                        break
                    new_vals.append(val)
                
                if not valid:
                    continue
                
                if 2*x in sums or 2*x in new_vals[1:]: # 2x checked against sums and other new vals?
                    # new_vals[0] is 2x. new_vals[1:] are x+s.
                    # We checked val in sums. We checked val in new_vals (which includes 2x).
                    # So if we got here, 2x is not in sums and not in x+s.
                    pass
                
                # Calculate coverage
                # How many y in available are blocked by x?
                cov = 0
                for y in available:
                    if y == x:
                        continue
                    # Check if S U {x, y} is Sidon
                    # We know S is Sidon. x is compatible with S.
                    # Check if y is compatible with S U {x}.
                    
                    # Sums to add for y: 2y, y+s for s in S, x+y
                    # Check these against:
                    # 1. sums (existing)
                    # 2. new_vals (sums from x)
                    # 3. each other
                    
                    # Quick check: 2y
                    if 2*y in sums or 2*y in new_vals:
                        cov += 1
                        continue
                        
                    # Check x+y
                    if x+y in sums or x+y in new_vals:
                        cov += 1
                        continue
                        
                    # Check y+s for s in S
                    blocked = False
                    for s in S:
                        val = y + s
                        if val in sums or val in new_vals:
                            blocked = True
                            break
                    if blocked:
                        cov += 1
                        continue
                        
                    # Check internal conflicts among {2y, x+y, y+s...}
                    # 2y vs x+y -> y==x (false)
                    # 2y vs y+s -> y==s (possible if s==y, but s in S, y in avail, disjoint)
                    # x+y vs y+s -> x==s (possible if s==x, but x not in S yet? No, x is candidate, not in S)
                    # Wait, x is NOT in S. S is the current set.
                    # So s != x.
                    # So x+y != y+s.
                    # 2y vs y+s2 -> y == s2.
                    
                    # If we passed the above, y is NOT blocked by x in context S.
                    # So cov does NOT increment.
                    
                # Tie-breaking
                if cov > best_cov or (cov == best_cov and (run % 2 == 0 and x < best_tie) or (run % 2 == 1 and x > best_tie)):
                    best_cov = cov
                    best_x = x
                    best_tie = x
            
            if best_x == -1:
                break
                
            # Add best_x
            S.append(best_x)
            sums.add(2*best_x)
            for s in S[:-1]:
                sums.add(best_x + s)
                
            available.remove(best_x)
            
            # Prune available
            new_available = []
            for y in available:
                valid = True
                new_vals_y = [2*y]
                for s in S:
                    val = y + s
                    if val in sums or val in new_vals_y:
                        valid = False
                        break
                    new_vals_y.append(val)
                if valid and 2*y not in sums:
                    new_available.append(y)
            available = new_available
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return best_set
