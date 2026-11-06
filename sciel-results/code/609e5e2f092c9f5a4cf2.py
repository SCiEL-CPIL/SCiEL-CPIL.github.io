import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        S = []
        S_set = set()
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        max_sum = 512 * 48
        dp_S = [0] * 49
        dp_S[0] = 1
        
        dp_minus = {}
        
        while len(S) < 48:
            if time.time() > deadline:
                break
            
            n = len(S)
            
            found = False
            # Iterate through candidates
            # We can filter candidates not in S_set
            for x in candidates:
                if x in S_set:
                    continue
                
                valid = True
                
                # Check 1: x is mean of subset of S
                for k in range(2, n + 1):
                    target = k * x
                    if target > max_sum:
                        break
                    if (dp_S[k] >> target) & 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check 2: y in S is mean of subset of S U {x} \ {y}
                for y in S:
                    dp_y = dp_minus[y]
                    for j in range(1, n):
                        target = (j + 1) * y - x
                        if target < 0:
                            continue
                        if target > max_sum:
                            break
                        if (dp_y[j] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Save old dp_S for dp_minus[x]
                    old_dp_S = dp_S[:]
                    
                    # Update dp_S with x
                    for k in range(n, -1, -1):
                        if dp_S[k]:
                            dp_S[k+1] |= (dp_S[k] << x)
                            
                    # Update dp_minus for existing y
                    for y in S:
                        dp_y = dp_minus[y]
                        for k in range(n - 1, -1, -1):
                            if dp_y[k]:
                                dp_y[k+1] |= (dp_y[k] << x)
                                
                    # Set dp_minus[x] to old_dp_S
                    dp_minus[x] = old_dp_S
                    
                    S.append(x)
                    S_set.add(x)
                    found = True
                    break
            
            if not found:
                break
                
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                break
                
    return best_set
