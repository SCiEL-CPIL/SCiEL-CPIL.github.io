import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    
    # Multiple restarts to find a large set
    for restart in range(100):
        if time.time() > deadline:
            break
            
        S = []
        # Randomize the candidate order to escape local optima
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # Heuristic: try to pick numbers that are spread out
        # Sort by distance from 256 (center) in descending order?
        # Or just random.
        
        dp_S = [0] * (50)  # dp_S[k] is bitset of sums of k elements from S
        dp_S[0] = 1
        
        while len(S) < 48:
            if time.time() > deadline:
                break
            
            n = len(S)
            
            # Precompute dp for S \ {a} for all a in S
            # This is expensive, so we only do it if we have candidates to test
            # Actually, we can just test candidates and compute on the fly if needed, 
            # but precomputing for the current S is better if we test many candidates.
            
            dp_minus = {}
            for a in S:
                # Compute dp for S \ {a}
                temp_S = [x for x in S if x != a]
                dp_temp = [0] * (n) # size n-1 max
                dp_temp[0] = 1
                for elem in temp_S:
                    for k in range(len(temp_S) - 1, -1, -1):
                        if dp_temp[k]:
                            dp_temp[k+1] |= (dp_temp[k] << elem)
                dp_minus[a] = dp_temp
            
            found = False
            # Try candidates
            for x in candidates:
                if time.time() > deadline:
                    break
                
                if x in S:
                    continue
                
                # Check 1: Is x the mean of a subset of S?
                valid = True
                # dp_S[k] contains sums of k elements from S
                # We need sum = k * x
                for k in range(2, n + 1):
                    target = k * x
                    if target > 512 * 48: # max possible sum
                        break
                    if (dp_S[k] >> target) & 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check 2: Is any a in S the mean of a subset of S U {x} \ {a}?
                for a in S:
                    # Subsets must contain x.
                    # T = {x} U U, U subset of S \ {a}, |U| = j >= 1
                    # Sum(T) = x + Sum(U) = (j+1) * a
                    # Sum(U) = (j+1) * a - x
                    dp_a = dp_minus[a]
                    # j ranges from 1 to n-1
                    for j in range(1, n):
                        target = (j + 1) * a - x
                        if target < 0:
                            continue
                        if target > 512 * 48:
                            break
                        # Check if target is in dp_a[j]
                        if (dp_a[j] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Add x to S
                    S.append(x)
                    # Update dp_S
                    for k in range(n, -1, -1):
                        if dp_S[k]:
                            dp_S[k+1] |= (dp_S[k] << x)
                    # Remove x from candidates list to speed up future iterations?
                    # candidates.remove(x) is O(N). 
                    # Just skip it in future iterations.
                    found = True
                    break
            
            if not found:
                # Stuck, break inner loop and restart
                break
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                break
    
    return best_set
