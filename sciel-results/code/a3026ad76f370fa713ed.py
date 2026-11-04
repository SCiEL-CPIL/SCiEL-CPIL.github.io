import numpy as np

N = 50

def construct():
    # Find a Sidon set of size K
    # Greedy approach
    S = [0]
    used_sums = {0} # 2*0
    
    for k in range(1, 10): # Try to get up to 10
        found = False
        for i in range(1, N):
            if i in S:
                continue
            # Check if adding i creates duplicate sums
            new_sums = set()
            valid = True
            for j in S:
                s = i + j
                if s in used_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                S.append(i)
                used_sums.update(new_sums)
                found = True
                break
        
        if not found:
            break

    K = len(S)
    
    # Create vector
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0
        
    # Normalize to L1 = 1
    h = h / np.sum(h)
    
    return h.tolist()
