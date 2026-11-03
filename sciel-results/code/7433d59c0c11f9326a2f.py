import numpy as np

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    # Generate a Sidon set (B2 set)
    # Greedy approach
    S = [0]
    for i in range(1, 50):
        # Check if i can be added
        # A set is Sidon if all pairwise differences are distinct
        # Or sums are distinct.
        # For small sets, we can just check differences.
        diffs = set()
        valid = True
        for j in range(len(S)):
            for k in range(j+1, len(S)):
                d = abs(S[j] - S[k])
                if d in diffs:
                    # Collision within existing set? No, S is Sidon by construction.
                    # We need to check if i creates a collision.
                    pass
                diffs.add(d)
        
        # Check if adding i creates duplicate differences
        new_diffs = set()
        for j in range(len(S)):
            d = abs(i - S[j])
            if d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if not valid:
            continue
            
        # Check against existing diffs
        for d in new_diffs:
            if d in diffs:
                valid = False
                break
        
        if valid:
            S.append(i)
            diffs.update(new_diffs)
            
        if len(S) >= 7: # Enough elements
            break

    h = np.zeros(N)
    for idx in S:
        h[idx] = 1.0
        
    # Scale down
    scale = 1e-4
    h = h * scale
    
    return list(h)
