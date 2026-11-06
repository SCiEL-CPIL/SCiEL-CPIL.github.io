import time

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    p = 1009
    
    # Map each non-zero quadratic residue to a valid root
    # We store the smallest positive root for determinism, though any valid root works
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
    
    # Build the Legendre symbol sign sequence e_x:
    # e_x = 1 if x is a non-zero quadratic residue
    # e_x = -1 if x is a non-zero quadratic non-residue
    # e_x = 0 if x == 0
    # For x in 1..1008, e_x is 1 or -1
    e = [0] * (p + 1)  # index 0..p
    for x in range(1, p):
        if x in roots:
            e[x] = 1
        else:
            e[x] = -1
    
    # Use sliding window to find the smallest s such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues
    # Condition: sum of e[s..s+4] == 5 (all are 1)
    # This is equivalent to all five being residues since e values are in {-1, 1}
    
    # Compute the window sum for s=1 (window covers indices 1..5)
    window_sum = sum(e[1:6])
    
    for s in range(1, p - 3):  # s from 1 to 1004 (p-4 = 1005, so range(1, 1005))
        if time.time() > deadline:
            break
        if window_sum == 5:
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
        # Slide the window: remove e[s], add e[s+5]
        # Next window is s+1..s+5
        if s + 5 <= p - 1:  # s+5 <= 1008
            window_sum += e[s + 5] - e[s]
        else:
            break  # No more valid windows
    
    # Fallback: direct search (should not happen for p=1009)
    valid_residues = set(roots.keys())
    for s in range(1, p - 3):
        if all((s + i) in valid_residues for i in range(5)):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
    
    raise RuntimeError("No run found")
