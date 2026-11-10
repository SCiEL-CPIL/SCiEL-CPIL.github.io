import numpy as np

def construct():
    p = 1009
    
    # 1. Identify quadratic residues
    # r^2 mod p for r in 1..p-1
    # Using vectorization
    r = np.arange(1, p)
    res = (r * r) % p
    
    # is_res[i] is True if i is a non-zero quadratic residue
    is_res = np.zeros(p, dtype=bool)
    # Exclude 0, though r^2 mod p for r in 1..p-1 never yields 0
    np.put(is_res, res, True)
    
    # 2. Find the smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    # We need is_res[s] and is_res[s+1] and ... and is_res[s+4] to be True
    # This is equivalent to checking if the minimum of the window [s, s+4] is True (1)
    # Or summing the window and checking if sum == 5
    
    # We only care about s in [1, 1004]
    # s+4 max is 1008. is_res index 1008 is valid.
    
    # Create a view or just compute sums
    # Window size 5
    # valid s indices: 1 to 1004
    
    # Using a simple loop with numpy slicing is fast enough, but let's try to be fancy
    # Or just use the fact that p is small.
    
    min_s = None
    # Iterate s from 1 to 1004
    # Check if all 5 are True
    for s in range(1, 1005):
        if np.all(is_res[s:s+5]):
            min_s = s
            break
            
    if min_s is None:
        # Fallback, should not happen
        min_s = 1004
        
    # 3. Find roots for s, s+1, ..., s+4
    # We need ri such that ri^2 mod p == s+i
    # We can precompute a map or just find them
    # Since p is small, we can just search for each
    
    roots = []
    for i in range(5):
        target = min_s + i
        # Find r in 1..p-1 such that r^2 mod p == target
        # We can use the precomputed res array and r array
        # np.where(res == target) gives indices in r
        # r is 1..1008
        idxs = np.where(res == target)[0]
        if len(idxs) > 0:
            # Take the first one (smallest root)
            r_val = r[idxs[0]]
            roots.append(int(r_val))
        else:
            # Should not happen
            roots.append(1)
            
    return {'start': min_s, 'roots': roots}
