import time

def construct():
    p = 1009
    deadline = time.time() + 80

    # 1. Precompute non-zero quadratic residues and their roots
    res_to_root = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We just need one root for each residue.
            # Storing the first one found is fine.
            if r not in res_to_root:
                res_to_root[r] = x
    
    # 2. Get the list of residues and sort it
    residues = sorted(res_to_root.keys())
    
    # 3. Sliding window of size 5
    # We look for 5 consecutive integers in the sorted list of residues
    # If residues[i], residues[i+1], ..., residues[i+4] are consecutive,
    # then residues[i] is a candidate start s.
    # Since the list is sorted, the first such window will have the minimum s.
    
    n = len(residues)
    for i in range(n - 4):
        if time.time() > deadline:
            break
            
        # Check if the next 4 elements are consecutive to the current one
        # residues[i] = s
        # residues[i+1] must be s+1
        # residues[i+2] must be s+2
        # residues[i+3] must be s+3
        # residues[i+4] must be s+4
        
        if (residues[i+1] == residues[i] + 1 and
            residues[i+2] == residues[i] + 2 and
            residues[i+3] == residues[i] + 3 and
            residues[i+4] == residues[i] + 4):
            
            s = residues[i]
            roots = [res_to_root[s + j] for j in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive quadratic residues found')
