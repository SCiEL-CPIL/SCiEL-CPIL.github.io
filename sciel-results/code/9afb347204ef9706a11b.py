import numpy as np

def construct():
    p = 1009
    
    # Step 1: Compute roots using NumPy
    x = np.arange(1, p)
    residues = (x * x) % p
    
    # Map residue to root. 
    # We want roots[r] to be a valid root for residue r.
    # Initialize with 0 (invalid, since roots are 1..p-1)
    roots = np.zeros(p, dtype=int)
    
    # For each residue, store a root.
    # Since x and p-x give same residue, we can just assign.
    # The order of x is 1..p-1.
    # roots[residues] = x will result in the last x encountered for each residue being stored.
    # This is a valid root.
    roots[residues] = x
    
    # Step 2: Find minimum s
    # valid[i] is True if i is a quadratic residue (and not 0)
    # Note: 0 is a residue of 0, but we excluded 0 in x range?
    # x is 1..1008. 0^2=0 is not in residues.
    # So roots[0] will be 0.
    # We need roots[s+i] != 0.
    
    # We can check s from 1 to 1004
    for s in range(1, 1005):
        # Check s, s+1, s+2, s+3, s+4
        # Using numpy slicing for speed?
        # But we are in a python loop.
        # Let's just index.
        if (roots[s] != 0 and 
            roots[s+1] != 0 and 
            roots[s+2] != 0 and 
            roots[s+3] != 0 and 
            roots[s+4] != 0):
            
            res_roots = [int(roots[s+i]) for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    raise RuntimeError("No valid run found")
