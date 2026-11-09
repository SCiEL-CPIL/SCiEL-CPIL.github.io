def construct():
    p = 1009
    # is_residue[y] will be True if y is a quadratic residue mod p
    # root[y] will store one root x such that x^2 % p == y
    is_residue = [False] * p
    root = [0] * p
    
    # Precompute residues
    for x in range(1, p):
        y = (x * x) % p
        is_residue[y] = True
        # Store the first root found. 
        # Since we iterate x from 1, root[y] will be the smallest x in 1..p-1 
        # such that x^2 % p == y. This is a valid root.
        if root[y] == 0:
            root[y] = x
            
    # Search for the smallest s
    for s in range(1, 1005): # s can be up to 1004 so s+4 <= 1008
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        valid = True
        roots = []
        for i in range(5):
            val = s + i
            if not is_residue[val]:
                valid = False
                break
            roots.append(root[val])
        
        if valid:
            return {'start': s, 'roots': roots}
            
    # Fallback, though a solution is guaranteed to exist for p=1009
    raise RuntimeError("No solution found")
