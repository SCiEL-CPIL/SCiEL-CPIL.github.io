def construct():
    n = 256
    S = []
    # We add elements in descending order to maintain the property that 
    # new element x is smaller than all existing elements y, z.
    # Condition: For all y, z in S (y <= z), x must not divide y+z.
    
    for x in range(n, 0, -1):
        # Check if x can be added to S
        valid = True
        # We need to check pairs y, z in S.
        # Instead of O(|S|^2) pair checks, we use residues modulo x.
        # If there exist y, z in S such that (y + z) % x == 0, it's invalid.
        
        # Optimize: 
        # If x is large, S is small.
        # If x is small, S is large, but modulo operation is cheap.
        
        # Let's just do the O(|S|^2) check first, it's simple and n=256 is small.
        # |S| max ~ 128. Pairs ~ 16k. Total iterations 256 * 16k ~ 4M. 
        # Python can do 4M simple ops in < 1 second.
        
        for i in range(len(S)):
            y = S[i]
            for j in range(i, len(S)):
                z = S[j]
                if (y + z) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            
    return S
