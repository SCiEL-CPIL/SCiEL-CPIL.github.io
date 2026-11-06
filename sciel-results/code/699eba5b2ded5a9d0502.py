import time

def construct():
    """
    Finds the minimum start s such that s, s+1, s+2, s+3, s+4 are 
    non-zero quadratic residues modulo 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    p = 1009
    
    # Precompute quadratic residues and one root for each
    # is_res[i] = True if i is a non-zero QR mod p
    # root[i] = x such that x*x % p == i
    is_res = [False] * p
    root = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # r will be non-zero for x in 1..p-1 since p is prime
        if not is_res[r]:
            is_res[r] = True
            root[r] = x
            
    # Search for the minimum s
    # s must be at least 1
    # s+i must be <= p-1 to be a non-zero residue in the standard 1..p-1 range?
    # The problem states 1 <= ri <= 1008 and ri**2 mod 1009 == s+i.
    # If s+i == 1009, then s+i mod 1009 == 0, which is excluded ("zero residues ... excluded").
    # So we need s+4 <= 1008 => s <= 1004.
    
    # We iterate s from 1 upwards to find the minimum.
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        # Since s starts at 1, s+i is at least 1.
        # We must ensure s+i does not wrap around to 0 mod 1009.
        # Max s is 1004, so max s+i is 1008. This is safe.
        
        valid = True
        roots_found = []
        for i in range(5):
            val = s + i
            if not is_res[val]:
                valid = False
                break
            roots_found.append(root[val])
            
        if valid:
            return {'start': s, 'roots': roots_found}
            
    # Fallback, though s=1 should work
    raise RuntimeError("No solution found")
