import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 59.0  # Safety margin before 60s limit

    # Precompute quadratic residues and one root for each
    residues = set()
    roots = {}
    
    for x in range(1, p):
        r = (x * x) % p
        residues.add(r)
        # Store the first root found. Since we iterate x from 1, 
        # this will be the smaller of the two roots (x, p-x) if x < p/2.
        # It doesn't matter which valid root we pick as long as 1 <= root <= 1008.
        if r not in roots:
            roots[r] = x

    # Search for the minimum start s
    # s ranges from 1 to 1004 because s+4 must be <= 1008 (since residues are 1..1008, 0 is excluded)
    # Actually, the problem says 1 <= ri <= 1008, so residues cannot be 0.
    # The values s+i must be in 1..1008.
    # Max s is 1004 because s+4 = 1008.
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # This should never happen, but just in case
            raise TimeoutError("Search took too long")
        
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # and not zero (which is handled by the set construction excluding 0 
        # if we are careful, but 0 is not in residues because x is 1..1008 and p is prime)
        # Wait, 0 is a quadratic residue (0^2=0), but the problem states "zero residues ... are excluded" 
        # and "1 <= ri <= 1008". This implies the residues themselves must be non-zero.
        # Our set `residues` contains values from x^2 mod p for x in 1..1008.
        # Does it contain 0? No, because p is prime and x is not a multiple of p.
        # So `residues` only contains non-zero residues.
        
        if all((s + i) in residues for i in range(5)):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
            
    raise RuntimeError("No baseline run found")
