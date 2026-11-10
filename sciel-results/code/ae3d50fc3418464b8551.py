def construct():
    p = 1009
    
    # Precompute roots for each quadratic residue
    # roots[x] = r if r*r % p == x, else 0
    roots = [0] * p
    
    for r in range(1, p):
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r
            
    # Check for the smallest s
    for s in range(1, p - 4):
        # s, s+1, s+2, s+3, s+4 must be quadratic residues
        # i.e., roots[s+i] != 0
        if (roots[s] != 0 and 
            roots[s+1] != 0 and 
            roots[s+2] != 0 and 
            roots[s+3] != 0 and 
            roots[s+4] != 0):
            
            candidate_roots = [roots[s+i] for i in range(5)]
            
            # Optional: Verify
            # for i in range(5):
            #     if (candidate_roots[i] * candidate_roots[i]) % p != (s + i) % p:
            #         raise RuntimeError("Verification failed")
            
            return {
                'start': s,
                'roots': candidate_roots
            }
            
    # If no solution found (should not happen for p=1009)
    raise RuntimeError("No solution found")
