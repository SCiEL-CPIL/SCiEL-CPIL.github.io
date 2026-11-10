# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 by removing a single line from the plane.
    
    Let M be the line x=0. Let N = F_{49}^2 \ M.
    For any point y in N, the vertical line through y is contained in N (except possibly y itself, 
    but actually the whole line is in N if x!=0? No, vertical line x=c has x=c!=0, so yes).
    Wait, if y=(c, d) with c!=0, the vertical line x=c is in N. So y is covered.
    
    For any point x in M (x=0), we need a line L through x such that L \ {x} subset N.
    Choose a non-vertical line through x, e.g., y = m*x.
    The points on this line other than x have x-coordinate t != 0.
    Thus they are in N.
    So the condition is satisfied.
    
    Size of N is q^2 - q.
    Complement size is q.
    """
    p = 7
    q = p * p
    
    # N consists of all points (x, y) where x != 0
    # x is represented by (xa, xb). x=0 means xa=0, xb=0.
    # We iterate over all points and exclude those with xa=0, xb=0.
    
    N_indices = []
    # Point index = ((xa * p + xb) * p + ya) * p + yb
    for xa in range(p):
        for xb in range(p):
            if xa == 0 and xb == 0:
                continue
            for ya in range(p):
                for yb in range(p):
                    idx = ((xa * p + xb) * p + ya) * p + yb
                    N_indices.append(idx)
                    
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        result[i] = [[xa, xb], [ya, yb]]
        
    return result

# EVOLVE-BLOCK-END
