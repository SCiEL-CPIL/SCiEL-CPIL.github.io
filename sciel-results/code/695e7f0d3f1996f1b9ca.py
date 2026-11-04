"""Program for matrix multiplication 3x5 x 5x6 via Strassen-style block decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will determine the rank by building the decomposition
    # and counting the number of non-zero columns generated.
    # However, the interface requires returning U, V, W.
    # We can build them as lists of lists or numpy arrays.
    # The problem statement says "integer/half-integer entries".
    
    # Initialize U, V, W as zero matrices of size (n*m, 0), etc?
    # We don't know the rank upfront if we build dynamically.
    # But we calculated it as 84.
    # Let's build a list of terms first, then assemble.
    
    terms = []
    
    def add_schoolbook(i0, j0, k0, ns, ms, ps):
        """Add schoolbook terms for a block of size ns x ms x ps at offset (i0, j0, k0)."""
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    u_vec = { (i, j): 1 }
                    v_vec = { (j, k): 1 }
                    w_vec = { (k, i): 1 }
                    terms.append((i0, j0, k0, u_vec, v_vec, w_vec))

    def add_strassen(i0, j0, k0):
        """Add Strassen terms for a 2x2x2 block at offset (i0, j0, k0)."""
        # Term 1: (a+d)(e+h)
        terms.append((i0, j0, k0, 
                      {(0,0): 1, (1,1): 1}, 
                      {(0,0): 1, (1,1): 1}, 
                      {(0,0): 1, (1,1): 1}))
        # Term 2: (c+d)e
        terms.append((i0, j0, k0, 
                      {(1,0): 1, (1,1): 1}, 
                      {(0,0): 1}, 
                      {(0,1): 1, (1,1): -1})) # W indices are (k,i). C10 is k=0,i=1. C11 is k=1,i=1.
        # Wait, C10 corresponds to row 1, col 0. In W index k*n+i, this is 0*2+1 = 1.
        # My previous mapping: W index k*n+i.
        # C_10: i=1, k=0. Index 0*2+1 = 1.
        # C_11: i=1, k=1. Index 1*2+1 = 3.
        # In the term 2 w_vec above: {(0,1): 1, (1,1): -1} corresponds to (k=0, i=1) and (k=1, i=1). Correct.
        
        # Term 3: a(f-h)
        terms.append((i0, j0, k0, 
                      {(0,0): 1}, 
                      {(0,1): 1, (1,1): -1}, 
                      {(0,0): 1, (1,1): 1})) # C01 (k=0,i=1) and C11 (k=1,i=1).
        # C01: i=1, k=0. Index 1.
        # C11: i=1, k=1. Index 3.
        # w_vec {(0,1): 1, (1,1): 1} means (k=0,i=1) and (k=1,i=1). Correct.
        
        # Term 4: d(g-e)
        terms.append((i0, j0, k0, 
                      {(1,1): 1}, 
                      {(1,0): 1, (0,0): -1}, 
                      {(0,0): 1, (0,1): 1})) # C00 (k=0,i=0) and C10 (k=0,i=1).
        # C00: i=0, k=0. Index 0.
        # C10: i=1, k=0. Index 1.
        # w_vec {(0,0): 1, (0,1): 1} means (k=0,i=0) and (k=0,i=1). Correct.
        
        # Term 5: (a+b)h
        terms.append((i0, j0, k0, 
                      {(0,0): 1, (0,1): 1}, 
                      {(1,1): 1}, 
                      {(0,0): -1, (0,1): 1})) # C00 (k=0,i=0) and C01 (k=0,i=1).
        # C00: i=0, k=0. Index 0.
        # C01: i=1, k=0. Index 1.
        # w_vec {(0,0): -1, (0,1): 1} means (k=0,i=0) and (k=0,i=1). Correct.
        
        # Term 6: (c-a)(e+f)
        terms.append((i0, j0, k0, 
                      {(1,0): 1, (0,0): -1}, 
                      {(0,0): 1, (0,1): 1}, 
                      {(1,1): 1})) # C11 (k=1,i=1).
        # C11: i=1, k=1. Index 3.
        # w_vec {(1,1): 1} means (k=1,i=1). Correct.
        
        # Term 7: (b-d)(g+h)
        terms.append((i0, j0, k0, 
                      {(0,1): 1, (1,1): -1}, 
                      {(1,0): 1, (1,1): 1}, 
                      {(0,0): 1})) # C00 (k=0,i=0).
        # C00: i=0, k=0. Index 0.
        # w_vec {(0,0): 1} means (k=0,i=0). Correct.

    # Decomposition Logic
    
    # 1. Split n=3 into 2 and 1.
    # Block 1.1: n=2, m=5, p=6. Offset (0,0,0)
    # Block 1.2: n=1, m=5, p=6. Offset (2,0,0)
    
    # --- Block 1.1: M_{2,5,6} ---
    # Split m=5 into 2 and 3.
    # Sub 1.1.1: n=2, m=2, p=6. Offset (0,0,0)
    # Sub 1.1.2: n=2, m=3, p=6. Offset (0,2,0)
    
    # --- Sub 1.1.1: M_{2,2,6} ---
    # Split p=6 into 2, 2, 2.
    # Sub 1.1.1.1: n=2, m=2, p=2. Offset (0,0,0) -> Strassen
    add_strassen(0, 0, 0)
    # Sub 1.1.1.2: n=2, m=2, p=2. Offset (0,0,2) -> Strassen
    add_strassen(0, 0, 2)
    # Sub 1.1.1.3: n=2, m=2, p=2. Offset (0,0,4) -> Strassen
    add_strassen(0, 0, 4)
    
    # --- Sub 1.1.2: M_{2,3,6} ---
    # Split p=6 into 2, 2, 2.
    # Sub 1.1.2.1: n=2, m=3, p=2. Offset (0,2,0)
    # Split m=3 into 1, 2.
    # Sub 1.1.2.1.1: n=2, m=1, p=2. Offset (0,2,0) -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 1, 2)
    # Sub 1.1.2.1.2: n=2, m=2, p=2. Offset (0,3,0) -> Strassen
    add_strassen(0, 3, 0)
    
    # Sub 1.1.2.2: n=2, m=3, p=2. Offset (0,2,2)
    # Split m=3 into 1, 2.
    # Sub 1.1.2.2.1: n=2, m=1, p=2. Offset (0,2,2) -> Schoolbook
    add_schoolbook(0, 2, 2, 2, 1, 2)
    # Sub 1.1.2.2.2: n=2, m=2, p=2. Offset (0,3,2) -> Strassen
    add_strassen(0, 3, 2)
    
    # Sub 1.1.2.3: n=2, m=3, p=2. Offset (0,2,4)
    # Split m=3 into 1, 2.
    # Sub 1.1.2.3.1: n=2, m=1, p=2. Offset (0,2,4) -> Schoolbook
    add_schoolbook(0, 2, 4, 2, 1, 2)
    # Sub 1.1.2.3.2: n=2, m=2, p=2. Offset (0,3,4) -> Strassen
    add_strassen(0, 3, 4)
    
    # --- Block 1.2: M_{1,5,6} ---
    # Split m=5 into 2 and 3.
    # Sub 1.2.1: n=1, m=2, p=6. Offset (2,0,0)
    # Split p=6 into 2, 2, 2.
    # Sub 1.2.1.1: n=1, m=2, p=2. Offset (2,0,0) -> Schoolbook
    add_schoolbook(2, 0, 0, 1, 2, 2)
    # Sub 1.2.1.2: n=1, m=2, p=2. Offset (2,0,2) -> Schoolbook
    add_schoolbook(2, 0, 2, 1, 2, 2)
    # Sub 1.2.1.3: n=1, m=2, p=2. Offset (2,0,4) -> Schoolbook
    add_schoolbook(2, 0, 4, 1, 2, 2)
    
    # Sub 1.2.2: n=1, m=3, p=6. Offset (2,2,0)
    # Split p=6 into 2, 2, 2.
    # Sub 1.2.2.1: n=1, m=3, p=2. Offset (2,2,0)
    # Split m=3 into 1, 2.
    # Sub 1.2.2.1.1: n=1, m=1, p=2. Offset (2,2,0) -> Schoolbook
    add_schoolbook(2, 2, 0, 1, 1, 2)
    # Sub 1.2.2.1.2: n=1, m=2, p=2. Offset (2,3,0) -> Schoolbook
    add_schoolbook(2, 3, 0, 1, 2, 2)
    
    # Sub 1.2.2.2: n=1, m=3, p=2. Offset (2,2,2)
    # Split m=3 into 1, 2.
    # Sub 1.2.2.2.1: n=1, m=1, p=2. Offset (2,2,2) -> Schoolbook
    add_schoolbook(2, 2, 2, 1, 1, 2)
    # Sub 1.2.2.2.2: n=1, m=2, p=2. Offset (2,3,2) -> Schoolbook
    add_schoolbook(2, 3, 2, 1, 2, 2)
    
    # Sub 1.2.2.3: n=1, m=3, p=2. Offset (2,2,4)
    # Split m=3 into 1, 2.
    # Sub 1.2.2.3.1: n=1, m=1, p=2. Offset (2,2,4) -> Schoolbook
    add_schoolbook(2, 2, 4, 1, 1, 2)
    # Sub 1.2.2.3.2: n=1, m=2, p=2. Offset (2,3,4) -> Schoolbook
    add_schoolbook(2, 3, 4, 1, 2, 2)

    # Assemble U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (i0, j0, k0, u_vec, v_vec, w_vec) in enumerate(terms):
        for (i, j), val in u_vec.items():
            U[(i0 + i) * m + (j0 + j)][col] = val
        for (j, k), val in v_vec.items():
            V[(j0 + j) * p + (k0 + k)][col] = val
        for (k, i), val in w_vec.items():
            W[(k0 + k) * n + (i0 + i)][col] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
