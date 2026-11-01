import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline
    
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition for M_{2,3,9}
    # by splitting p=9 into three blocks of p=3.
    # Each block is M_{2,3,3}.
    # M_{2,3,3} is split into M_{2,2,2} (rank 7) and M_{2,1,3} (rank 6) + M_{2,2,1} (rank 4)?
    # Wait, my previous analysis:
    # M_{2,3,3} -> M_{2,2,3} (rank 11) + M_{2,1,3} (rank 6) = 17.
    # M_{2,2,3} -> M_{2,2,2} (rank 7) + M_{2,2,1} (rank 4) = 11.
    
    # Total rank 51.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # We need to map local indices to global indices
    # A is 2x3. Indices (i, j) -> i*m + j
    # B is 3x9. Indices (j, k) -> j*p + k
    # C is 2x9. Indices (i, k) -> k*n + i (Trace convention W[k*n+i])
    
    # Block 1: k in 0..2
    # Block 2: k in 3..5
    # Block 3: k in 6..8
    
    for block_idx in range(3):
        k_offset = block_idx * 3
        
        # Part 1: M_{2,2,2} using A cols 0,1 and B rows 0,1 and C cols 0,1 (local)
        # A_local: 2x2 (cols 0,1)
        # B_local: 2x2 (rows 0,1, cols 0,1)
        # C_local: 2x2 (cols 0,1)
        
        # Strassen terms for M_{2,2,2}
        # u, v, w are vectors of length n*m, m*p, p*n
        # We only set the relevant entries
        
        def get_u(i, j):
            vec = [0] * (n*m)
            vec[i*m + j] = 1
            return vec
            
        def get_v(j, k):
            vec = [0] * (m*p)
            vec[j*p + (k_offset + k)] = 1
            return vec
            
        def get_w(k, i):
            vec = [0] * (p*n)
            vec[(k_offset + k)*n + i] = 1
            return vec

        # Strassen 7 terms
        # M1: (A22-A11)(B11+B12) -> C11 part
        # We need linear combinations
        
        # This is getting complex to generate programmatically without a library.
        # I will hardcode the 51 terms.
        
    # Since I cannot easily generate the Strassen terms in code without a lot of boilerplate,
    # and the deadline is tight, I will return the schoolbook decomposition (rank 54) 
    # if I can't quickly implement the 51 rank one.
    # But 51 is a definite improvement.
    
    # Let's try to implement the 51 rank one.
    
    # ... (Implementation details omitted in thought, will write in final code)
    
    pass
