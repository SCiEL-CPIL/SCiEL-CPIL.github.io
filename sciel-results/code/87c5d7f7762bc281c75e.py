import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = 2, 2, 11
    
    # Rank 39: 5 Strassen blocks (rank 7) + 1 Schoolbook block (rank 4)
    # We define the terms as lists of (index, coeff) pairs for U, V, W
    terms = []
    
    # Helper to add a term
    def add_term(u_pairs, v_pairs, w_pairs):
        terms.append((u_pairs, v_pairs, w_pairs))
        
    # Indices for A (2x2): 0=a, 1=b, 2=c, 3=d
    # Indices for B (2x11): 0..10 for row 0, 11..21 for row 1
    # V index for B[j, k] = j*p + k
    # W indices (C indices): C[i, k]. Trace convention: W[k*n + i, r]. n=2. W[2*k + i, r].
    
    # Define Strassen block for columns k0, k1
    def strassen_block(k0, k1):
        # B indices
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        add_term(
            [(0, 1), (3, 1)],
            [(v_e, 1), (v_h, 1)],
            [(w_c00, 1), (w_c11, 1)]
        )
        
        # m2 = (c+d)e
        add_term(
            [(2, 1), (3, 1)],
            [(v_e, 1)],
            [(w_c10, 1), (w_c11, -1)]
        )
        
        # m3 = a(f-h)
        add_term(
            [(0, 1)],
            [(v_f, 1), (v_h, -1)],
            [(w_c01, 1), (w_c11, 1)]
        )
        
        # m4 = d(g-e)
        add_term(
            [(3, 1)],
            [(v_g, 1), (v_e, -1)],
            [(w_c00, 1), (w_c10, 1)]
        )
        
        # m5 = (a+b)h
        add_term(
            [(0, 1), (1, 1)],
            [(v_h, 1)],
            [(w_c00, -1), (w_c01, 1)]
        )
        
        # m6 = (c-a)(e+f)
        add_term(
            [(2, 1), (0, -1)],
            [(v_e, 1), (v_f, 1)],
            [(w_c11, 1)]
        )
        
        # m7 = (b-d)(g+h)
        add_term(
            [(1, 1), (3, -1)],
            [(v_g, 1), (v_h, 1)],
            [(w_c00, 1)]
        )
        
    # Define Schoolbook block for column k2
    def schoolbook_block(k2):
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        # Term 1: a * B[0, k2]
        add_term(
            [(0, 1)],
            [(v_e2, 1)],
            [(w_c02, 1)]
        )
        
        # Term 2: b * B[1, k2]
        add_term(
            [(1, 1)],
            [(v_g2, 1)],
            [(w_c02, 1)]
        )
        
        # Term 3: c * B[0, k2]
        add_term(
            [(2, 1)],
            [(v_e2, 1)],
            [(w_c12, 1)]
        )
        
        # Term 4: d * B[1, k2]
        add_term(
            [(3, 1)],
            [(v_g2, 1)],
            [(w_c12, 1)]
        )

    # Generate blocks
    # 5 Strassen blocks for columns 0-9
    for block_idx in range(5):
        if time.time() > deadline:
            # Return partial result if timeout, though this shouldn't happen for static generation
            pass
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        strassen_block(k0, k1)
        
    # 1 Schoolbook block for column 10
    schoolbook_block(10)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_pairs, v_pairs, w_pairs) in enumerate(terms):
        for idx, coeff in u_pairs:
            U[idx][r] += coeff
        for idx, coeff in v_pairs:
            V[idx][r] += coeff
        for idx, coeff in w_pairs:
            W[idx][r] += coeff
            
    return {"U": U, "V": V, "W": W}
