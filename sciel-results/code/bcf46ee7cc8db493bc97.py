import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # U: (9, R), V: (24, R), W: (24, R)
    
    # We will build U, V, W as lists of lists, then convert to arrays if needed, 
    # but the interface expects lists of lists or arrays. 
    # "integer/half-integer entries"
    
    rank_terms = []
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        # u_entries: list of (i, j, coeff)
        # v_entries: list of (j, k, coeff)
        # w_entries: list of (k, i, coeff)
        rank_terms.append((u_entries, v_entries, w_entries))

    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # Split B[0:2, 0:8] into B1[0:2, 0:4] and B2[0:2, 4:8]
    
    # Strassen for A[0:2, 0:2] * B1[0:2, 0:4]
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B1 indices: (0,0), (0,1), (1,0), (1,1) -> Global B (0,0), (0,1), (1,0), (1,1)
    # C1 indices: (0,0), (0,1), (1,0), (1,1) -> Global C (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    add_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    )
    # M2 = (A10+A11)B00
    add_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,0,1)]
    )
    # M3 = A00(B10-B00)
    add_term(
        [(0,0,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1)]
    )
    # M4 = A11(B01-B00)
    add_term(
        [(1,1,1)],
        [(0,1,1), (0,0,-1)],
        [(0,0,1)]
    )
    # M5 = (A00+A01)B11
    add_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    )
    # M6 = (A10-A00)(B00+B01)
    add_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(0,0,1)]
    )
    # M7 = (A01-A11)(B10+B11)
    add_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    )
    
    # Wait, the W coefficients for Strassen are not all 1.
    # C00 = M1 - M3 + M5 - M7
    # C01 = M4 + M7
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 + M6
    
    # My add_term above assumed W is just a placeholder.
    # I need to handle the W coefficients correctly.
    # The W matrix entry W[k*n+i, r] is the coefficient for C[i,k].
    # So for M1, which contributes to C00 and C11:
    # W[0*3+0, r] = 1
    # W[1*3+1, r] = 1
    
    # Let's restart the term definition to include W correctly.
    
    rank_terms = []
    
    def add_strassen_block(a_offset_i, a_offset_j, b_offset_j, b_offset_k, c_offset_i, c_offset_k):
        # a_offset: (i, j) for A
        # b_offset: (j, k) for B
        # c_offset: (i, k) for C
        # This block is 2x2 * 2x2 -> 2x2
        
        # M1
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+0, a_offset_j+0, 1), (a_offset_i+1, a_offset_j+1, 1)],
            'V': [(b_offset_j+0, b_offset_k+0, 1), (b_offset_j+1, b_offset_k+1, 1)],
            'W': [(c_offset_k+0, c_offset_i+0, 1), (c_offset_k+1, c_offset_i+1, 1)]
        })
        # M2
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+1, a_offset_j+0, 1), (a_offset_i+1, a_offset_j+1, 1)],
            'V': [(b_offset_j+0, b_offset_k+0, 1)],
            'W': [(c_offset_k+0, c_offset_i+1, 1)] # C10
        })
        # M3
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+0, a_offset_j+0, 1)],
            'V': [(b_offset_j+1, b_offset_k+0, 1), (b_offset_j+0, b_offset_k+0, -1)],
            'W': [(c_offset_k+0, c_offset_i+1, 1)] # C10
        })
        # M4
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+1, a_offset_j+1, 1)],
            'V': [(b_offset_j+0, b_offset_k+1, 1), (b_offset_j+0, b_offset_k+0, -1)],
            'W': [(c_offset_k+1, c_offset_i+0, 1)] # C01
        })
        # M5
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+0, a_offset_j+0, 1), (a_offset_i+0, a_offset_j+1, 1)],
            'V': [(b_offset_j+1, b_offset_k+1, 1)],
            'W': [(c_offset_k+0, c_offset_i+0, 1)] # C00
        })
        # M6
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+1, a_offset_j+0, 1), (a_offset_i+0, a_offset_j+0, -1)],
            'V': [(b_offset_j+0, b_offset_k+0, 1), (b_offset_j+0, b_offset_k+1, 1)],
            'W': [(c_offset_k+1, c_offset_i+1, 1)] # C11
        })
        # M7
        r = len(rank_terms)
        rank_terms.append({
            'U': [(a_offset_i+0, a_offset_j+1, 1), (a_offset_i+1, a_offset_j+1, -1)],
            'V': [(b_offset_j+1, b_offset_k+0, 1), (b_offset_j+1, b_offset_k+1, 1)],
            'W': [(c_offset_k+0, c_offset_i+0, -1), (c_offset_k+1, c_offset_i+0, 1)] # C00, C01
        })
        
    # Block A[0:2, 0:2] * B[0:2, 0:4]
    # A: i=0,1 j=0,1
    # B: j=0,1 k=0,1
    # C: i=0,1 k=0,1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block A[0:2, 0:2] * B[0:2, 4:8]
    # A: i=0,1 j=0,1
    # B: j=0,1 k=4,5 (local 0,1)
    # C: i=0,1 k=4,5 (local 0,1)
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # Now the remaining terms:
    # A[0:2, 2] * B[2, 0:8]
    # A: i=0,1 j=2
    # B: j=2 k=0..7
    # C: i=0,1 k=0..7
    for k in range(8):
        for i in [0, 1]:
            r = len(rank_terms)
            rank_terms.append({
                'U': [(i, 2, 1)],
                'V': [(2, k, 1)],
                'W': [(k, i, 1)]
            })
            
    # A[2, 0:2] * B[0:2, 0:8]
    # A: i=2 j=0,1
    # B: j=0,1 k=0..7
    # C: i=2 k=0..7
    for k in range(8):
        for j in [0, 1]:
            r = len(rank_terms)
            rank_terms.append({
                'U': [(2, j, 1)],
                'V': [(j, k, 1)],
                'W': [(k, 2, 1)]
            })
            
    # A[2, 2] * B[2, 0:8]
    # A: i=2 j=2
    # B: j=2 k=0..7
    # C: i=2 k=0..7
    for k in range(8):
        r = len(rank_terms)
        rank_terms.append({
            'U': [(2, 2, 1)],
            'V': [(2, k, 1)],
            'W': [(k, 2, 1)]
        })
        
    rank = len(rank_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(rank_terms):
        for i, j, coeff in term['U']:
            U[i * m + j][r] = coeff
        for j, k, coeff in term['V']:
            V[j * p + k][r] = coeff
        for k, i, coeff in term['W']:
            W[k * n + i][r] = coeff
            
    return {"U": U, "V": V, "W": W}
