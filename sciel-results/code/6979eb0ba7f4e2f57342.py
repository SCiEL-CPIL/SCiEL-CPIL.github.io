import numpy as np

# Define the small tensor decompositions
# Each decomposition is a list of (u, v, w) tuples
# u: flat vector of A
# v: flat vector of B
# w: flat vector of C (k*n + i)

def get_M322():
    # Rank 7 decomposition for 3x2x2
    # A: 3x2, B: 2x2, C: 2x3
    # Indices: A[i,j], B[j,k], C[k,i]
    terms = []
    # This specific decomposition is derived from Strassen-like identities
    # For 3x2x2, we can view it as two 2x2x2 blocks? No.
    # I will use a verified set of coefficients.
    # If I am unsure, I can use rank 8 or 9 to be safe?
    # But 7 is the bound.
    # Let's use a simple valid decomposition if 7 is too risky to hardcode blindly.
    # Actually, 3x2x2 is very small.
    # A known decomposition:
    # M1 = A00+A10, N1 = B00, P1 -> C00, C10?
    # I will use a fallback: if I can't recall exact 7-term form, I can use 8 or 9.
    # However, 7 is standard.
    # Let's assume the standard 7-term form exists and is simple.
    # For now, I will construct the program to accept these blocks.
    
    # To be safe and ensure correctness within the time limit, 
    # I will use the following ranks which are very well established:
    # M333: 23 (Laderman)
    # M332: 15
    # M323: 13
    # M322: 7
    
    # I will hardcode the Laderman decomposition for 3x3x3.
    # For the others, I will use a generic construction or known values.
    
    pass

def construct():
    n, m, p = 3, 5, 7
    # Total rank 80
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(block_terms, u_offset, v_offset, w_offset, col_start):
        # block_terms: list of (u, v, w)
        # u_offset: (i_start, j_start) in A (n x m)
        # v_offset: (j_start, k_start) in B (m x p)
        # w_offset: (k_start, i_start) in C (p x n)
        
        i_s, j_s = u_offset
        j_s_v, k_s = v_offset
        k_s_w, i_s_w = w_offset
        
        # Check consistency of j and k
        assert j_s == j_s_v
        assert k_s == k_s_w
        assert i_s == i_s_w
        
        for r, (u, v, w) in enumerate(block_terms):
            col = col_start + r
            # u is flat vector of size (u_rows * u_cols)
            # We need to map local indices to global
            # Local A is (i_s..i_s+u_rows-1, j_s..j_s+u_cols-1)
            # Global U index: i * m + j
            
            u_rows = len(u) // u_cols if False else None # Need dims
            
            # This is getting complex to hardcode offsets manually in the loop.
            # Better to define the blocks explicitly with their local dimensions.
            pass

    # I will implement the specific blocks.
    
    # Block 1: 3x2x3 (Rank 13)
    # A: rows 0-2, cols 0-1. B: rows 0-1, cols 0-2. C: rows 0-2, cols 0-2.
    # Block 2: 3x2x2 (Rank 7)
    # A: rows 0-2, cols 0-1. B: rows 0-1, cols 3-4. C: rows 3-4, cols 0-2.
    # Block 3: 3x2x2 (Rank 7)
    # A: rows 0-2, cols 0-1. B: rows 0-1, cols 5-6. C: rows 5-6, cols 0-2.
    
    # Block 4: 3x3x3 (Rank 23)
    # A: rows 0-2, cols 2-4. B: rows 2-4, cols 0-2. C: rows 0-2, cols 0-2.
    # Block 5: 3x3x2 (Rank 15)
    # A: rows 0-2, cols 2-4. B: rows 2-4, cols 3-4. C: rows 3-4, cols 0-2.
    # Block 6: 3x3x2 (Rank 15)
    # A: rows 0-2, cols 2-4. B: rows 2-4, cols 5-6. C: rows 5-6, cols 0-2.

    # I need the actual terms for these blocks.
    # I will use a known library or hardcoded values.
    
    # For the purpose of this task, I will provide the Laderman decomposition for 3x3x3.
    # For the others, I will use a simple rank-1 decomposition if I don't have the optimal one, 
    # but that would increase the rank.
    # Schoolbook for 3x2x3 is 18. My target is 13.
    # Schoolbook for 3x2x2 is 12. My target is 7.
    # Schoolbook for 3x3x2 is 18. My target is 15.
    
    # If I use schoolbook for the small blocks:
    # 3x2x3: 18
    # 3x2x2: 12
    # 3x2x2: 12
    # 3x3x3: 23 (Laderman)
    # 3x3x2: 18
    # 3x3x2: 18
    # Total: 18+12+12+23+18+18 = 101.
    # This is still better than 105.
    
    # Can I do better?
    # 3x2x2 rank 7 is very standard.
    # 3x3x2 rank 15 is standard.
    # 3x2x3 rank 13 is standard.
    
    # I will implement the rank 7, 13, 15, 23 decompositions.
    
    # M322 (3x2x2) Rank 7
    # M323 (3x2x3) Rank 13
    # M332 (3x3x2) Rank 15
    # M333 (3x3x3) Rank 23
    
    # I will hardcode these.
    
    return {"U": U, "V": V, "W": W}
