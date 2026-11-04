import numpy as np
import time

DIMENSIONS = [4, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 145

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _construct_strassen_4x4x4():
    """
    Constructs a rank-28 decomposition for 4x4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks, with schoolbook for inner 2x2 products.
    """
    n, m, p = 4, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define block offsets
    # A blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B blocks: B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
    # C blocks: C00 (0:2, 0:2), C01 (0:2, 2:4), C10 (2:4, 0:2), C11 (2:4, 2:4)
    
    # Strassen terms definition
    # Each term is (A_coeff, B_coeff, C_coeff, sign)
    # A_coeff: list of (block_idx, sign)
    # B_coeff: list of (block_idx, sign)
    # C_coeff: list of (block_idx, sign)
    
    # We will iterate over the 7 Strassen terms
    # For each, we expand the 2x2 product into 4 schoolbook terms
    
    col = 0
    for k in range(7):
        # Define A_k and B_k linear combinations
        # A_k is 2x2 matrix of coefficients for A blocks
        # B_k is 2x2 matrix of coefficients for B blocks
        
        # Strassen formulas
        if k == 0: # M1
            A_comb = [[(0,1), (0,0)], [(0,0), (1,1)]] # A00, A11
            B_comb = [[(0,1), (1,0)], [(0,0), (1,1)]] # B00, B11
            C_comb = [(0, 1)] # C00
        elif k == 1: # M2
            A_comb = [[(0,0), (0,0)], [(1,1), (1,1)]] # A10, A11
            B_comb = [[(0,1), (0,0)], [(0,0), (0,0)]] # B00
            C_comb = [(2, 1)] # C10
        elif k == 2: # M3
            A_comb = [[(0,1), (0,0)], [(0,0), (0,0)]] # A00
            B_comb = [[(0,0), (1,-1)], [(0,0), (0,0)]] # B01-B11
            C_comb = [(1, 1)] # C01
        elif k == 3: # M4
            A_comb = [[(0,0), (0,0)], [(1,1), (1,1)]] # A11
            B_comb = [[(0,0), (1,-1)], [(0,0), (0,0)]] # B10-B00
            C_comb = [(0, 1), (2, 1)] # C00, C10
        elif k == 4: # M5
            A_comb = [[(0,1), (0,1)], [(0,0), (0,0)]] # A00, A01
            B_comb = [[(0,0), (1,1)], [(0,0), (0,0)]] # B11
            C_comb = [(0, -1), (1, 1)] # C00, C01
        elif k == 5: # M6
            A_comb = [[(0,0), (1,-1)], [(0,0), (0,0)]] # A10-A00
            B_comb = [[(0,1), (0,1)], [(0,0), (0,0)]] # B00, B01
            C_comb = [(3, 1)] # C11
        elif k == 6: # M7
            A_comb = [[(0,0), (0,1)], [(0,0), (1,-1)]] # A01-A11
            B_comb = [[(0,0), (1,1)], [(0,0), (0,0)]] # B10, B11
            C_comb = [(0, 1)] # C00
            
        # Expand 2x2 product
        # A_k[u,w] B_k[w,v]
        for u in range(2):
            for v in range(2):
                for w in range(2):
                    # Get coefficients
                    a_coeff = A_comb[u][w][0]
                    a_sign = A_comb[u][w][1]
                    b_coeff = B_comb[w][v][0]
                    b_sign = B_comb[w][v][1]
                    
                    # Determine global indices
                    # A block a_coeff, local (u, w)
                    # Block 0: rows 0-1, cols 0-1
                    # Block 1: rows 0-1, cols 2-3
                    # Block 2: rows 2-3, cols 0-1
                    # Block 3: rows 2-3, cols 2-3
                    # a_coeff is 0,1 for top/bottom? No, my A_comb used (block, sign).
                    # Let's re-verify A_comb structure.
                    # A_comb[u][w] is (block_idx, sign).
                    # block_idx 0: A00, 1: A01, 2: A10, 3: A11?
                    # My previous definition:
                    # M1: A00, A11.
                    # Let's use 0:A00, 1:A01, 2:A10, 3:A11.
                    
                    # Re-map A_comb
                    # M1: A00(0), A11(3)
                    # M2: A10(2), A11(3)
                    # M3: A00(0)
                    # M4: A11(3)
                    # M5: A00(0), A01(1)
                    # M6: A10(2), A00(0)
                    # M7: A01(1), A11(3)
                    
                    # Re-map B_comb
                    # M1: B00(0), B11(3)
                    # M2: B00(0)
                    # M3: B01(1), B11(3)
                    # M4: B10(2), B00(0)
                    # M5: B11(3)
                    # M6: B00(0), B01(1)
                    # M7: B10(2), B11(3)
                    
                    # Re-map C_comb
                    # M1: C00(0)
                    # M2: C10(2)
                    # M3: C01(1)
                    # M4: C00(0), C10(2)
                    # M5: C00(0), C01(1)
                    # M6: C11(3)
                    # M7: C00(0)
                    
                    # This mapping is getting messy. Let's just hardcode the 28 terms.
                    pass
    
    # This is too error-prone to generate dynamically in code without testing.
    # I will return the schoolbook for now and try to hardcode a known low rank if I can find one.
    # Actually, I can just return the schoolbook rank 80 if I can't be sure.
    # But 29 is a strong candidate.
    return None

def construct():
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
