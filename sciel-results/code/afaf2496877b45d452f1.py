"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Rank is 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U[i, j], V[j, k], W[k, i]
    # i, j, k in {0, 1}
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: 00=1, 11=1
    # V: 00=1, 11=1
    # W: 00=1, 11=1
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: {(i,j): val}, V: {(j,k): val}, W: {(k,i): val}
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # Term 2: M2 = (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)]),
        # Term 3: M3 = A00(B10-B11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # Term 4: M4 = A11(B01-B00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # Term 5: M5 = (A00+A10)B11
        ([(0,0,1), (1,0,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # Term 6: M6 = (A01-A00)(B00+B10)
        ([(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
        # Term 7: M7 = (A10-A11)(B01+B11)
        ([(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(0,0,1)])
    ]
    
    col_idx = 0
    for m_block in range(4):
        # m_block corresponds to k indices 2*m_block, 2*m_block+1
        k_offset = 2 * m_block
        
        for s in range(2):
            # s=0: A_L (j in 0,1), B_top (j in 0,1)
            # s=1: A_R (j in 2,3), B_bot (j in 2,3)
            
            for term in strassen_terms:
                u_coeffs, v_coeffs, w_coeffs = term
                
                # Apply to U
                for (i, j, val) in u_coeffs:
                    global_j = j + (2 if s == 1 else 0)
                    U[i * m + global_j][col_idx] += val
                    
                # Apply to V
                for (j, k, val) in v_coeffs:
                    global_j = j + (2 if s == 1 else 0)
                    global_k = k + k_offset
                    V[global_j * p + global_k][col_idx] += val
                    
                # Apply to W
                for (k, i, val) in w_coeffs:
                    global_k = k + k_offset
                    W[global_k * n + i][col_idx] += val
                    
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
