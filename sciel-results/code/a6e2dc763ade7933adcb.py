"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # We use a hybrid decomposition:
    # 6 blocks of 2 columns (using Strassen rank 7)
    # 1 block of 1 column (using Schoolbook rank 4)
    # Total rank = 6 * 7 + 4 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen terms for 2x2x2 block
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices for block g:
    #   0: B[0, 2g]
    #   1: B[1, 2g]
    #   2: B[0, 2g+1]
    #   3: B[1, 2g+1]
    # W indices for block g:
    #   0: C[2g, 0] -> W[2g*n + 0] ? No, W index is k*n+i.
    #   Task says: W[k*n+i, r].
    #   C has shape (p, n). C[k, i].
    #   So W row index is k*n + i.
    #   For block g, k is 2g and 2g+1. i is 0 and 1.
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (2, 1), (4, -1), (6, 1)]), # c00=M1+M4-M5+M7? No, this is just M1 contribution.
        # We need to sum contributions to W.
        # Let's define the terms as:
        # r: U[row] = sum(u_idx, val), V[col] = sum(v_idx, val), W[row] = sum(w_idx, val)
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1)]), # Adds to C00? No, M1 adds to C00 and C11.
        # Let's just list the 7 terms and their contributions to the 4 outputs.
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00, +1 to C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        
        # Term 1: M2 = (A10+A11)B00
        # Contributes +1 to C10, -1 to C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        
        # Term 2: M3 = A00(B01-B11)
        # Contributes +1 to C01, +1 to C11
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        
        # Term 3: M4 = A11(B10-B00)
        # Contributes +1 to C00, +1 to C10
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        
        # Term 4: M5 = (A00+A01)B11
        # Contributes -1 to C00, +1 to C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)], [(0, 1)]),
    ]
    
    for g in range(6):
        # Block g handles columns 2g, 2g+1 of B, and rows 2g, 2g+1 of C.
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            # U part (A indices are global 0..3)
            for u_idx, val in u_coeffs:
                U[u_idx][col_idx] += val
                
            # V part (B indices are local to block)
            # V global index for B[j, k] is j*p + k.
            # Here j is 0 or 1. k is 2g or 2g+1.
            # Local indices in term: 0->B[0,2g], 1->B[1,2g], 2->B[0,2g+1], 3->B[1,2g+1]
            for v_idx, val in v_coeffs:
                # v_idx 0, 1 are row 0, 1 of B?
                # My local mapping:
                # 0: B[0, 2g] -> global 0*p + 2g
                # 1: B[1, 2g] -> global 1*p + 2g
                # 2: B[0, 2g+1] -> global 0*p + 2g+1
                # 3: B[1, 2g+1] -> global 1*p + 2g+1
                if v_idx == 0:
                    v_global = 0 * p + 2 * g
                elif v_idx == 1:
                    v_global = 1 * p + 2 * g
                elif v_idx == 2:
                    v_global = 0 * p + (2 * g + 1)
                else:
                    v_global = 1 * p + (2 * g + 1)
                V[v_global][col_idx] += val
                
            # W part (C indices)
            # W global index for C[k, i] is k*n + i.
            # Local indices in term:
            # 0: C[2g, 0] -> global (2g)*n + 0
            # 1: C[2g, 1] -> global (2g)*n + 1
            # 2: C[2g+1, 0] -> global (2g+1)*n + 0
            # 3: C[2g+1, 1] -> global (2g+1)*n + 1
            for w_idx, val in w_coeffs:
                if w_idx == 0:
                    w_global = (2 * g) * n + 0
                elif w_idx == 1:
                    w_global = (2 * g) * n + 1
                elif w_idx == 2:
                    w_global = (2 * g + 1) * n + 0
                else:
                    w_global = (2 * g + 1) * n + 1
                W[w_global][col_idx] += val
                
            col_idx += 1
            
    # Last block: column 12
    # Schoolbook for 2x2x1
    # c0 = A00 B0 + A01 B1
    # c1 = A10 B0 + A11 B1
    k = 12
    # Term 0: A00 B[0,12] -> C[12,0]
    U[0][col_idx] = 1
    V[0 * p + k][col_idx] = 1
    W[k * n + 0][col_idx] = 1
    col_idx += 1
    
    # Term 1: A01 B[1,12] -> C[12,0]
    U[1][col_idx] = 1
    V[1 * p + k][col_idx] = 1
    W[k * n + 0][col_idx] = 1
    col_idx += 1
    
    # Term 2: A10 B[0,12] -> C[12,1]
    U[2][col_idx] = 1
    V[0 * p + k][col_idx] = 1
    W[k * n + 1][col_idx] = 1
    col_idx += 1
    
    # Term 3: A11 B[1,12] -> C[12,1]
    U[3][col_idx] = 1
    V[1 * p + k][col_idx] = 1
    W[k * n + 1][col_idx] = 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
