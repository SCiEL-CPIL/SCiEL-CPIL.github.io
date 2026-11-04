"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on Strassen's algorithm for 2x2 blocks.
    # The rank of 2x2 matrix multiplication is 7 (Strassen).
    # 4x4 matrix multiplication can be viewed as 8 products of 2x2 blocks.
    # Total rank = 8 * 7 = 56.
    
    # Strassen decomposition for 2x2 matrices:
    # C = A * B
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U, V, W vectors for each term k in 1..7
    # U_k corresponds to coefficients of A entries (A11, A12, A21, A22)
    # V_k corresponds to coefficients of B entries (B11, B12, B21, B22)
    # W_k corresponds to coefficients of C entries (C11, C12, C21, C22)
    
    # Note: The tensor decomposition is sum u_i * v_j * w_k.
    # Here, for each term, we have a vector u (4 entries), v (4 entries), w (4 entries).
    # The contribution to C is u * v^T * w? No.
    # The term is rank-1 tensor: u \otimes v \otimes w.
    # (u \otimes v \otimes w)_{i,j,k} = u_i v_j w_k.
    # This matches C_{ij} = sum_k A_{ik} B_{kj}.
    
    # Let's define the 7 terms for 2x2.
    # Indices for 2x2: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1).
    
    strassen_terms = [
        # (u, v, w)
        # M1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        # M3
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),
        # M4
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 0, 0)),
        # M5
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 0, 0, 0)),
        # M6
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]
    
    # Check Strassen correctness:
    # C11: M1(1) + M4(1) - M5(-1) + M7(1) -> 1+1-(-1)+1 = 4? No.
    # Let's re-verify Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # M1 w11=1, M4 w11=1, M5 w11=-1, M7 w11=1. Sum = 1+1-(-1)+1 = 4? 
    # Wait, M5 = (A11+A12)B22. w for M5 in C11 is -1.
    # M7 = (A12-A22)(B21+B22). w for M7 in C11 is 1.
    # Is the formula correct?
    # Standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # Let's check a specific entry.
    # A11=1, others 0. B11=1, others 0. C11=1.
    # M1: A11+A22=1, B11+B22=1 -> 1. w11=1.
    # M2: A21+A22=0, B11=1 -> 0.
    # M3: A11=1, B12-B22=0 -> 0.
    # M4: A22=0, B21-B11=-1 -> 0.
    # M5: A11+A12=1, B22=0 -> 0.
    # M6: A21-A11=-1, B11+B12=1 -> -1. w22=1.
    # M7: A12-A22=0, B21+B22=0 -> 0.
    # C11 = 1 + 0 - 0 + 0 = 1. Correct.
    # C22 = M1 - M2 + M3 + M6
    # C22 = 1 - 0 + 0 + (-1) = 0. Correct (since B12=0, A21=0).
    
    # The w vectors in my list above:
    # M1: (1,0,0,1) -> C11+=1, C22+=1. Correct.
    # M2: (0,0,1,0) -> C21+=1. Correct.
    # M3: (0,1,0,0) -> C12+=1. Correct.
    # M4: (1,0,0,0) -> C11+=1. Correct.
    # M5: (-1,0,0,0) -> C11-=-1. Correct.
    # M6: (0,0,0,1) -> C22+=1. Correct.
    # M7: (1,0,0,0) -> C11+=1. Correct.
    
    # Now, for 4x4, we have 8 block products.
    # Block (I,J) of A, Block (J,K) of B, Block (I,K) of C.
    # I, J, K in {0, 1}.
    # For each I, K, we have J=0 and J=1.
    # So 4 pairs of (I,K), each with 2 J's. Total 8.
    
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)] # 16 x 56
    V = [[0] * rank for _ in range(m * p)] # 16 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    for I in range(2):
        for J in range(2):
            for K in range(2):
                # This corresponds to A[I,J] * B[J,K] contributing to C[I,K]
                # But wait, C[I,K] = sum_J A[I,J] B[J,K].
                # So for fixed I, K, we iterate J.
                # My loop structure: I, J, K.
                # This covers all 8 combinations.
                
                # Map 2x2 indices to 4x4 indices.
                # A block (I,J) has rows 2I, 2I+1 and cols 2J, 2J+1.
                # B block (J,K) has rows 2J, 2J+1 and cols 2K, 2K+1.
                # C block (I,K) has rows 2I, 2I+1 and cols 2K, 2K+1.
                
                for term in strassen_terms:
                    u_2x2, v_2x2, w_2x2 = term
                    # u_2x2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in local 2x2 coords.
                    # Map to global A indices:
                    # local (r,c) -> global (2I+r, 2J+c)
                    # U index is i*m + j = i*4 + j.
                    
                    # Fill U
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            val = u_2x2[local_idx]
                            if val != 0:
                                global_i = 2 * I + r
                                global_j = 2 * J + c
                                U[global_i * 4 + global_j][col] += val
                                
                    # Fill V
                    # v_2x2 indices: local (r,c) in B block (J,K).
                    # B is m x p = 4 x 4. Index j*p + k.
                    # local (r,c) -> global (2J+r, 2K+c)
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            val = v_2x2[local_idx]
                            if val != 0:
                                global_j = 2 * J + r
                                global_k = 2 * K + c
                                V[global_j * 4 + global_k][col] += val
                                
                    # Fill W
                    # w_2x2 indices: local (r,c) in C block (I,K).
                    # W index is k*n + i.
                    # local (r,c) -> global (2I+r, 2K+c)
                    # So i = 2I+r, k = 2K+c.
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            val = w_2x2[local_idx]
                            if val != 0:
                                global_i = 2 * I + r
                                global_k = 2 * K + c
                                W[global_k * 4 + global_i][col] += val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
