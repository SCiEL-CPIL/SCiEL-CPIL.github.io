import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Rank 54 construction
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 standard multiplication block
    def add_2x2x2_block(i_offset, j_offset, k_offset, start_col):
        # i in {0,1}, j in {0,1}, k in {0,1}
        # Global i = i_offset + i
        # Global j = j_offset + j
        # Global k = k_offset + k
        
        # 8 terms
        for a_i in range(2):
            for a_j in range(2):
                for b_k in range(2):
                    # M = A[i,j] * B[j,k]
                    # U[i*m+j, col] = 1
                    # V[j*p+k, col] = 1
                    # W[k*n+i, col] = 1
                    
                    gi = i_offset + a_i
                    gj = j_offset + a_j
                    gk = k_offset + b_k
                    
                    # Wait, the term is A[gi, gj] * B[gj, gk]
                    # So U row is gi*m + gj
                    # V row is gj*p + gk
                    # W row is gk*n + gi
                    
                    U[gi * m + gj, start_col] = 1
                    V[gj * p + gk, start_col] = 1
                    W[gk * n + gi, start_col] = 1
                    start_col += 1
        return start_col

    # Helper to add a 2x1x2 standard multiplication block
    def add_2x1x2_block(i_offset, j_offset, k_offset, start_col):
        # i in {0,1}, j in {0} (fixed), k in {0,1}
        # Global i = i_offset + i
        # Global j = j_offset + 0
        # Global k = k_offset + k
        
        # 2 terms: A[i, j] * B[j, k]
        # We need to cover C[i, k] for i in 0,1 and k in 0,1.
        # C[0,0] = A[0,j]B[j,0]
        # C[0,1] = A[0,j]B[j,1]
        # C[1,0] = A[1,j]B[j,0]
        # C[1,1] = A[1,j]B[j,1]
        
        # Term 1: A[0,j] * B[j,0] -> C[0,0]
        # Term 2: A[1,j] * B[j,1] -> C[1,1]
        # This leaves C[0,1] and C[1,0] uncovered.
        
        # Rank 2 decomposition for 2x1x2:
        # M1 = A[0,j] * (B[j,0] + B[j,1])
        # M2 = A[1,j] * (B[j,0] + B[j,1])
        # This doesn't work directly for separate C entries.
        
        # Let's use the explicit rank 2 factors derived earlier:
        # alpha 1: U=[1,0], V=[1], W=[1,0,1,0] (indices k,i: 00, 10, 01, 11)
        # Wait, W index is k*n + i.
        # n=2.
        # W[0*2+0] = W[0]
        # W[1*2+0] = W[2]
        # W[0*2+1] = W[1]
        # W[1*2+1] = W[3]
        
        # My previous derivation:
        # alpha=1: U=[1,0], V=[1], W=[1, 0, 1, 0]
        # T_000 = 1*1*1 = 1
        # T_001 = 1*1*0 = 0
        # T_100 = 0*1*1 = 0
        # T_101 = 0*1*0 = 0
        # This covers C[0,0].
        
        # alpha=2: U=[0,1], V=[1], W=[0, 1, 0, 1]
        # T_000 = 0
        # T_001 = 0
        # T_100 = 0
        # T_101 = 1
        # This covers C[1,1].
        
        # We still need C[0,1] and C[1,0].
        # So rank 2 is NOT sufficient for 2x1x2?
        # Rank of 2x1x2 tensor is 2?
        # Tensor T_{i0k} = 1.
        # T = e_0 e_0^T + e_1 e_1^T? No.
        # T = [1 0; 0 1] in the i-k plane.
        # This is the identity matrix.
        # Rank of identity matrix of size 2 is 2.
        # So yes, rank 2 is sufficient.
        
        # But my factors above only covered diagonal.
        # I need factors that cover off-diagonal or a combination.
        
        # Let's try:
        # M1 = A[0,j] * B[j,0]
        # M2 = A[1,j] * B[j,1]
        # This gives C[0,0] and C[1,1].
        
        # We need C[0,1] and C[1,0].
        # Can we do it with 2 more terms? Total 4?
        # If rank is 2, we must be able to do it with 2 terms total.
        
        # Let's check the rank of the 2x1x2 tensor again.
        # T_{i0k} = delta_{ik}? No, T_{i0k} = 1 for all i,k?
        # No, C_{ik} = A_{i0} B_{0k}.
        # The tensor is T_{i0k} = 1.
        # This is a rank 1 tensor?
        # T_{i0k} = U_{i0\alpha} V_{0k\alpha} W_{ki\alpha}.
        # If we set U=[1,1], V=[1], W=[1,1,1,1]?
        # T_000 = 1*1*1 = 1
        # T_001 = 1*1*1 = 1
        # T_100 = 1*1*1 = 1
        # T_101 = 1*1*1 = 1
        # Yes! Rank 1 is sufficient for 2x1x2!
        
        # So we can use 1 column per 2x1x2 block.
        # Total rank = 6 * 8 + 3 * 1 = 48 + 3 = 51.
        
        # Let's verify Rank 1 for 2x1x2.
        # U shape (2, 1). V shape (1, 1). W shape (4, 1).
        # U = [1, 1]. V = [1]. W = [1, 1, 1, 1].
        # Check T_{i0k} = U_i V_0 W_{ki}.
        # i=0, k=0: 1*1*W_0 = 1.
        # i=0, k=1: 1*1*W_1 = 1.
        # i=1, k=0: 1*1*W_2 = 1.
        # i=1, k=1: 1*1*W_3 = 1.
        # Correct.
        
        # So update add_2x1x2_block to use 1 column.
        
        gi0 = i_offset
        gi1 = i_offset + 1
        gj = j_offset
        gk0 = k_offset
        gk1 = k_offset + 1
        
        # U rows: gi0*m + gj, gi1*m + gj
        U[gi0 * m + gj, start_col] = 1
        U[gi1 * m + gj, start_col] = 1
        
        # V row: gj*p + gj (j is fixed, so j=0 in local, but global j is gj)
        # Wait, V index is j*p + k.
        # Here j is the middle dimension.
        # In 2x1x2, the middle dimension size is 1.
        # So V has only one row corresponding to that j.
        V[gj * p + 0, start_col] = 1 # Local k=0? No, V is for B[j,k].
        # B is 1x2. So B[0,0] and B[0,1].
        # V row index is j*p + k.
        # We need V to cover both k=0 and k=1?
        # No, V is a vector.
        # If V = [1], it multiplies with B[0,0] and B[0,1]?
        # No, V is part of the decomposition.
        # V_{j k \alpha}.
        # If we use 1 column, V must have non-zero entries for both k=0 and k=1?
        # If V[0]=1, V[1]=1.
        # Then V[gj*p + 0, start_col] = 1
        # V[gj*p + 1, start_col] = 1
        
        V[gj * p + 0, start_col] = 1
        V[gj * p + 1, start_col] = 1
        
        # W rows: gk0*n + gi0, gk0*n + gi1, gk1*n + gi0, gk1*n + gi1
        W[gk0 * n + gi0, start_col] = 1
        W[gk0 * n + gi1, start_col] = 1
        W[gk1 * n + gi0, start_col] = 1
        W[gk1 * n + gi1, start_col] = 1
        
        return start_col + 1

    # Block 1: A1 B11
    col = add_2x2x2_block(0, 0, 0, col)
    # Block 2: A1 B12
    col = add_2x2x2_block(0, 0, 2, col)
    # Block 3: A1 B13
    col = add_2x2x2_block(0, 0, 4, col)
    # Block 4: A2 B21
    col = add_2x2x2_block(0, 2, 0, col)
    # Block 5: A2 B22
    col = add_2x2x2_block(0, 2, 2, col)
    # Block 6: A2 B23
    col = add_2x2x2_block(0, 2, 4, col)
    
    # Block 7: A3 B31
    col = add_2x1x2_block(0, 4, 0, col)
    # Block 8: A3 B32
    col = add_2x1x2_block(0, 4, 2, col)
    # Block 9: A3 B33
    col = add_2x1x2_block(0, 4, 4, col)
    
    # Verify
    # T[i,j,k] = sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
    # Should be 1 for all i,j,k?
    # No, the tensor for matrix multiplication is 1 for all i,j,k.
    # So we check if the sum is 1.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for alpha in range(col):
                    s += U[i * m + j, alpha] * V[j * p + k, alpha] * W[k * n + i, alpha]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback
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
# EVOLVE-BLOCK-END
