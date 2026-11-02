import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to construct Rank 88 using Strassen 2x2x2 blocks
    # Strategy:
    # A (3x4) -> [A_top (2x2), A_bot (1x2)] x [A_L (3x2), A_R (3x2)]?
    # Let's split m=4 into m1=2, m2=2.
    # A = [A_L (3x2), A_R (3x2)]
    # B = [B_L (2x8); B_R (2x8)]
    # C = A_L B_L + A_R B_R
    
    # For A_L (3x2) * B_L (2x8):
    # Split n=3 into n1=2, n2=1.
    # A_L = [A_L_top (2x2); A_L_bot (1x2)]
    # C_L = A_L_top B_L + A_L_bot B_L
    
    # Term 1: A_L_top (2x2) * B_L (2x8)
    # This is 4 instances of 2x2x2 multiplication (treating B_L as 4 columns of 2x2? No, B_L is 2x8.
    # B_L = [B_L_1 (2x2), B_L_2 (2x2), B_L_3 (2x2), B_L_4 (2x2)]
    # A_L_top * B_L = A_L_top * B_L_1 + ... + A_L_top * B_L_4
    # Each is 2x2x2. Rank 7 each. Total 28.
    
    # Term 2: A_L_bot (1x2) * B_L (2x8)
    # Rank 16 (Schoolbook).
    
    # Total Rank = (28 + 16) * 2 = 88.
    
    # Strassen 2x2x2 Matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # U, V, W are 4x7, 4x7, 4x7
    U_222 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0]
    ]
    V_222 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, -1, 0, 0, 0, 0, 1],
        [0, 0, 1, -1, 0, 0, 0]
    ]
    W_222 = [
        [0.5, 0.5, 0, 0, 0, -0.5, 0.5],
        [0, 0, 0.5, 0.5, 1, 0, 0],
        [0.5, -0.5, 0, 0, 0, 0, 0.5],
        [0, 0, 0.5, -0.5, 0, 0, 0]
    ]
    
    # Helper to verify 2x2x2
    def verify_222(U, V, W):
        # T = sum_k U_i k * V_j k * W_l k
        # Check if it equals A*B
        T = np.zeros((2, 2, 2))
        for k in range(7):
            u_col = np.array(U[:, k])
            v_col = np.array(V[:, k])
            w_col = np.array(W[:, k])
            # Outer product u_col (2) x v_col (2) x w_col (2)
            # T[i, j, l] += u_col[i] * v_col[j] * w_col[l]
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        T[i, j, l] += u_col[i] * v_col[j] * w_col[l]
        
        # Target tensor: T[i, j, k] = 1 if A[i, k] and B[k, j] contribute to C[i, j]
        # Multiplication tensor T_mul[i, j, k] is non-zero when?
        # C[i, j] = sum_k A[i, k] B[k, j]
        # The tensor is T[i, j, k] = 1 if we are computing C[i, j] using A[i, k] and B[k, j].
        # In our formulation, we sum over the "inner" index of the rank decomposition.
        # Here, the rank index is the 7th dimension.
        # The "input" dimensions are i (A row), k (A col / B row), j (B col).
        # Wait, the standard tensor T[i, k, j] for A(i,k)B(k,j) -> C(i,j).
        # My U, V, W above are indexed by (row/col) and rank.
        # Let's check the indices.
        # U[i, r] corresponds to A[i, :]? No.
        # A is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
        # U is 4x7. Row 0 is (0,0)? Row 1 is (0,1)? Row 2 is (1,0)? Row 3 is (1,1)?
        # V is 4x7. Row 0 is (0,0)? (B row 0, col 0)?
        # B is 2x2. Indices (k, j).
        # W is 4x7. Row 0 is (0,0)? (C row 0, col 0)?
        
        # Let's assume standard row-major order for the 4 rows.
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1) -> k is row, j is col.
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Check T[i, k, j] == sum_r U[i, r] V[k, r] W[j, r]?
        # No, the decomposition is:
        # C[i, j] = sum_r (sum_k U[i, k] A[k, ?])?
        
        # Standard Tensor Decomposition:
        # T = sum_r u_r \otimes v_r \otimes w_r
        # Where u_r is vector of coeffs for A, v_r for B, w_r for C.
        # My U is 4x7. If U[i, r] is the coeff for A_flat[i] in term r.
        # Then A_contrib_r = sum_i U[i, r] A[i]
        # B_contrib_r = sum_k V[k, r] B[k]
        # C[i, j] += W[i, r] * A_contrib_r * B_contrib_r?
        
        # Let's re-verify the Strassen matrices with this convention.
        # A_flat = [a, b, c, d]
        # B_flat = [e, f, g, h] (Row major: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h)
        
        # U[0] = [1, 0, 1, 0] -> a + c? No, a+d?
        # My U[0] = [1, 0, 1, 0] -> a + c.
        # My U[3] = [0, 1, 0, 1] -> b + d.
        # This does not match Strassen $a+d$.
        
        # I will not risk incorrect matrices. I will use Schoolbook.
        # But I will try to implement a simple, correct Winograd 2x2x2 if I can find one.
        
        # For now, return schoolbook to ensure validity.
        return schoolbook()

# EVOLVE-BLOCK-END
