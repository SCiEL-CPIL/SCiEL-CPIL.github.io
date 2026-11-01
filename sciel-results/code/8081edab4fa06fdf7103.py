import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try Strassen-based rank 42
    # Strassen 2x2x2 factors
    # U_X (4x7), V_Y (4x7), W_Z (4x7)
    U_str = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 0, -1]
    ]
    V_str = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # W_Z rows correspond to Z00, Z01, Z10, Z11
    # Using formulas:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M2 + M3
    # Z10 = M6 - M3
    # Z11 = M1 - M4 + M5 - M2
    W_str = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [1, -1, 0, -1, 1, 0, 0]
    ]

    # Map to 2x4x6
    # Blocks:
    # 1. A[0:2,0:2] B[0:2,0:2] -> C[0:2,0:2]
    # 2. A[0:2,0:2] B[0:2,2:4] -> C[0:2,2:4]
    # 3. A[0:2,0:2] B[0:2,4:6] -> C[0:2,4:6]
    # 4. A[0:2,2:4] B[2:4,0:2] -> C[0:2,0:2]
    # 5. A[0:2,2:4] B[2:4,2:4] -> C[0:2,2:4]
    # 6. A[0:2,2:4] B[2:4,4:6] -> C[0:2,4:6]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, offset):
        # A block is 2x2, indices a_row_start..a_row_start+1, a_col_start..a_col_start+1
        # B block is 2x2, indices b_row_start..b_row_start+1, b_col_start..b_col_start+1
        # C block is 2x2, indices c_row_start..c_row_start+1, c_col_start..c_col_start+1
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    u_idx = (a_row_start + i) * m + (a_col_start + j)
                    U[u_idx][offset + r] += U_str[i * 2 + j][r]
            # V
            for j in range(2):
                for k in range(2):
                    v_idx = (b_row_start + j) * p + (b_col_start + k)
                    V[v_idx][offset + r] += V_str[j * 2 + k][r]
            # W
            for k in range(2):
                for i in range(2):
                    w_idx = (c_col_start + k) * n + (c_row_start + i)
                    W[w_idx][offset + r] += W_str[k * 2 + i][r]

    # Block 1
    add_block(0, 0, 0, 0, 0, 0, 0)
    # Block 2
    add_block(0, 0, 0, 2, 0, 2, 7)
    # Block 3
    add_block(0, 0, 0, 4, 0, 4, 14)
    # Block 4
    add_block(0, 2, 2, 0, 0, 0, 21)
    # Block 5
    add_block(0, 2, 2, 2, 0, 2, 28)
    # Block 6
    add_block(0, 2, 2, 4, 0, 4, 35)

    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want T[i,j,k] = 1 if this is a valid matmul decomposition?
        # No, the decomposition represents the multiplication tensor.
        # The tensor for matmul is T[i,j,k] = 1 for all i,j,k? No.
        # The multiplication tensor is defined by T_{ijk} = A_{ij} B_{jk}.
        # The rank decomposition is T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
        # For the standard matmul, the tensor entries are 1 if we consider the basis?
        # No, the tensor is fixed. T_{ijk} is the coefficient of A_{ij}B_{jk} in the product?
        # Actually, the task is to decompose the tensor of the bilinear map.
        # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
        # The tensor is $\mathcal{T} = \sum_{i,j,k} A_{ij} B_{jk} C_{ki}$?
        # The standard tensor for matrix multiplication $n \times m \times p$ is:
        # $T_{ijk} = 1$ if we map to the identity?
        # Actually, the tensor is defined such that $\langle T, A \otimes B \otimes C \rangle = \text{tr}(A B C)$.
        # Or $\sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = \text{tr}(ABC)$.
        # The entries of the multiplication tensor are $T_{ijk} = 1$ for all $i,j,k$?
        # No, $T_{ijk}$ is the coefficient in the expansion.
        # For matmul, $T_{ijk} = 1$ if the indices match the cycle?
        # Actually, the tensor is simply $T_{ijk} = 1$ for all $i,j,k$ is wrong.
        # The tensor is $T = \sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}$?
        # Let's check the schoolbook.
        # Schoolbook: $U_{ij,r}=1, V_{jk,r}=1, W_{ki,r}=1$ for unique r per (i,j,k).
        # Then $\sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1$ for each (i,j,k).
        # So the target tensor is $T_{ijk} = 1$ for all $i,j,k$.
        
        # Let's verify if our U,V,W produce T_{ijk} = 1.
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    T[i,j,k] = val
                    if abs(val - 1.0) > 1e-9:
                        return {"U": U_school, "V": V_school, "W": W_school}
        
        return {"U": U, "V": V, "W": W}
    except:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
