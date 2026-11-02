"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 via Strassen on 2x2 blocks.

This program constructs a rank-42 decomposition by splitting the 4-dimension of
the inner product into two 2-dimensions. Each resulting 2x2 @ 2x6 subproblem
is further split into three 2x2 @ 2x2 blocks, each handled by Strassen's algorithm
(rank 7). Total rank = 2 * 3 * 7 = 42.

The output follows the official trace convention W[k*n+i, r] where
k is the column index and i is the row index of the output matrix C.
"""
import numpy as np

import time

def construct():
    n, m, p = 2, 4, 6
    rank = 42

    # Pre-allocate factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen 2x2 @ 2x2: 7 rank-1 terms
    # Each term is a tuple (U_sparse, V_sparse, W_sparse)
    # U_sparse: list of (local_idx, coeff)
    # local_idx: 0=a11, 1=a12, 2=a21, 3=a22
    # V_sparse: list of (local_idx, coeff)
    # local_idx: 0=b11, 1=b12, 2=b21, 3=b22
    # W_sparse: list of (local_idx, coeff)
    # local_idx: 0=c11, 1=c12, 2=c21, 3=c22
    
    strassen_terms = [
        # M1 = (a11+a22)(b11+b22) -> c11, c22
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a21+a22)b11 -> c21, -c22
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a11(b12-b22) -> c12, c22
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a22(b21-b11) -> c11, c21
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a11+a12)b22 -> -c11, c12
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a21-a11)(b11+b12) -> c22
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a12-a22)(b21+b22) -> c11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    current_rank = 0
    # side: 0 for left 2 columns of A and top 2 rows of B
    #       1 for right 2 columns of A and bottom 2 rows of B
    for side in range(2):
        u_col_off = side * 2
        v_row_off = side * 2

        # col_block: 0, 1, 2 for the three 2-column blocks of B
        for col_block in range(3):
            v_col_off = col_block * 2

            for u_sparse, v_sparse, w_sparse in strassen_terms:
                # Process U
                for local_a_idx, coeff in u_sparse:
                    a_row = local_a_idx // 2
                    a_col_local = local_a_idx % 2
                    a_col_global = u_col_off + a_col_local
                    u_row_global = a_row * m + a_col_global
                    U[u_row_global][current_rank] += coeff

                # Process V
                for local_b_idx, coeff in v_sparse:
                    b_row_local = local_b_idx // 2
                    b_col_local = local_b_idx % 2
                    b_row_global = v_row_off + b_row_local
                    b_col_global = v_col_off + b_col_local
                    v_row_global = b_row_global * p + b_col_global
                    V[v_row_global][current_rank] += coeff

                # Process W
                for local_c_idx, coeff in w_sparse:
                    c_row_local = local_c_idx // 2
                    c_col_local = local_c_idx % 2
                    c_row_global = c_row_local
                    c_col_global = v_col_off + c_col_local
                    # Trace convention: W[k*n+i, r]
                    w_row_global = c_col_global * n + c_row_global
                    W[w_row_global][current_rank] += coeff

                current_rank += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test to ensure correctness
    import numpy as np
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test matrices
    A = np.random.randn(2, 4)
    B = np.random.randn(4, 6)
    C_direct = A @ B
    
    # Reconstruct C using the decomposition
    # T_r = sum_i U[i,r] A[i] * sum_j V[j,r] B[j]
    # C = sum_r T_r * W[:,r] (reshaped)
    
    # U has shape (8, 42), V has shape (24, 42), W has shape (12, 42)
    # A flattened is shape (8,), B flattened is shape (24,)
    # We need to compute:
    # S = np.einsum('ir,jr,r -> i j', U, V, ???) - no, that's not right.
    
    # The tensor decomposition is:
    # C_{i k} = sum_{j} A_{i j} B_{j k}
    # = sum_{r=1}^{R} W_{k i r} * (sum_{j} U_{j r} A_j) * (sum_{k'} V_{k' r} B_{k'})
    
    # Let's flatten A and B
    A_flat = A.flatten() # size 8
    B_flat = B.flatten() # size 24
    
    # Compute the intermediate products
    # u_proj = U @ A_flat (shape 42)
    # v_proj = V @ B_flat (shape 42)
    u_proj = U @ A_flat
    v_proj = V @ B_flat
    
    # The product for each rank term is u_proj[r] * v_proj[r]
    # Then C_flat = W.T @ (u_proj * v_proj) ?
    # C_{k i} = sum_r W_{k i r} * u_proj[r] * v_proj[r]
    # Let c_vals = u_proj * v_proj (shape 42)
    # C_flat (size 12) = W.T @ c_vals
    
    c_vals = u_proj * v_proj
    C_flat = W.T @ c_vals
    C_decomp = C_flat.reshape(2, 6)
    
    err = np.linalg.norm(C_direct - C_decomp)
    if err > 1e-9:
        print(f"Warning: Reconstruction error {err}")
    else:
        print("Decomposition verified successfully.")
