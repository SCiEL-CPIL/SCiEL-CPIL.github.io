import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    rank = len(U[0]) if U else 0
    
    # Build tensor from decomposition
    T = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for r in range(rank):
        for a in range(dim_A):
            if U[a][r] != 0:
                for b in range(dim_B):
                    if V[b][r] != 0:
                        uv = U[a][r] * V[b][r]
                        for c in range(dim_C):
                            if W[c][r] != 0:
                                T[a, b, c] += uv * W[c][r]
    
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    # Check no spurious entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Already checked above
    # Check that non-tensor entries are zero (optional, for strictness)
    # The tensor has 1s at (i*m+j, j*p+k, k*n+i) for all valid i,j,k
    # All other entries should be 0
    tensor_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.add((a, b, c))
    
    for a in range(dim_A):
        for b in range(dim_B):
            for c in range(dim_C):
                if (a, b, c) not in tensor_entries and abs(T[a, b, c]) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _strassen_44(n, m, p):
    """
    Construct a Strassen-based decomposition for 3x4 x 4x4.
    Pad A to 4x4, use 4x4x4 Strassen (rank 7), take top 3 rows of output.
    """
    # 4x4x4 Strassen decomposition
    # Split 4x4 into 2x2 blocks
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 3x4 x 4x4:
    # A is 3x4, pad to 4x4 with zero row: A_padded = [[A11, A12], [A21, A22], [0, 0]]
    # where A11 is 2x2 (rows 0-1, cols 0-1), A12 is 2x2 (rows 0-1, cols 2-3)
    # A21 is 1x2 (row 2, cols 0-1), A22 is 1x2 (row 2, cols 2-3)
    #
    # This doesn't fit the 2x2 block structure cleanly. Let me use a different approach.
    #
    # Alternative: Use the fact that we can compute 3x4 * 4x4 as:
    # Split B into columns: B = [B_left, B_right] where each is 4x2
    # C = A * B = [A*B_left, A*B_right]
    # Each A*B_left is 3x2, which can be computed with rank 3*2*2=12 schoolbook
    # Total rank = 24. Not better than 48.
    #
    # Better: Use block Strassen on the 4x4 part.
    # Think of A as 3x4, B as 4x4.
    # Split B into 2x2 blocks of 2x2: B = [[B11, B12], [B21, B22]]
    # Split A into two parts: A_top (2x4) and A_bottom (1x4)
    # A_top = [A_top_left, A_top_right] where each is 2x2
    # A_bottom = [A_bottom_left, A_bottom_right] where each is 1x2
    #
    # C_top = A_top * B (2x4)
    # C_bottom = A_bottom * B (1x4)
    #
    # C_top_left = A_top_left * B11 + A_top_right * B21 (2x2)
    # C_top_right = A_top_left * B12 + A_top_right * B22 (2x2)
    # C_bottom_left = A_bottom_left * B11 + A_bottom_right * B21 (1x2)
    # C_bottom_right = A_bottom_left * B12 + A_bottom_right * B22 (1x2)
    #
    # This is getting complex. Let me just use a known rank-44 decomposition.
    #
    # Actually, let me try a simpler approach: use the schoolbook but with some sharing.
    # For 3x4 x 4x4, we can group the multiplications.
    #
    # Let me just return schoolbook for now and try to reduce it.
    return _schoolbook(n, m, p)

def _gaussian_reduce(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    """
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    rank = len(U[0]) if U else 0
    
    # Build the tensor
    T = np.zeros((dim_A, dim_B, dim_C), dtype=float)
    for r in range(rank):
        for a in range(dim_A):
            if U[a][r] != 0:
                for b in range(dim_B):
                    if V[b][r] != 0:
                        uv = U[a][r] * V[b][r]
                        for c in range(dim_C):
                            if W[c][r] != 0:
                                T[a, b, c] += uv * W[c][r]
    
    # Try to find a lower rank decomposition via SVD of mode-1 unfolding
    T_1 = T.reshape(dim_A, dim_B * dim_C)
    U1, s1, Vt1 = np.linalg.svd(T_1, full_matrices=False)
    tol = 1e-10
    rank_T1 = int(np.sum(s1 > tol))
    
    if rank_T1 >= rank:
        return U, V, W
    
    # Try to construct a decomposition of rank rank_T1
    Vt1_2d = Vt1[:rank_T1, :].reshape(rank_T1, dim_B, dim_C)
    
    all_rank_1 = True
    for r in range(rank_T1):
        slice_matrix = Vt1_2d[r, :, :]
        s_slice = np.linalg.svd(slice_matrix, compute_uv=False)
        if len(s_slice) > 1 and s_slice[1] > tol:
            all_rank_1 = False
            break
    
    if not all_rank_1:
        return U, V, W
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r in range(rank_T1):
        U_cols.append(U1[:, r] * s1[r])
        slice_matrix = Vt1_2d[r, :, :]
        u_slice, s_slice, vt_slice = np.linalg.svd(slice_matrix, full_matrices=False)
        v_vec = u_slice[:, 0] * s_slice[0]
        w_vec = vt_slice[0, :]
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    U_arr = np.column_stack(U_cols)
    V_arr = np.column_stack(V_cols)
    W_arr = np.column_stack(W_cols)
    
    U_new = [[float(x) for x in row] for row in U_arr]
    V_new = [[float(x) for x in row] for row in V_arr]
    W_new = [[float(x) for x in row] for row in W_arr]
    
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = sb["U"], sb["V"], sb["W"]
    best_rank = len(best_U[0])
    
    # Try to reduce using Gaussian elimination
    try:
        U_red, V_red, W_red = _gaussian_reduce(best_U, best_V, best_W, n, m, p, deadline)
        rank_red = len(U_red[0]) if U_red else best_rank
        if rank_red < best_rank and _verify(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = rank_red
    except Exception:
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
