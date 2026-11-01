"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_222():
    """
    Build U, V, W for 2x2x2 Strassen decomposition.
    A (2,2), B (2,2), C (2,2).
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Indices for A, B, C are (row, col)
    # A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3
    
    # U maps A indices to 7 products
    # V maps B indices to 7 products
    # W maps 7 products to C indices
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1; W[3, 1] = 1  # C21, C22
    
    # M3
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[1, 4] = 1; W[2, 4] = 1  # C12, C21
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[0, 5] = 1; W[1, 5] = 1  # C11, C12
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[1, 6] = 1; W[2, 6] = 1  # C12, C21
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that sum_c U[:,c] * V[:,c]^T * W[:,c]^T = T
    where T is the matrix multiplication tensor for (n,m) x (m,p) -> (n,p).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    # Tensor T has indices (i, k, j) corresponding to C[i,k] = sum_j A[i,j] B[j,k]
    # But the trace convention is W[k*n+i, r].
    # Let's construct the tensor T as a 3D array T[i, j, k] = 1 if valid.
    # Actually, the tensor is T_{i,j,k} = 1 for all i,j,k.
    # The decomposition is sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1.
    
    # Let's check this directly.
    # Reshape U, V, W into tensors
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute sum over r
    # This is a tensor contraction.
    # We can do this via einsum
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    try:
        result = np.einsum('ijr,jkr,ki r->ijk', U_t, V_t, W_t)
        # Note: W_t indices are (p, n, rank) -> (k, i, r)
        # So W_t[k, i, r]
        # The result should be 1 for all i, j, k.
        if not np.allclose(result, 1.0, atol=tol):
            return False
    except Exception:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try Rank 28 (Strassen based)
    # A (2,4), B (4,4), C (2,4)
    # Split A into A1 (2,2), A2 (2,2) horizontally.
    # Split B into B11 (2,2), B12 (2,2); B21 (2,2), B22 (2,2) vertically/horizontally.
    # B = [[B11, B12], [B21, B22]] where rows are split into 2,2 and cols into 2,2.
    # C = A1 * [[B11, B12]] + A2 * [[B21, B22]]?
    # No, A is (2,4). A = [A1, A2] where A1, A2 are (2,2).
    # B is (4,4). B = [[B11, B12], [B21, B22]]?
    # Let B rows be 0,1,2,3.
    # A = [A_0, A_1] where A_0 is cols 0,1 of A? No, A is 2 rows, 4 cols.
    # A = [A_left, A_right] where A_left is (2,2) cols 0,1 and A_right is (2,2) cols 2,3.
    # B is (4,4). Rows 0,1 correspond to A_left. Rows 2,3 correspond to A_right.
    # B = [[B_top_left, B_top_right], [B_bot_left, B_bot_right]]
    # where each block is 2x2.
    # C = A_left * B_top + A_right * B_bot
    # where B_top is B[0:2, :] (2,4) and B_bot is B[2:4, :] (2,4).
    # C = [C_left, C_right] where C_left = A_left * B_top_left + A_right * B_bot_left ?
    # No. C[i, k] = sum_{j=0}^3 A[i, j] B[j, k].
    # j=0,1 uses A[i,0], A[i,1] and B[0,:], B[1,:].
    # j=2,3 uses A[i,2], A[i,3] and B[2,:], B[3,:].
    
    # Let's use 4 Strassen blocks of 2x2x2.
    # Block 1: A_left (2,2) * B_top_left (2,2) -> contributes to C_left (2,2)
    # Block 2: A_left (2,2) * B_top_right (2,2) -> contributes to C_right (2,2)
    # Block 3: A_right (2,2) * B_bot_left (2,2) -> contributes to C_left (2,2)
    # Block 4: A_right (2,2) * B_bot_right (2,2) -> contributes to C_right (2,2)
    
    # Indices:
    # A: i in 0..1, j in 0..3.
    # A_left: j=0,1. A_right: j=2,3.
    # B: j in 0..3, k in 0..3.
    # B_top: j=0,1. B_bot: j=2,3.
    # B_top_left: k=0,1. B_top_right: k=2,3.
    # B_bot_left: k=0,1. B_bot_right: k=2,3.
    # C: i in 0..1, k in 0..3.
    # C_left: k=0,1. C_right: k=2,3.
    
    # Strassen factors for 2x2x2
    U_s, V_s, W_s = _build_strassen_222()
    
    U_28 = np.zeros((n * m, 28))
    V_28 = np.zeros((m * p, 28))
    W_28 = np.zeros((p * n, 28))
    
    block_sizes = [
        (0, 1, 0, 1),  # A_left, B_top_left -> C_left
        (0, 1, 2, 3),  # A_left, B_top_right -> C_right
        (2, 3, 0, 1),  # A_right, B_bot_left -> C_left
        (2, 3, 2, 3),  # A_right, B_bot_right -> C_right
    ]
    
    offset = 0
    for j_min, j_max, k_min, k_max in block_sizes:
        # j in [j_min, j_max-1], k in [k_min, k_max-1]
        # A indices: i*m + j. i in 0..1.
        # B indices: j*p + k.
        # C indices: k*n + i.
        
        for r in range(7):
            # U
            for i in range(2):
                for ja in range(2):
                    j = j_min + ja
                    # U_s[i, ja, r] is the coefficient for A[i, ja] in product r
                    coeff = U_s[i, ja, r]
                    if coeff != 0:
                        U_28[i * m + j, offset + r] = coeff
                        
            # V
            for jb in range(2):
                j = j_min + jb
                for kb in range(2):
                    k = k_min + kb
                    coeff = V_s[jb, kb, r]
                    if coeff != 0:
                        V_28[j * p + k, offset + r] = coeff
                        
            # W
            for i in range(2):
                for kb in range(2):
                    k = k_min + kb
                    coeff = W_s[kb, i, r]
                    if coeff != 0:
                        W_28[k * n + i, offset + r] = coeff
                        
        offset += 7
        
    if _verify_decomposition(U_28, V_28, W_28, n, m, p):
        best_U, best_V, best_W = U_28, V_28, W_28
        best_rank = 28
        
    # If we have time, try to improve? 
    # The prompt says "Keep an EXPLICIT deadline guard".
    # Since we have a static construction, we don't need much search.
    # But we can check for any trivial improvements or just return.
    
    if time.time() > deadline:
        pass # We are just returning static matrices, so this is fast.
        
    # Convert to lists to match interface if needed, or return arrays.
    # The interface says: 'U', 'V', 'W': factor matrices of shapes ...
    # The seed program returned lists of lists.
    # Let's return lists of lists to be safe, or arrays if the evaluator accepts.
    # The seed used: U = [[0] * rank for _ in range(n * m)]
    # So it expects a list of lists.
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
