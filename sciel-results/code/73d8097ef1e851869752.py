"""Matrix multiplication 4x5x6 tensor decomposition using Strassen-based embedding."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]


def _strassen_2x2_factors():
    """Return U, V, W factors for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    U rows: A11, A12, A21, A22
    V rows: B11, B12, B21, B22
    W rows: C11, C21, C12, C22 (indexed by k*2+i: C11->0, C21->1, C12->2, C22->3)
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, 1],    # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, -1],   # A22
    ], dtype=np.int64)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 0, 1, 0],    # B12
        [0, 0, 0, 1, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1],   # B22
    ], dtype=np.int64)
    # W indexed by W[k*2+i]: C11->0, C21->1, C12->2, C22->3
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11 (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],   # C22 (k=1,i=1)
    ], dtype=np.int64)
    return U, V, W


def _build_strassen_recursive(S):
    """Build Strassen decomposition for S x S x S.
    Returns U, V, W as arrays of shape (S*S, rank).
    U indexed by i*S+j, V by j*S+k, W by k*S+i.
    """
    if S == 2:
        U2, V2, W2 = _strassen_2x2_factors()
        # Transpose: from (4, 7) to (4, 7) - already correct
        # U2[i, c] where i is row of A (A11=0, A12=1, A21=2, A22=3)
        # V2[j, c] where j is row of B (B11=0, B12=1, B21=2, B22=3)
        # W2[k, c] where k is W index (C11=0, C21=1, C12=2, C22=3)
        return U2, V2, W2
    
    half = S // 2
    U_sub, V_sub, W_sub = _build_strassen_recursive(half)
    sub_rank = U_sub.shape[1]
    
    # Strassen product definitions:
    # t=0: M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # t=1: M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # t=2: M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # t=3: M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # t=4: M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # t=5: M6 = (A21-A11)(B11+B12) -> C22 += M6
    # t=6: M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # A block coefficients: dict from block_idx -> coeff
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    A_coeffs = [
        {0: 1, 3: 1},   # t=0
        {2: 1, 3: 1},   # t=1
        {0: 1},         # t=2
        {3: 1},         # t=3
        {0: 1, 1: 1},   # t=4
        {2: 1, 0: -1},  # t=5
        {1: 1, 3: -1},  # t=6
    ]
    
    # B block coefficients
    B_coeffs = [
        {0: 1, 3: 1},   # t=0
        {0: 1},         # t=1
        {1: 1, 3: -1},  # t=2
        {2: 1, 0: -1},  # t=3
        {3: 1},         # t=4
        {0: 1, 1: 1},   # t=5
        {2: 1, 3: 1},   # t=6
    ]
    
    # C block coefficients: block 0=C11, 1=C12, 2=C21, 3=C22
    C_coeffs = [
        {0: 1, 3: 1},   # t=0
        {2: 1, 3: -1},  # t=1
        {1: 1, 3: 1},   # t=2
        {0: 1, 2: 1},   # t=3
        {0: -1, 1: 1},  # t=4
        {3: 1},         # t=5
        {0: 1},         # t=6
    ]
    
    total_rank = 7 * sub_rank
    U_full = np.zeros((S * S, total_rank), dtype=np.int64)
    V_full = np.zeros((S * S, total_rank), dtype=np.int64)
    W_full = np.zeros((S * S, total_rank), dtype=np.int64)
    
    for t in range(7):
        col_offset = t * sub_rank
        for c in range(sub_rank):
            col = col_offset + c
            
            # U contributions
            for a_block, a_coeff in A_coeffs[t].items():
                bi = a_block // 2
                bj = a_block % 2
                # Sub indices: i*half+j for i,j in range(half)
                # Full indices: (bi*half+i)*S + (bj*half+j)
                for i in range(half):
                    for j in range(half):
                        sub_idx = i * half + j
                        full_idx = (bi * half + i) * S + (bj * half + j)
                        U_full[full_idx, col] += a_coeff * U_sub[sub_idx, c]
            
            # V contributions
            for b_block, b_coeff in B_coeffs[t].items():
                bi = b_block // 2
                bj = b_block % 2
                for i in range(half):
                    for j in range(half):
                        sub_idx = i * half + j
                        full_idx = (bi * half + i) * S + (bj * half + j)
                        V_full[full_idx, col] += b_coeff * V_sub[sub_idx, c]
            
            # W contributions
            # W is indexed by k*S+i where k is col, i is row
            for c_block, c_coeff in C_coeffs[t].items():
                bi = c_block // 2  # row block
                bj = c_block % 2   # col block
                for i in range(half):
                    for k in range(half):
                        sub_idx = k * half + i
                        i_full = bi * half + i
                        k_full = bj * half + k
                        full_idx = k_full * S + i_full
                        W_full[full_idx, col] += c_coeff * W_sub[sub_idx, c]
    
    return U_full, V_full, W_full


def _embed_and_extract(n, m, p, S):
    """Embed n x m x p into S x S x S using Strassen, then extract.
    Returns U, V, W as lists of lists with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    U_full, V_full, W_full = _build_strassen_recursive(S)
    rank = U_full.shape[1]
    
    # Extract rows
    U_rows = np.array([i * S + j for i in range(n) for j in range(m)])
    V_rows = np.array([j * S + k for j in range(m) for k in range(p)])
    W_rows = np.array([k * S + i for k in range(p) for i in range(n)])
    
    U_out = U_full[U_rows, :]
    V_out = V_full[V_rows, :]
    W_out = W_full[W_rows, :]
    
    # Remove zero columns
    non_zero = np.any(U_out != 0, axis=0) | np.any(V_out != 0, axis=0) | np.any(W_out != 0, axis=0)
    U_out = U_out[:, non_zero]
    V_out = V_out[:, non_zero]
    W_out = W_out[:, non_zero]
    
    # Convert to lists
    return U_out.tolist(), V_out.tolist(), W_out.tolist()


def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify_tensor(U, V, W, n, m, p):
    """Verify the decomposition by checking the tensor identity.
    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
    """
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(t_val - 1.0) > 1e-9:
                    return False
    return True


def _verify_random(U, V, W, n, m, p, max_checks=20):
    """Verify by checking random matrix multiplications."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for _ in range(max_checks):
        A = np.random.randint(-10, 11, size=(n, m)).astype(np.float64)
        B = np.random.randint(-10, 11, size=(m, p)).astype(np.float64)
        C_expected = A @ B
        
        # Compute from decomposition
        C_decomp = np.zeros((n, p), dtype=np.float64)
        for c in range(rank):
            # U_c is a vector of length n*m, reshaped to (n, m)
            U_c = U_arr[:, c].reshape(n, m)
            V_c = V_arr[:, c].reshape(m, p)
            W_c = W_arr[:, c].reshape(p, n)
            
            # The bilinear form: C[i,k] += W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            # Actually, for the tensor decomposition:
            # C[i,k] = sum_c W[k*n+i, c] * (U_c^T A_vec) * (V_c^T B_vec)
            # where A_vec is A flattened and B_vec is B flattened
            # But this is only correct if the decomposition is of the form:
            # C = sum_c W_c * (U_c^T A) * (V_c^T B)
            # which is NOT the standard tensor decomposition.
            
            # The standard tensor decomposition is:
            # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # And C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k]
            # = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
            # Wait, that's not right either. Let me think more carefully.
            
            # The tensor T has indices (i, j, k) and T_{i,j,k} = 1 for matrix multiplication.
            # The decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c
            # where (U_c ⊗ V_c ⊗ W_c)_{i,j,k} = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Now, C[i,k] = sum_j A[i,j] * B[j,k]
            # And we want to express this using the tensor decomposition.
            # The key insight is that for matrix multiplication, T_{i,j,k} = 1 for all valid (i,j,k).
            # So C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k]
            # = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
            # = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            
            # So the formula is:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            
            # Let's compute this:
            a_proj = np.array([np.sum(U_arr[i * m + j, c] * A[i, j] for j in range(m)) for i in range(n)])
            b_proj = np.array([np.sum(V_arr[j * p + k, c] * B[j, k] for j in range(m)) for k in range(p)])
            
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W_arr[k * n + i, c] * a_proj[i] * b_proj[k]
        
        if not np.allclose(C_expected, C_decomp, atol=1e-9):
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Find the smallest power of 2 >= max(n, m, p)
    min_S = max(n, m, p)
    S = 1
    while S < min_S:
        S *= 2
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Try S, S*2, etc.
    for trial_S in [S, S * 2, S * 4]:
        if time.time() > deadline - 20:
            break
        
        try:
            U, V, W = _embed_and_extract(n, m, p, trial_S)
            rank = len(U[0]) if U else 0
            
            if rank > 0 and rank < best_rank:
                # Verify using tensor identity (fast)
                if _verify_tensor(U, V, W, n, m, p):
                    best_result = (U, V, W)
                    best_rank = rank
                    print(f"Found rank {rank} with S={trial_S}")
        except Exception as e:
            print(f"Error with S={trial_S}: {e}")
    
    # If no better result found, use schoolbook
    if best_result is None:
        U, V, W = _make_schoolbook(n, m, p)
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}
