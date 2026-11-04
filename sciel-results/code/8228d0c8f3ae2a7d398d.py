"""Matrix multiplication 4x5x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]


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
    """Verify the decomposition by checking the tensor identity."""
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


def _strassen_2x2_factors():
    """Return U, V, W factors for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    U rows: A11, A12, A21, A22 (indexed by i*2+j)
    V rows: B11, B12, B21, B22 (indexed by j*2+k)
    W rows: C11, C21, C12, C22 (indexed by k*2+i)
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A11 (i=0,j=0)
        [0, 0, 0, 0, 1, 0, 1],    # A12 (i=0,j=1)
        [0, 1, 0, 0, 0, 1, 0],    # A21 (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1],   # A22 (i=1,j=1)
    ], dtype=np.int64)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B11 (j=0,k=0)
        [0, 0, 1, 0, 0, 1, 0],    # B12 (j=0,k=1)
        [0, 0, 0, 1, 0, 0, 1],    # B21 (j=1,k=0)
        [1, 0, -1, 0, 1, 0, 1],   # B22 (j=1,k=1)
    ], dtype=np.int64)
    # W indexed by k*2+i: C11(k=0,i=0)->0, C21(k=0,i=1)->1, C12(k=1,i=0)->2, C22(k=1,i=1)->3
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11 (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],   # C22 (k=1,i=1)
    ], dtype=np.int64)
    return U, V, W


def _build_strassen(S):
    """Build Strassen decomposition for S x S x S.
    Returns U, V, W as arrays of shape (S*S, rank).
    U indexed by i*S+j, V by j*S+k, W by k*S+i.
    """
    if S == 2:
        return _strassen_2x2_factors()
    
    half = S // 2
    U_sub, V_sub, W_sub = _build_strassen(half)
    sub_rank = U_sub.shape[1]
    
    # Strassen product definitions:
    # t=0: M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # t=1: M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # t=2: M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # t=3: M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # t=4: M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # t=5: M6 = (A21-A11)(B11+B12) -> C22 += M6
    # t=6: M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    A_coeffs = [
        {0: 1, 3: 1},   # t=0
        {2: 1, 3: 1},   # t=1
        {0: 1},         # t=2
        {3: 1},         # t=3
        {0: 1, 1: 1},   # t=4
        {2: 1, 0: -1},  # t=5
        {1: 1, 3: -1},  # t=6
    ]
    
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
            for c_block, c_coeff in C_coeffs[t].items():
                bi = c_block // 2  # row block (i dimension)
                bj = c_block % 2   # col block (k dimension)
                for i in range(half):
                    for k in range(half):
                        sub_idx = k * half + i
                        i_full = bi * half + i
                        k_full = bj * half + k
                        full_idx = k_full * S + i_full
                        W_full[full_idx, col] += c_coeff * W_sub[sub_idx, c]
    
    return U_full, V_full, W_full


def _try_strassen_embedding(n, m, p, S):
    """Try embedding n x m x p into S x S x S using Strassen.
    Returns (U, V, W, rank) or None if no improvement.
    """
    U_full, V_full, W_full = _build_strassen(S)
    full_rank = U_full.shape[1]
    
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
    
    rank = U_out.shape[1]
    if rank == 0:
        return None
    
    # Verify
    if not _verify_tensor(U_out.tolist(), V_out.tolist(), W_out.tolist(), n, m, p):
        return None
    
    return U_out.tolist(), V_out.tolist(), W_out.tolist(), rank


def _try_block_partition(n, m, p):
    """Try block partitioning with Strassen on core and schoolbook on strips.
    
    Split A (n x m) into blocks, B (m x p) into blocks, C (n x p) into blocks.
    For 4x5x6, we can try various partitions.
    """
    best = None
    best_rank = n * m * p
    
    # Try splitting into 2x2 blocks of various sizes
    # For n=4, m=5, p=6:
    # Option 1: n split 2+2, m split 2+3, p split 3+3
    # Option 2: n split 2+2, m split 3+2, p split 3+3
    # etc.
    
    n_splits = [(2, 2)]
    m_splits = [(2, 3), (3, 2)]
    p_splits = [(3, 3)]
    
    for n1, n2 in n_splits:
        for m1, m2 in m_splits:
            for p1, p2 in p_splits:
                if n1 + n2 != n or m1 + m2 != m or p1 + p2 != p:
                    continue
                
                # Build decomposition
                # Core: A11 (n1 x m1) * B11 (m1 x p1) -> C11 (n1 x p1) using Strassen if possible
                # Strips: other blocks using schoolbook
                
                U_list = []
                V_list = []
                W_list = []
                
                # Core block A11 (n1 x m1) * B11 (m1 x p1) -> C11 (n1 x p1)
                # If n1, m1, p1 are all powers of 2 or can use Strassen
                if n1 == 2 and m1 == 2 and p1 == 2:
                    U2, V2, W2 = _strassen_2x2_factors()
                    # U2 is 4x7, indexed by i*2+j
                    # V2 is 4x7, indexed by j*2+k
                    # W2 is 4x7, indexed by k*2+i
                    for c in range(7):
                        U_c = [0] * (n * m)
                        V_c = [0] * (m * p)
                        W_c = [0] * (p * n)
                        
                        # Fill U_c for A11
                        for i in range(n1):
                            for j in range(m1):
                                U_c[i * m + j] = U2[i * m1 + j, c]
                        
                        # Fill V_c for B11
                        for j in range(m1):
                            for k in range(p1):
                                V_c[j * p + k] = V2[j * p1 + k, c]
                        
                        # Fill W_c for C11
                        for k in range(p1):
                            for i in range(n1):
                                W_c[k * n + i] = W2[k * n1 + i, c]
                        
                        U_list.append(U_c)
                        V_list.append(V_c)
                        W_list.append(W_c)
                else:
                    # Schoolbook for core
                    for i in range(n1):
                        for j in range(m1):
                            for k in range(p1):
                                U_c = [0] * (n * m)
                                V_c = [0] * (m * p)
                                W_c = [0] * (p * n)
                                U_c[i * m + j] = 1
                                V_c[j * p + k] = 1
                                W_c[k * n + i] = 1
                                U_list.append(U_c)
                                V_list.append(V_c)
                                W_list.append(W_c)
                
                # Strip A12 (n1 x m2) * B21 (m2 x p1) -> C11 (n1 x p1)
                for i in range(n1):
                    for j in range(m1, m):
                        for k in range(p1):
                            U_c = [0] * (n * m)
                            V_c = [0] * (m * p)
                            W_c = [0] * (p * n)
                            U_c[i * m + j] = 1
                            V_c[j * p + k] = 1
                            W_c[k * n + i] = 1
                            U_list.append(U_c)
                            V_list.append(V_c)
                            W_list.append(W_c)
                
                # Strip A21 (n2 x m1) * B12 (m1 x p2) -> C21 (n2 x p2)
                for i in range(n1, n):
                    for j in range(m1):
                        for k in range(p1, p):
                            U_c = [0] * (n * m)
                            V_c = [0] * (m * p)
                            W_c = [0] * (p * n)
                            U_c[i * m + j] = 1
                            V_c[j * p + k] = 1
                            W_c[k * n + i] = 1
                            U_list.append(U_c)
                            V_list.append(V_c)
                            W_list.append(W_c)
                
                # Strip A22 (n2 x m2) * B22 (m2 x p2) -> C22 (n2 x p2)
                for i in range(n1, n):
                    for j in range(m1, m):
                        for k in range(p1, p):
                            U_c = [0] * (n * m)
                            V_c = [0] * (m * p)
                            W_c = [0] * (p * n)
                            U_c[i * m + j] = 1
                            V_c[j * p + k] = 1
                            W_c[k * n + i] = 1
                            U_list.append(U_c)
                            V_list.append(V_c)
                            W_list.append(W_c)
                
                rank = len(U_list)
                if rank < best_rank:
                    if _verify_tensor(U_list, V_list, W_list, n, m, p):
                        best = (U_list, V_list, W_list, rank)
                        best_rank = rank
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try Strassen embeddings
    min_S = max(n, m, p)
    S = 1
    while S < min_S:
        S *= 2
    
    for trial_S in [S, S * 2, S * 4, S * 8]:
        if time.time() > deadline - 30:
            break
        
        try:
            result = _try_strassen_embedding(n, m, p, trial_S)
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_result = (U, V, W)
                    best_rank = rank
                    print(f"Found rank {rank} with S={trial_S}")
        except Exception as e:
            print(f"Error with S={trial_S}: {e}")
    
    # Try block partitioning
    if time.time() < deadline - 20:
        try:
            block_result = _try_block_partition(n, m, p)
            if block_result is not None:
                U, V, W, rank = block_result
                if rank < best_rank:
                    best_result = (U, V, W)
                    best_rank = rank
                    print(f"Found rank {rank} with block partition")
        except Exception as e:
            print(f"Error with block partition: {e}")
    
    # If no better result found, use schoolbook
    if best_result is None:
        U, V, W = _make_schoolbook(n, m, p)
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}
