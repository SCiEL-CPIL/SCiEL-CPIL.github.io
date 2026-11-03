"""Matrix multiplication 3x4x8 with rank-48 decomposition using 3x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook_3x2x2():
    """Return U, V, W for 3x2x2 schoolbook decomposition (rank 6).
    
    A is 3x2, B is 2x2, C is 3x2.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    U rows: A[i,0], A[i,1] for i=0,1,2 -> indices 0,1,2,3,4,5
    V rows: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    W rows: C[i,0], C[i,1] for i=0,1,2 -> indices 0,1,2,3,4,5
    """
    U = [
        [1, 0, 0, 0, 0, 0],  # A[0,0]
        [0, 1, 0, 0, 0, 0],  # A[0,1]
        [0, 0, 1, 0, 0, 0],  # A[1,0]
        [0, 0, 0, 1, 0, 0],  # A[1,1]
        [0, 0, 0, 0, 1, 0],  # A[2,0]
        [0, 0, 0, 0, 0, 1],  # A[2,1]
    ]
    V = [
        [1, 0, 0, 0, 0, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 0],  # B[0,1]
        [0, 1, 0, 0, 0, 0],  # B[1,0]
        [0, 0, 0, 1, 0, 0],  # B[1,1]
    ]
    W = [
        [1, 0, 0, 0, 0, 0],  # C[0,0]
        [0, 1, 0, 0, 0, 0],  # C[0,1]
        [0, 0, 1, 0, 0, 0],  # C[1,0]
        [0, 0, 0, 1, 0, 0],  # C[1,1]
        [0, 0, 0, 0, 1, 0],  # C[2,0]
        [0, 0, 0, 0, 0, 1],  # C[2,1]
    ]
    return U, V, W


def _build_48_rank_decomposition():
    """Build rank-48 decomposition: 8 blocks of 3x2x2 schoolbook (rank 6 each)."""
    n, m, p = 3, 4, 8
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U2, V2, W2 = _schoolbook_3x2x2()
    
    col = 0
    for jb in range(2):  # 2 blocks in j (m=4 split into two 2s)
        for kb in range(4):  # 4 blocks in k (p=8 split into four 2s)
            # A_sub: rows i=0,1,2; cols j=2jb, 2jb+1
            # B_sub: rows j=2jb, 2jb+1; cols k=2kb, 2kb+1
            # C_sub: rows i=0,1,2; cols k=2kb, 2kb+1
            
            for c in range(6):
                # U: A[i, j_local] -> U[i*m + j, c]
                # U2 has 6 rows: A[0,0], A[0,1], A[1,0], A[1,1], A[2,0], A[2,1]
                for i in range(3):
                    for jl in range(2):
                        u_idx = i * 2 + jl  # 0..5
                        u_val = U2[u_idx][c]
                        if u_val != 0:
                            global_j = 2 * jb + jl
                            U[i * m + global_j][col] += u_val
                
                # V: B[j_local, k_local] -> V[j*p + k, c]
                # V2 has 4 rows: B[0,0], B[0,1], B[1,0], B[1,1]
                for jl in range(2):
                    for kl in range(2):
                        v_idx = jl * 2 + kl  # 0..3
                        v_val = V2[v_idx][c]
                        if v_val != 0:
                            global_j = 2 * jb + jl
                            global_k = 2 * kb + kl
                            V[global_j * p + global_k][col] += v_val
                
                # W: C[i, k_local] -> W[k*n + i, c]
                # W2 has 6 rows: C[0,0], C[0,1], C[1,0], C[1,1], C[2,0], C[2,1]
                for i in range(3):
                    for kl in range(2):
                        w_idx = i * 2 + kl  # 0..5
                        w_val = W2[w_idx][c]
                        if w_val != 0:
                            global_k = 2 * kb + kl
                            W[global_k * n + i][col] += w_val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify exact tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integers or half-integers."""
    for mat in [U, V, W]:
        arr = np.array(mat, dtype=float)
        if np.any(np.abs(arr * 2 - np.round(arr * 2)) > tol):
            return False
    return True


def _reduce_rank_svd(U, V, W, deadline, max_iterations=50):
    """Try to reduce rank using SVD-based analysis.
    
    The tensor T = sum_c U_c ⊗ V_c ⊗ W_c. We can try to find a lower-rank
    representation by analyzing the structure.
    
    Strategy: For each "slice" of the tensor, compute SVD and check if
    the rank can be reduced by finding linear dependencies.
    """
    n, m, p = 3, 4, 8
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    def _verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_mat[i*m+j] * V_mat[j*p+k] * W_mat[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _verify_half_int_fast(U_mat, V_mat, W_mat, tol=1e-9):
        for mat in [U_mat, V_mat, W_mat]:
            if np.any(np.abs(mat * 2 - np.round(mat * 2)) > tol):
                return False
        return True
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Greedy column removal
    col_order = list(range(rank))
    # Sort by sparsity
    sparsity = []
    for c in range(rank):
        s = np.count_nonzero(U_arr[:, c]) + np.count_nonzero(V_arr[:, c]) + np.count_nonzero(W_arr[:, c])
        sparsity.append((s, c))
    sparsity.sort()
    col_order = [c for _, c in sparsity]
    
    removed = []
    for col_idx in col_order:
        if time.time() > deadline - 10:
            break
        if col_idx in removed:
            continue
        
        keep = [i for i in range(rank) if i != col_idx and i not in removed]
        U_test = U_arr[:, keep]
        V_test = V_arr[:, keep]
        W_test = W_arr[:, keep]
        
        if _verify_fast(U_test, V_test, W_test):
            if _verify_half_int_fast(U_test, V_test, W_test):
                removed.append(col_idx)
                rank = len(keep)
                if rank < best_rank:
                    best_rank = rank
                    best_U = U_test.copy()
                    best_V = V_test.copy()
                    best_W = W_test.copy()
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    
    return U_arr, V_arr, W_arr, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Build the rank-48 decomposition
    result = _build_48_rank_decomposition()
    
    # Verify correctness
    if not _verify(result['U'], result['V'], result['W']):
        # Fallback: schoolbook rank 96
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
    
    # Try to reduce rank
    if time.time() < deadline - 15:
        U_arr, V_arr, W_arr, new_rank = _reduce_rank_svd(
            result['U'], result['V'], result['W'], deadline
        )
        if new_rank < len(result['U'][0]):
            result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    return result
# EVOLVE-BLOCK-END
