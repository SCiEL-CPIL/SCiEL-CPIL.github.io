import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2_factors():
    """Return U, V, W factors for 2x2x2 Strassen decomposition."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=np.int64)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    return U, V, W

def _schoolbook_2x1x2_factors():
    """Return U, V, W factors for 2x1x2 schoolbook decomposition."""
    # A (2x1), B (1x2), C (2x2)
    # C[0,0] = A[0,0]*B[0,0]
    # C[0,1] = A[0,0]*B[0,1]
    # C[1,0] = A[1,0]*B[0,0]
    # C[1,1] = A[1,0]*B[0,1]
    U = np.array([
        [1, 0, 0, 0], # A[0,0]
        [0, 0, 1, 0]  # A[1,0]
    ], dtype=np.int64)
    V = np.array([
        [1, 1, 0, 0]  # B[0,0], B[0,1]
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 0], # C[0,0] (k=0,i=0)
        [0, 1, 0, 0], # C[0,1] (k=1,i=0) -> Wait, W index is k*n+i. n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # C[0,0] is i=0,k=0 -> W[0]
        # C[0,1] is i=0,k=1 -> W[2]
        # C[1,0] is i=1,k=0 -> W[1]
        # C[1,1] is i=1,k=1 -> W[3]
        [0, 0, 1, 0], # C[1,0] -> W[1]? No.
        # Let's re-verify W indices.
        # W[k*n+i].
        # Term 1: A[0,0]*B[0,0] -> C[0,0]. i=0, k=0. W[0*2+0]=W[0].
        # Term 2: A[0,0]*B[0,1] -> C[0,1]. i=0, k=1. W[1*2+0]=W[2].
        # Term 3: A[1,0]*B[0,0] -> C[1,0]. i=1, k=0. W[0*2+1]=W[1].
        # Term 4: A[1,0]*B[0,1] -> C[1,1]. i=1, k=1. W[1*2+1]=W[3].
        
        # So:
        # Col 0: U[0]=1, V[0]=1, W[0]=1 (C[0,0])
        # Col 1: U[0]=0, V[0]=1, W[2]=1 (C[0,1]) -> V[0] is B[0,1]? No, V is 1x4.
        # V row is j*p+k. Here m=1, so j=0. p=2.
        # V[0,0] corresponds to B[0,0].
        # V[0,1] corresponds to B[0,1].
        # So:
        # Col 0: U[0,0]=1, V[0,0]=1, W[0,0]=1.
        # Col 1: U[0,1]=1, V[0,1]=1, W[0,2]=1.
        # Col 2: U[1,2]=1, V[0,2]=1, W[0,1]=1.
        # Col 3: U[1,3]=1, V[0,3]=1, W[0,3]=1.
        
    ], dtype=np.int64)
    
    U = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=np.int64)
    V = np.array([
        [1, 1, 1, 1]
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 0], # C[0,0]
        [0, 0, 1, 0], # C[1,0]
        [0, 1, 0, 0], # C[0,1]
        [0, 0, 0, 1]  # C[1,1]
    ], dtype=np.int64)
    return U, V, W

def _add_block(n, m, p, i_off, j_off, k_off, U, V, W, rank_offset):
    """Recursively add decomposition for block n x m x p."""
    # Base case 1: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        U_sub, V_sub, W_sub = _strassen_2x2_factors()
        for c in range(U_sub.shape[1]):
            for i in range(2):
                for j in range(2):
                    U[i_off + i, j_off + j, rank_offset + c] += U_sub[i*2+j, c]
            for j in range(2):
                for k in range(2):
                    V[j_off + j, k_off + k, rank_offset + c] += V_sub[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    W[k_off + k, i_off + i, rank_offset + c] += W_sub[k*2+i, c]
        return U_sub.shape[1]

    # Base case 2: 2x1x2 Schoolbook
    if n == 2 and m == 1 and p == 2:
        U_sub, V_sub, W_sub = _schoolbook_2x1x2_factors()
        for c in range(U_sub.shape[1]):
            for i in range(2):
                U[i_off + i, j_off, rank_offset + c] += U_sub[i, c]
            V[j_off, k_off, rank_offset + c] += V_sub[0, c]
            for k in range(2):
                for i in range(2):
                    W[k_off + k, i_off + i, rank_offset + c] += W_sub[k*2+i, c]
        return U_sub.shape[1]
    
    # Recursive cases
    total_rank = 0
    
    # Heuristic splits
    # 4x2x6 -> 2x(2x2x6)
    if n == 4 and m == 2 and p == 6:
        total_rank += _add_block(2, 2, 6, i_off, j_off, k_off, U, V, W, rank_offset + total_rank)
        total_rank += _add_block(2, 2, 6, i_off + 2, j_off, k_off, U, V, W, rank_offset + total_rank)
        return total_rank

    # 2x2x6 -> 3x(2x2x2)
    if n == 2 and m == 2 and p == 6:
        for kk in range(3):
            total_rank += _add_block(2, 2, 2, i_off, j_off, k_off + kk*2, U, V, W, rank_offset + total_rank)
        return total_rank

    # 4x3x6 -> 2x(2x3x6)
    if n == 4 and m == 3 and p == 6:
        total_rank += _add_block(2, 3, 6, i_off, j_off, k_off, U, V, W, rank_offset + total_rank)
        total_rank += _add_block(2, 3, 6, i_off + 2, j_off, k_off, U, V, W, rank_offset + total_rank)
        return total_rank

    # 2x3x6 -> 3x(2x3x2)
    if n == 2 and m == 3 and p == 6:
        for kk in range(3):
            total_rank += _add_block(2, 3, 2, i_off, j_off, k_off + kk*2, U, V, W, rank_offset + total_rank)
        return total_rank

    # 2x3x2 -> (2x2x2) + (2x1x2)
    if n == 2 and m == 3 and p == 2:
        total_rank += _add_block(2, 2, 2, i_off, j_off, k_off, U, V, W, rank_offset + total_rank)
        total_rank += _add_block(2, 1, 2, i_off, j_off + 2, k_off, U, V, W, rank_offset + total_rank)
        return total_rank

    # Fallback: Schoolbook
    rank = n * m * p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i_off + i, j_off + j, rank_offset + total_rank] = 1
                V[j_off + j, k_off + k, rank_offset + total_rank] = 1
                W[k_off + k, i_off + i, rank_offset + total_rank] = 1
                total_rank += 1
    return total_rank

def _verify_tensor(U, V, W, n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U = np.zeros((n * m, 200), dtype=np.int64) # Max possible rank
    V = np.zeros((m * p, 200), dtype=np.int64)
    W = np.zeros((p * n, 200), dtype=np.int64)
    
    total_rank = _add_block(n, m, p, 0, 0, 0, U, V, W, 0)
    
    U = U[:, :total_rank].tolist()
    V = V[:, :total_rank].tolist()
    W = W[:, :total_rank].tolist()
    
    if not _verify_tensor(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _make_schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}

def _make_schoolbook(n, m, p):
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
