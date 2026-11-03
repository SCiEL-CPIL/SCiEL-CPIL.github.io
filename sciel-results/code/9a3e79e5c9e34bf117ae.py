import time
import sys

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _apply_factors(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    nmp = n * m * p
    result = [0.0] * nmp
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[j][r] for j in range(m * p)]
        w_col = [W[k][r] for k in range(p * n)]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    result[i * m * p + j * p + k] += u_val * v_val * w_val
    return result

def _verify(U, V, W, n, m, p, T):
    result = _apply_factors(U, V, W, n, m, p)
    for i in range(len(T)):
        if abs(result[i] - T[i]) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W, rank

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _block_decomp(n, m, p, T, j_groups, k_groups):
    """
    j_groups: list of lists of j indices
    k_groups: list of lists of k indices
    Each (j_group, k_group) pair forms a block.
    """
    max_rank = 200
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    sU, sV, sW, srank = _strassen_2x2x2()
    
    for j_group in j_groups:
        for k_group in k_groups:
            j_size = len(j_group)
            k_size = len(k_group)
            
            if j_size == 2 and k_size == 2:
                for r in range(srank):
                    for i in range(n):
                        for jj, j_idx in enumerate(j_group):
                            U[i * m + j_idx][col] += sU[i * 2 + jj][r]
                    for jj, j_idx in enumerate(j_group):
                        for kk, k_idx in enumerate(k_group):
                            V[j_idx * p + k_idx][col] += sV[jj * 2 + kk][r]
                    for kk, k_idx in enumerate(k_group):
                        for i in range(n):
                            W[k_idx * n + i][col] += sW[kk * 2 + i][r]
                    col += 1
            elif j_size == 2 and k_size == 1:
                for i in range(n):
                    for jj, j_idx in enumerate(j_group):
                        U[i * m + j_idx][col] = 1
                        V[j_idx * p + k_group[0]][col] = 1
                        W[k_group[0] * n + i][col] = 1
                        col += 1
            elif j_size == 1 and k_size == 2:
                for i in range(n):
                    for kk, k_idx in enumerate(k_group):
                        U[i * m + j_group[0]][col] = 1
                        V[j_group[0] * p + k_idx][col] = 1
                        W[k_idx * n + i][col] = 1
                        col += 1
            elif j_size == 1 and k_size == 1:
                for i in range(n):
                    U[i * m + j_group[0]][col] = 1
                    V[j_group[0] * p + k_group[0]][col] = 1
                    W[k_group[0] * n + i][col] = 1
                    col += 1
            else:
                for i in range(n):
                    for jj, j_idx in enumerate(j_group):
                        for kk, k_idx in enumerate(k_group):
                            U[i * m + j_idx][col] = 1
                            V[j_idx * p + k_idx][col] = 1
                            W[k_idx * n + i][col] = 1
                            col += 1
    
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    return U, V, W, col

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    T, n, m, p = _build_tensor()
    
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    best = (U_school, V_school, W_school, rank_school)
    
    # Try different j splits
    j_splits = [
        [[0, 1, 2]],
        [[0, 1], [2]],
        [[0], [1, 2]],
        [[0], [1], [2]],
    ]
    
    # Try different k splits (must cover all 7 indices)
    k_splits = [
        [[0,1], [2,3], [4,5], [6]],
        [[0,1], [2,3], [4,5,6]],
        [[0,1,2], [3,4], [5,6]],
        [[0,1], [2,3,4], [5,6]],
        [[0,1,2,3], [4,5], [6]],
        [[0,1], [2,3], [4,5], [6]],
        [[0,1,2], [3,4,5], [6]],
        [[0,1], [2,3,4,5], [6]],
        [[0,1,2,3,4], [5,6]],
        [[0,1], [2,3,4,5,6]],
    ]
    
    for j_split in j_splits:
        for k_split in k_splits:
            if time.time() > deadline - 5:
                break
            
            result = _block_decomp(n, m, p, T, j_split, k_split)
            if result is not None:
                U, V, W, rank = result
                if rank < best[3]:
                    best = (U, V, W, rank)
    
    U, V, W, rank = best
    return {"U": U, "V": V, "W": W}
