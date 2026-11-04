"""Tensor decomposition for 2x3 * 3x10 matrix multiplication using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """
    Return (U, V, W) for 2x2 * 2x2 multiplication using Strassen's rank-7 algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # p1 = (a+d)(e+h)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    
    # p2 = (c+d)e
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    
    # p3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    
    # p4 = d(g-e)
    U[3][3] = 1
    V[0][3] = -1
    V[2][3] = 1
    
    # p5 = (a+b)h
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    
    # p6 = (c-a)(e+f)
    U[0][5] = -1
    U[2][5] = 1
    V[0][5] = 1
    V[1][5] = 1
    
    # p7 = (b-d)(g+h)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    
    # W: C[ki] where k=0,1 and i=0,1
    # C[0,0] = p1 + p4 - p5 + p7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # C[0,1] = p3 + p5
    W[1][2] = 1
    W[1][4] = 1
    
    # C[1,0] = p2 + p4
    W[2][1] = 1
    W[2][3] = 1
    
    # C[1,1] = p1 - p2 + p3 + p6
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    return U, V, W, rank


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r u_{ij}^{(r)} v_{jk}^{(r)} = w_{ki}^{(r)} for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U_arr[i * m + j] @ V_arr[j * p + k]
                rhs = W_arr[k * n + i]
                if not np.allclose(lhs, rhs, atol=1e-9):
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Split p=10 into five blocks of size 2.
    # For each 2x3 * 3x2 block, use Strassen (rank 7) + schoolbook (rank 4) = rank 11.
    # Total rank: 5 * 11 = 55.
    
    block_rank = 11
    num_blocks = 5  # p / 2 = 10 / 2
    total_rank = block_rank * num_blocks  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    for block in range(num_blocks):
        block_base = block * block_rank
        
        # Get the 2x3 * 3x2 decomposition
        sU, sV, sW, sr = _build_strassen_222()
        
        # Place Strassen part: A_rows(0,1) x B_cols(block*2, block*2+1)
        for i in range(4):
            for r in range(sr):
                U[i][block_base + r] += sU[i][r]
        for j in range(2):
            for k_local in range(2):
                k_global = block * 2 + k_local
                v_idx = j * p + k_global
                bV_idx = j * 2 + k_local
                for r in range(sr):
                    V[v_idx][block_base + r] += sV[bV_idx][r]
        for i in range(2):
            for k_local in range(2):
                k_global = block * 2 + k_local
                w_idx = k_global * n + i
                bW_idx = k_local * n + i
                for r in range(sr):
                    W[w_idx][block_base + r] += sW[bW_idx][r]
        
        # Schoolbook for A_row(2) x B_cols(block*2, block*2+1)
        # C[i, block*2] += a_{i,2} * b_{2, block*2}
        # C[i, block*2+1] += a_{i,2} * b_{2, block*2+1}
        base = block_base + 7
        for i in range(2):
            a_idx = i * 3 + 2  # a_{i,2}
            for k_local in range(2):
                k_global = block * 2 + k_local
                b_idx = 2 * p + k_global  # b_{2, k}
                c_idx = k_global * n + i  # C[i, k]
                U[a_idx][base + k_local] = 1
                V[b_idx][base + k_local] = 1
                W[c_idx][base + k_local] = 1
    
    # Quick self-check
    if not _validate(n, m, p, U, V, W):
        # Fall back to trivial if validation fails
        U, V, W, _ = _build_trivial(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
