"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Build schoolbook factor matrices for (n,m,p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 multiplication, rank 7."""
    # U has 2*2=4 rows, V has 2*2=4 rows, W has 2*2=4 rows
    # Rows of U: index (i,j) -> i*2+j
    # Rows of V: index (j,k) -> j*2+k
    # Rows of W: index (k,i) -> k*2+i
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # U row (i,j):
    # (0,0)=0: M1:+1, M3:+1, M5:+1, M6:-1, M7:-1
    # (0,1)=1: M3:+1, M5:+1, M6:-1, M7:-1
    # (1,0)=2: M1:+1, M2:+1, M6:+1
    # (1,1)=3: M1:+1, M2:+1, M4:+1, M7:+1
    
    U[0][0] = 1.0; U[0][2] = 1.0; U[0][4] = 1.0; U[0][5] = -1.0; U[0][6] = -1.0
    U[1][2] = 1.0; U[1][4] = 1.0; U[1][5] = -1.0; U[1][6] = -1.0
    U[2][0] = 1.0; U[2][1] = 1.0; U[2][5] = 1.0
    U[3][0] = 1.0; U[3][1] = 1.0; U[3][3] = 1.0; U[3][6] = 1.0
    
    # V row (j,k):
    # (0,0)=0: M1:+1, M2:+1, M6:+1
    # (0,1)=1: M1:+1, M4:-1, M6:+1
    # (1,0)=2: M3:+1, M5:+1, M7:+1
    # (1,1)=3: M1:+1, M4:+1, M5:+1, M7:+1
    
    V[0][0] = 1.0; V[0][1] = 1.0; V[0][5] = 1.0
    V[1][0] = 1.0; V[1][3] = -1.0; V[1][5] = 1.0
    V[2][2] = 1.0; V[2][4] = 1.0; V[2][6] = 1.0
    V[3][0] = 1.0; V[3][3] = 1.0; V[3][4] = 1.0; V[3][6] = 1.0
    
    # W row (k,i):
    # (0,0)=0 (C11): M1:+1, M4:+1, M5:-1, M7:-1
    # (0,1)=1 (C12): M3:+1, M5:+1
    # (1,0)=2 (C21): M2:+1, M4:+1
    # (1,1)=3 (C22): M1:+1, M2:-1, M3:+1, M7:+1
    
    W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0; W[0][6] = -1.0
    W[1][2] = 1.0; W[1][4] = 1.0
    W[2][1] = 1.0; W[2][3] = 1.0
    W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][6] = 1.0
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta(i,j,k) for all i,j,k."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) == (0, 0, 0) else 0.0
                # Compute sum over c
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if abs(total - expected) > 1e-6:
                    return False
    return True


def construct():
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 10.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    def _build_block_decomp(i_blocks, j_blocks, k_blocks):
        """Build a block decomposition. Returns (U, V, W, rank) or None if invalid."""
        if time.time() - t0 > DEADLINE - MARGIN:
            return None
        
        # Compute total rank
        total_rank = 0
        for ib in range(len(i_blocks)):
            for jb in range(len(j_blocks)):
                for kb in range(len(k_blocks)):
                    ni = len(i_blocks[ib])
                    nj = len(j_blocks[jb])
                    nk = len(k_blocks[kb])
                    if ni == 2 and nj == 2 and nk == 2:
                        total_rank += 7
                    else:
                        total_rank += ni * nj * nk
        
        # Build U, V, W
        U_mat = [[0.0] * total_rank for _ in range(n * m)]
        V_mat = [[0.0] * total_rank for _ in range(m * p)]
        W_mat = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ib in range(len(i_blocks)):
            for jb in range(len(j_blocks)):
                for kb in range(len(k_blocks)):
                    i_list = list(i_blocks[ib])
                    j_list = list(j_blocks[jb])
                    k_list = list(k_blocks[kb])
                    ni, nj, nk = len(i_list), len(j_list), len(k_list)
                    
                    if ni == 2 and nj == 2 and nk == 2:
                        sU, sV, sW = _strassen_2x2x2()
                        for c in range(7):
                            for li in range(2):
                                for lj in range(2):
                                    gi = i_list[li]
                                    gj = j_list[lj]
                                    U_mat[gi * m + gj][col + c] += sU[li * 2 + lj][c]
                            for lj in range(2):
                                for lk in range(2):
                                    gj = j_list[lj]
                                    gk = k_list[lk]
                                    V_mat[gj * p + gk][col + c] += sV[lj * 2 + lk][c]
                            for lk in range(2):
                                for li in range(2):
                                    gk = k_list[lk]
                                    gi = i_list[li]
                                    W_mat[gk * n + gi][col + c] += sW[lk * 2 + li][c]
                        col += 7
                    else:
                        for li in range(ni):
                            for lj in range(nj):
                                for lk in range(nk):
                                    gi = i_list[li]
                                    gj = j_list[lj]
                                    gk = k_list[lk]
                                    U_mat[gi * m + gj][col] = 1.0
                                    V_mat[gj * p + gk][col] = 1.0
                                    W_mat[gk * n + gi][col] = 1.0
                                    col += 1
        
        if col != total_rank:
            return None
        
        return U_mat, V_mat, W_mat, total_rank
    
    # Strategy: Try multiple block partitionings and keep the best valid one
    # Partition 1: i={0,1}|{2}, j={0,1}|{2,3,4}, k={0,1}|{2,3,4} -> rank 74
    partitions = [
        # (i_blocks, j_blocks, k_blocks)
        ([range(0,2), range(2,3)], [range(0,2), range(2,5)], [range(0,2), range(2,5)]),
        # Try other splits to see if any give lower rank
        ([range(0,2), range(2,3)], [range(0,3), range(3,5)], [range(0,2), range(2,5)]),
        ([range(0,2), range(2,3)], [range(0,2), range(2,5)], [range(0,3), range(3,5)]),
        ([range(0,1), range(1,3)], [range(0,2), range(2,5)], [range(0,2), range(2,5)]),
        ([range(0,2), range(2,3)], [range(0,1), range(1,5)], [range(0,2), range(2,5)]),
        ([range(0,2), range(2,3)], [range(0,2), range(2,5)], [range(0,1), range(1,5)]),
    ]
    
    for i_blks, j_blks, k_blks in partitions:
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        result = _build_block_decomp(i_blks, j_blks, k_blks)
        if result is None:
            continue
        U_cand, V_cand, W_cand, r_cand = result
        if r_cand < best_rank:
            if _verify(U_cand, V_cand, W_cand, n, m, p):
                best_rank = r_cand
                best_U, best_V, best_W = U_cand, V_cand, W_cand
    
    # Also try the original 74-rank construction explicitly as a safety net
    if time.time() - t0 < DEADLINE - MARGIN and best_rank >= 75:
        i_blocks = [range(0, 2), range(2, 3)]
        j_blocks = [range(0, 2), range(2, 5)]
        k_blocks = [range(0, 2), range(2, 5)]
        result = _build_block_decomp(i_blocks, j_blocks, k_blocks)
        if result is not None:
            U_cand, V_cand, W_cand, r_cand = result
            if r_cand < best_rank and _verify(U_cand, V_cand, W_cand, n, m, p):
                best_rank = r_cand
                best_U, best_V, best_W = U_cand, V_cand, W_cand
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
