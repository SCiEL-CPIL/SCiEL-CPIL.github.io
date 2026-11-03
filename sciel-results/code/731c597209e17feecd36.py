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
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    U[0][0] = 1.0; U[0][2] = 1.0; U[0][4] = 1.0; U[0][5] = -1.0; U[0][6] = -1.0
    U[1][2] = 1.0; U[1][4] = 1.0; U[1][5] = -1.0; U[1][6] = -1.0
    U[2][0] = 1.0; U[2][1] = 1.0; U[2][5] = 1.0
    U[3][0] = 1.0; U[3][1] = 1.0; U[3][3] = 1.0; U[3][6] = 1.0
    
    V[0][0] = 1.0; V[0][1] = 1.0; V[0][5] = 1.0
    V[1][0] = 1.0; V[1][3] = -1.0; V[1][5] = 1.0
    V[2][2] = 1.0; V[2][4] = 1.0; V[2][6] = 1.0
    V[3][0] = 1.0; V[3][3] = 1.0; V[3][4] = 1.0; V[3][6] = 1.0
    
    W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0; W[0][6] = -1.0
    W[1][2] = 1.0; W[1][4] = 1.0
    W[2][1] = 1.0; W[2][3] = 1.0
    W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][6] = 1.0
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
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
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if abs(total - 1.0) > 1e-6:
                    return False
    return True


def _build_block_decomposition(n, m, p, i_blocks, j_blocks, k_blocks):
    """Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
    
    i_blocks, j_blocks, k_blocks: lists of lists of indices for each dimension.
    Returns (U, V, W, rank) or None if invalid.
    """
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
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in range(len(i_blocks)):
        for jb in range(len(j_blocks)):
            for kb in range(len(k_blocks)):
                i_list = i_blocks[ib]
                j_list = j_blocks[jb]
                k_list = k_blocks[kb]
                ni, nj, nk = len(i_list), len(j_list), len(k_list)
                
                if ni == 2 and nj == 2 and nk == 2:
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        for li in range(2):
                            for lj in range(2):
                                gi = i_list[li]
                                gj = j_list[lj]
                                U[gi * m + gj][col + c] += sU[li * 2 + lj][c]
                        for lj in range(2):
                            for lk in range(2):
                                gj = j_list[lj]
                                gk = k_list[lk]
                                V[gj * p + gk][col + c] += sV[lj * 2 + lk][c]
                        for lk in range(2):
                            for li in range(2):
                                gk = k_list[lk]
                                gi = i_list[li]
                                W[gk * n + gi][col + c] += sW[lk * 2 + li][c]
                    col += 7
                else:
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                gi = i_list[li]
                                gj = j_list[lj]
                                gk = k_list[lk]
                                U[gi * m + gj][col] = 1.0
                                V[gj * p + gk][col] = 1.0
                                W[gk * n + gi][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    return U, V, W, total_rank


def _generate_partition_candidates(n, m, p):
    """Generate candidate block partitions for each dimension.
    
    For each dimension, generate all ways to split into contiguous blocks
    where at least one block has size 2 (to enable Strassen).
    """
    def _partitions(dim_size):
        """Generate all contiguous partitions of range(dim_size)."""
        if dim_size == 1:
            return [[list(range(1))]]
        
        result = []
        # Single block
        result.append([list(range(dim_size))])
        
        # Two or more blocks
        # For each split point, recursively partition
        for split in range(1, dim_size):
            left = list(range(split))
            right = list(range(split, dim_size))
            # Just two blocks
            result.append([left, right])
            # Try further splits of left and right
            for left_parts in _partitions(split):
                if len(left_parts) > 1:
                    for right_parts in _partitions(dim_size - split):
                        result.append(left_parts + right_parts)
        
        # Deduplicate
        seen = set()
        unique = []
        for part in result:
            key = tuple(tuple(block) for block in part)
            if key not in seen:
                seen.add(key)
                unique.append(part)
        return unique
    
    i_parts = _partitions(n)
    j_parts = _partitions(m)
    k_parts = _partitions(p)
    
    return i_parts, j_parts, k_parts


def construct():
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 15.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    
    # Generate partition candidates
    i_parts, j_parts, k_parts = _generate_partition_candidates(n, m, p)
    
    # Try all combinations of partitions
    best_found = False
    for i_block_choice in i_parts:
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        for j_block_choice in j_parts:
            if time.time() - t0 > DEADLINE - MARGIN:
                break
            for k_block_choice in k_parts:
                if time.time() - t0 > DEADLINE - MARGIN:
                    break
                
                # Quick rank estimate
                est_rank = 0
                for ib in range(len(i_block_choice)):
                    for jb in range(len(j_block_choice)):
                        for kb in range(len(k_block_choice)):
                            ni = len(i_block_choice[ib])
                            nj = len(j_block_choice[jb])
                            nk = len(k_block_choice[kb])
                            if ni == 2 and nj == 2 and nk == 2:
                                est_rank += 7
                            else:
                                est_rank += ni * nj * nk
                
                if est_rank >= best_rank:
                    continue
                
                # Build and verify
                U, V, W, rank = _build_block_decomposition(n, m, p, i_block_choice, j_block_choice, k_block_choice)
                
                if _verify(U, V, W, n, m, p):
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        best_found = True
    
    # If no improvement found, use schoolbook
    if not best_found:
        best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
        best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
