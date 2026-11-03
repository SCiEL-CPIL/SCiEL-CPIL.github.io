import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    T_correct = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    
    # Efficient check: for each r, add the outer product contribution
    for r in range(U.shape[1]):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # Contribution: u_r[i,j] * v_r[j,k] * w_r[k,i]
        T_recon += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
    
    diff = np.max(np.abs(T_recon - T_correct))
    return diff < tol

def _build_strassen_2x2_block():
    """
    Build Strassen decomposition for a 2x2x2 block.
    Returns U (4, 7), V (4, 7), W (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: a11*(b22-b12)
    U[0, 0] = 1  # a11
    V[2, 0] = 1  # b22
    V[1, 0] = -1 # b12
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2: (a11+a12)*b22
    U[0, 1] = 1  # a11
    U[1, 1] = 1  # a12
    V[2, 1] = 1  # b22
    W[1, 1] = 1  # C12
    W[3, 1] = 1  # C22
    
    # M3: (a21+a22)*b11
    U[2, 2] = 1  # a21
    U[3, 2] = 1  # a22
    V[0, 2] = 1  # b11
    W[2, 2] = 1  # C21
    
    # M4: a22*(b12-b11)
    U[3, 3] = 1  # a22
    V[1, 3] = 1  # b12
    V[0, 3] = -1 # b11
    W[0, 3] = 1  # C11
    W[3, 3] = 1  # C22
    
    # M5: (a11+a21)*(b11+b12)
    U[0, 4] = 1  # a11
    U[2, 4] = 1  # a21
    V[0, 4] = 1  # b11
    V[1, 4] = 1  # b12
    W[0, 4] = 1  # C11
    W[2, 4] = 1  # C21
    
    # M6: (a12-a21)*(b21+b22)
    U[1, 5] = 1  # a12
    U[2, 5] = -1 # a21
    V[3, 5] = 1  # b21
    V[2, 5] = 1  # b22
    W[1, 5] = 1  # C12
    
    # M7: (a11-a12)*(b11+b21)
    U[0, 6] = 1  # a11
    U[1, 6] = -1 # a12
    V[0, 6] = 1  # b11
    V[3, 6] = 1  # b21
    W[0, 6] = 1  # C11
    W[2, 6] = 1  # C21
    
    return U, V, W

def _build_schoolbook_block(bn, bm, bp):
    """Build schoolbook decomposition for a bn x bm x bp block."""
    rank = bn * bm * bp
    U = np.zeros((bn * bm, rank))
    V = np.zeros((bm * bp, rank))
    W = np.zeros((bp * bn, rank))
    col = 0
    for i in range(bn):
        for j in range(bm):
            for k in range(bp):
                U[i * bm + j, col] = 1
                V[j * bp + k, col] = 1
                W[k * bn + i, col] = 1
                col += 1
    return U, V, W

def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Build a decomposition based on a specific partitioning of dimensions.
    Uses Strassen for 2x2x2 blocks, Schoolbook for others.
    """
    total_rank = 0
    # Calculate total rank to allocate memory
    for bn, bm, bp in zip(n_parts, m_parts, p_parts):
        if bn == 2 and bm == 2 and bp == 2:
            total_rank += 7
        else:
            total_rank += bn * bm * bp
            
    if total_rank == 0:
        return np.zeros((n*m, 0)), np.zeros((m*p, 0)), np.zeros((p*n, 0))

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    n_offset = 0
    for bn in n_parts:
        m_offset = 0
        for bm in m_parts:
            p_offset = 0
            for bp in p_parts:
                if bn == 2 and bm == 2 and bp == 2:
                    U_b, V_b, W_b = _build_strassen_2x2_block()
                else:
                    U_b, V_b, W_b = _build_schoolbook_block(bn, bm, bp)
                
                block_rank = U_b.shape[1]
                
                # Map local indices to global
                for r in range(block_rank):
                    for li in range(bn):
                        for lj in range(bm):
                            local_u_idx = li * bm + lj
                            global_u_idx = (n_offset + li) * m + (m_offset + lj)
                            U[global_u_idx, col_offset + r] = U_b[local_u_idx, r]
                    
                    for lj in range(bm):
                        for lk in range(bp):
                            local_v_idx = lj * bp + lk
                            global_v_idx = (m_offset + lj) * p + (p_offset + lk)
                            V[global_v_idx, col_offset + r] = V_b[local_v_idx, r]
                    
                    for lk in range(bp):
                        for li in range(bn):
                            local_w_idx = lk * bn + li
                            global_w_idx = (p_offset + lk) * n + (n_offset + li)
                            W[global_w_idx, col_offset + r] = W_b[local_w_idx, r]
                            
                col_offset += block_rank
                p_offset += bp
            m_offset += bm
        n_offset += bn
        
    return U, V, W

def _generate_partitions(total, max_size=3):
    """Generate all partitions of 'total' into parts of size <= max_size."""
    # Since dimensions are small (3, 4, 7), we can just generate all compositions
    # and filter for parts that allow Strassen (2) or Schoolbook (any)
    # Actually, Strassen only helps if we have 2x2x2.
    # So we only care about partitions that include 2s if we want to use Strassen.
    # But Schoolbook is rank n*m*p. Strassen is rank 7 vs 8.
    # So any 2x2x2 block should use Strassen.
    
    parts = []
    def backtrack(remaining, current):
        if remaining == 0:
            parts.append(tuple(current))
            return
        for i in range(1, remaining + 1):
            backtrack(remaining - i, current + [i])
            
    backtrack(total, [])
    return parts

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Build Schoolbook baseline
    U_sb, V_sb, W_sb = _build_schoolbook_block(n, m, p)
    best_rank = n * m * p
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # 2. Exhaustive search over partitions
    n_parts_list = _generate_partitions(n)
    m_parts_list = _generate_partitions(m)
    p_parts_list = _generate_partitions(p)
    
    # We can prune partitions that are obviously worse?
    # Rank of a partition is sum of ranks of blocks.
    # Rank of block (bn, bm, bp) is 7 if (2,2,2) else bn*bm*bp.
    
    best_partition_rank = best_rank
    
    for n_parts in n_parts_list:
        for m_parts in m_parts_list:
            for p_parts in p_parts_list:
                # Calculate theoretical rank
                rank = 0
                for bn, bm, bp in zip(n_parts, m_parts, p_parts):
                    if bn == 2 and bm == 2 and bp == 2:
                        rank += 7
                    else:
                        rank += bn * bm * bp
                
                if rank >= best_partition_rank:
                    continue
                    
                # Build and verify
                try:
                    U_cand, V_cand, W_cand = _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts)
                    if _verify(U_cand, V_cand, W_cand, n, m, p):
                        best_partition_rank = rank
                        best_U = U_cand
                        best_V = V_cand
                        best_W = W_cand
                except Exception:
                    pass
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
