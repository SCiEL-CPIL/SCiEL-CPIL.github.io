"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    The tensor has dimensions (n*m, m*p, p*n) indexed by (i*m+j, j*p+k, k*n+i)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    The decomposition satisfies: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # c=0: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # c=2: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # c=3: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # c=5: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # c=6: M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_subblock(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset, n, m, p, U, V, W, col):
    """Add schoolbook decomposition for a sub-block. Returns new col."""
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                gi = i_offset + i
                gj = j_offset + j
                gk = k_offset + k
                U[gi * m + gj, col] = 1
                V[gj * p + gk, col] = 1
                W[gk * n + gi, col] = 1
                col += 1
    return col

def _strassen_subblock(i_offset, j_offset, k_offset, n, m, p, U, V, W, col):
    """Add Strassen decomposition for a 2x2x2 sub-block. Returns new col."""
    U_s, V_s, W_s = _strassen_2x2x2()
    for ci in range(2):
        for cj in range(2):
            for ck in range(2):
                gi = i_offset + ci
                gj = j_offset + cj
                gk = k_offset + ck
                for c in range(7):
                    U[gi * m + gj, col + c] = U_s[ci * 2 + cj, c]
                    V[gj * p + gk, col + c] = V_s[cj * 2 + ck, c]
                    W[gk * n + gi, col + c] = W_s[ck * 2 + ci, c]
    return col + 7

def _build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise."""
    total_rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    total_rank += 7
                else:
                    total_rank += sn * sm * sp
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    col = _strassen_subblock(nb[0], mb[0], pb[0], n, m, p, U, V, W, col)
                else:
                    col = _schoolbook_subblock(sn, sm, sp, nb[0], mb[0], pb[0], n, m, p, U, V, W, col)
    
    return U, V, W, total_rank

def _generate_block_partitions(n, m, p):
    """Generate all possible block partitions for dimensions n, m, p."""
    def get_partitions(d):
        """Get all ways to partition d into blocks of size 1 or 2."""
        if d == 0:
            return [[]]
        if d == 1:
            return [[(0,)]]
        if d == 2:
            return [[(0, 1)], [(0,), (1,)]]
        
        partitions = []
        # First block of size 1
        for rest in get_partitions(d - 1):
            partitions.append([(0,)] + [(x[0]+1, x[1]+1) if len(x)==2 else (x[0]+1,) for x in rest])
        # First block of size 2
        if d >= 2:
            for rest in get_partitions(d - 2):
                partitions.append([(0, 1)] + [(x[0]+2, x[1]+2) if len(x)==2 else (x[0]+2,) for x in rest])
        return partitions
    
    n_parts = get_partitions(n)
    m_parts = get_partitions(m)
    p_parts = get_partitions(p)
    
    all_combinations = []
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                all_combinations.append((np_, mp_, pp_))
    
    return all_combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try all block partition combinations
    partitions = _generate_block_partitions(n, m, p)
    
    for n_blocks, m_blocks, p_blocks in partitions:
        # Check time
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        U, V, W, total_rank = _build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks)
        
        if total_rank < best_rank:
            # Verify
            if _check_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = total_rank
    
    # Return best found
    if _check_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
