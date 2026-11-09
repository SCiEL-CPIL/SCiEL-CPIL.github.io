"""Rank-minimal tensor decomposition for 3x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_222():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where:
      U: (4, 7) - coefficients for A entries (A[0,0], A[0,1], A[1,0], A[1,1])
      V: (4, 7) - coefficients for B entries (B[0,0], B[0,1], B[1,0], B[1,1])
      W: (4, 7) - coefficients for C entries, W[k*2+i, c] for C[i,k]
    """
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # A indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B indices: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C indices: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    # W[k*2+i, c]: C[i,k] -> index k*2+i
    # C[0,0]: k=0,i=0 -> W[0]
    # C[0,1]: k=1,i=0 -> W[1]
    # C[1,0]: k=0,i=1 -> W[2]
    # C[1,1]: k=1,i=1 -> W[3]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    # C00 += M1, C11 += M1
    W[0][0] = 1.0  # C[0,0]
    W[3][0] = 1.0  # C[1,1]
    
    # M2 = (A10+A11)B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    # C10 += M2
    W[2][1] = 1.0  # C[1,0]
    
    # M3 = A00(B01-B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    # C01 += M3, C11 += M3
    W[1][2] = 1.0  # C[0,1]
    W[3][2] = 1.0  # C[1,1]
    
    # M4 = A11(B10-B00)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    # C00 += M4, C10 += M4
    W[0][3] = 1.0  # C[0,0]
    W[2][3] = 1.0  # C[1,0]
    
    # M5 = (A00+A01)B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    # C00 -= M5, C01 += M5
    W[0][4] = -1.0  # C[0,0]
    W[1][4] = 1.0  # C[0,1]
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    # C11 += M6
    W[3][5] = 1.0  # C[1,1]
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    # C00 += M7
    W[0][6] = 1.0  # C[0,0]
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """Schoolbook decomposition for an n_block x m_block x p_block subproblem."""
    rank = n_block * m_block * p_block
    U = [[0.0]*rank for _ in range(n_block * m_block)]
    V = [[0.0]*rank for _ in range(m_block * p_block)]
    W = [[0.0]*rank for _ in range(p_block * n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i*m_block+j][col] = 1.0
                V[j*p_block+k][col] = 1.0
                W[k*n_block+i][col] = 1.0
                col += 1
    return U, V, W, rank

def _combine_blocks(block_decomps, n, m, p, n_splits, m_splits, p_splits):
    """
    Combine block decompositions into a full 3x4x7 decomposition.
    
    n_splits: list of row sizes for A/C, e.g., [2, 1]
    m_splits: list of col sizes for A / row sizes for B, e.g., [2, 2]
    p_splits: list of col sizes for B/C, e.g., [2, 2, 2, 1]
    
    block_decomps[i][j][k] = (U_ij, V_ij, W_ij) for block (i,j,k)
    """
    total_rank = sum(
        len(block_decomps[i][j][k][0][0])
        for i in range(len(n_splits))
        for j in range(len(m_splits))
        for k in range(len(p_splits))
    )
    
    # Full U: (n*m, total_rank)
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                U_blk, V_blk, W_blk = block_decomps[i][j][k]
                blk_rank = len(U_blk[0])
                
                # Map block indices to full indices
                # A rows: sum of n_splits[0..i-1] + local_i
                # A cols: sum of m_splits[0..j-1] + local_j
                # B rows: sum of m_splits[0..j-1] + local_j
                # B cols: sum of p_splits[0..k-1] + local_k
                # C rows: sum of n_splits[0..i-1] + local_i
                # C cols: sum of p_splits[0..k-1] + local_k
                
                n_offset = sum(n_splits[:i])
                m_offset = sum(m_splits[:j])
                p_offset = sum(p_splits[:k])
                
                n_b = n_splits[i]
                m_b = m_splits[j]
                p_b = p_splits[k]
                
                for r in range(n_b * m_b):
                    local_i = r // m_b
                    local_j = r % m_b
                    full_i = n_offset + local_i
                    full_j = m_offset + local_j
                    full_row = full_i * m + full_j
                    for c in range(blk_rank):
                        U[full_row][rank_offset + c] = U_blk[r][c]
                
                for r in range(m_b * p_b):
                    local_j = r // p_b
                    local_k = r % p_b
                    full_j = m_offset + local_j
                    full_k = p_offset + local_k
                    full_row = full_j * p + full_k
                    for c in range(blk_rank):
                        V[full_row][rank_offset + c] = V_blk[r][c]
                
                for r in range(p_b * n_b):
                    local_k = r // n_b
                    local_i = r % n_b
                    full_k = p_offset + local_k
                    full_i = n_offset + local_i
                    full_row = full_k * n + full_i
                    for c in range(blk_rank):
                        W[full_row][rank_offset + c] = W_blk[r][c]
                
                rank_offset += blk_rank
    
    return U, V, W, total_rank

def _construct_block_decomposition():
    """
    Construct block decomposition for 3x4x7.
    Split: n=[2,1], m=[2,2], p=[2,2,2,1]
    
    For each (i,j,k) block of size (n_i, m_j, p_k):
    - If n_i=2, m_j=2, p_k=2: use Strassen (rank 7)
    - Otherwise: use schoolbook
    """
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 2, 1]
    
    block_decomps = [[[None]*len(p_splits) for _ in range(len(m_splits))] for _ in range(len(n_splits))]
    
    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                n_b = n_splits[i]
                m_b = m_splits[j]
                p_b = p_splits[k]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Use Strassen
                    U_blk, V_blk, W_blk = _strassen_222()
                else:
                    # Use schoolbook
                    U_blk, V_blk, W_blk, _ = _schoolbook_block(n_b, m_b, p_b)
                
                block_decomps[i][j][k] = (U_blk, V_blk, W_blk)
    
    n, m, p = 3, 4, 7
    U, V, W, total_rank = _combine_blocks(block_decomps, n, m, p, n_splits, m_splits, p_splits)
    
    return U, V, W, total_rank

def _local_search(U, V, W, n, m, p, max_time=90):
    """Try to reduce rank by finding and removing linearly dependent columns."""
    start_time = time.time()
    best_rank = len(U[0])
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = best_rank
    
    for _ in range(200):
        if time.time() - start_time > max_time:
            break
        
        r_cur = len(current_U[0])
        if r_cur <= 1:
            break
        
        found_removal = False
        for col_to_remove in range(r_cur):
            if time.time() - start_time > max_time:
                break
            
            other_cols = [c for c in range(r_cur) if c != col_to_remove]
            
            # Build system: find alpha such that col_to_remove = sum alpha_c * col_c
            n_eqs = len(current_U) + len(current_V) + len(current_W)
            n_unknowns = len(other_cols)
            
            A_mat = np.zeros((n_eqs, n_unknowns))
            b_vec = np.zeros(n_eqs)
            
            idx = 0
            for i in range(len(current_U)):
                for ci, c in enumerate(other_cols):
                    A_mat[idx, ci] = current_U[i][c]
                b_vec[idx] = current_U[i][col_to_remove]
                idx += 1
            for j in range(len(current_V)):
                for ci, c in enumerate(other_cols):
                    A_mat[idx, ci] = current_V[j][c]
                b_vec[idx] = current_V[j][col_to_remove]
                idx += 1
            for k in range(len(current_W)):
                for ci, c in enumerate(other_cols):
                    A_mat[idx, ci] = current_W[k][c]
                b_vec[idx] = current_W[k][col_to_remove]
                idx += 1
            
            try:
                alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            residual = np.linalg.norm(A_mat @ alpha - b_vec)
            if residual < 1e-8:
                # Remove this column
                new_U = []
                for i in range(len(current_U)):
                    new_row = [current_U[i][c] for c in other_cols]
                    new_U.append(new_row)
                
                new_V = []
                for j in range(len(current_V)):
                    new_row = [current_V[j][c] for c in other_cols]
                    new_V.append(new_row)
                
                new_W = []
                for k in range(len(current_W)):
                    new_row = [current_W[k][c] for c in other_cols]
                    new_W.append(new_row)
                
                if _verify(new_U, new_V, new_W, n, m, p):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = r_cur - 1
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U = [row[:] for row in current_U]
                        best_V = [row[:] for row in current_V]
                        best_W = [row[:] for row in current_W]
                    found_removal = True
                    break
        
        if not found_removal:
            # No more columns can be removed; try a different approach
            # Randomly perturb and see if we can find a valid lower-rank solution
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = 110  # Safety margin before 120s limit
    
    # Build block decomposition
    U_block, V_block, W_block, rank_block = _construct_block_decomposition()
    
    # Verify block decomposition
    if not _verify(U_block, V_block, W_block, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try local search to reduce rank further
    try:
        U_best, V_best, W_best, rank_best = _local_search(
            U_block, V_block, W_block, n, m, p, 
            max_time=min(90, deadline - (time.time() - 0))
        )
    except Exception:
        U_best, V_best, W_best = U_block, V_block, W_block
    
    # Final verification
    if not _verify(U_best, V_best, W_best, n, m, p):
        U_best, V_best, W_best = U_block, V_block, W_block
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
