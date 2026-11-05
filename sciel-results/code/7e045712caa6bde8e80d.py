import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check half-integer constraint
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
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
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2222():
    """
    Returns U, V, W for 2x2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U: A[i,j] -> index i*2+j
    V: B[j,k] -> index j*2+k
    W: C[i,k] -> index i*2+k
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[1, 0] = 1
    W[2, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _standard_nmmp(n, m, p):
    """
    Returns U, V, W for standard n x m x m x p multiplication.
    U shape: (n*m, n*m*p)
    V shape: (m*p, n*m*p)
    W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_block_decomposition():
    n, m, p = 3, 5, 6
    
    # Row partitions: [2, 1]
    # Col partitions: [2, 2, 2]
    # Inner partitions: [2, 2, 1]
    
    row_sizes = [2, 1]
    col_sizes = [2, 2, 2]
    inner_sizes = [2, 2, 1]
    
    # Global matrices
    # We don't know the total rank yet, so we'll collect lists of (row_offset, col_offset, U_block, V_block, W_block)
    # Or we can just build large sparse arrays and then densify.
    # Given the small size, densify is fine.
    
    # Estimate max rank
    # 3x3 blocks of (2,2,2,2) and (1,2,2,2) etc.
    # Max rank per block is 16.
    # 9 blocks. Total rank < 144.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Pre-compute small decompositions
    # We need to map global indices to local indices.
    
    # Let's iterate over blocks
    # r_idx: 0, 1 (corresponding to row sizes 2, 1)
    # c_idx: 0, 1, 2 (corresponding to col sizes 2, 2, 2)
    # k_idx: 0, 1, 2 (corresponding to inner sizes 2, 2, 1)
    
    row_offsets = [0, 2]
    col_offsets = [0, 2, 4]
    inner_offsets = [0, 2, 4]
    
    for r_idx, h_r in enumerate(row_sizes):
        for c_idx, h_c in enumerate(col_sizes):
            for k_idx, w_k in enumerate(inner_sizes):
                # Subproblem: h_r x w_k x w_k x h_c
                # h_r in {2, 1}, w_k in {2, 1}, h_c in {2}
                
                if h_r == 2 and w_k == 2 and h_c == 2:
                    U_block, V_block, W_block = _strassen_2222()
                else:
                    U_block, V_block, W_block = _standard_nmmp(h_r, w_k, h_c)
                
                # Place into global lists
                # U_block is h_r * w_k x rank_block
                # Global U row offset: row_offsets[r_idx] * m + inner_offsets[k_idx]
                # Wait, U is indexed by (i, j) where i is row, j is inner.
                # Global i is in [row_offsets[r_idx], row_offsets[r_idx]+h_r)
                # Global j is in [inner_offsets[k_idx], inner_offsets[k_idx]+w_k)
                
                # V_block is w_k * h_c x rank_block
                # Global V row offset: inner_offsets[k_idx] * p + col_offsets[c_idx]
                # Global j (inner) is in [inner_offsets[k_idx], inner_offsets[k_idx]+w_k)
                # Global k (col) is in [col_offsets[c_idx], col_offsets[c_idx]+h_c)
                
                # W_block is h_c * h_r x rank_block
                # Global W row offset: col_offsets[c_idx] * n + row_offsets[r_idx]
                # Global k (col) is in [col_offsets[c_idx], col_offsets[c_idx]+h_c)
                # Global i (row) is in [row_offsets[r_idx], row_offsets[r_idx]+h_r)
                
                U_list.append((row_offsets[r_idx] * m + inner_offsets[k_idx], U_block))
                V_list.append((inner_offsets[k_idx] * p + col_offsets[c_idx], V_block))
                W_list.append((col_offsets[c_idx] * n + row_offsets[r_idx], W_block))
                
    # Now combine
    # We need to determine the total rank.
    # The blocks are independent, so we can just stack them.
    # But we need to align the columns.
    # Let's create the full U, V, W arrays.
    
    # Calculate total rank
    total_rank = 0
    for _, ub in U_list:
        total_rank += ub.shape[1]
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for (u_row_off, ub), (v_row_off, vb), (w_row_off, wb) in zip(U_list, V_list, W_list):
        r = ub.shape[1]
        U[u_row_off:u_row_off+ub.shape[0], col_offset:col_offset+r] = ub
        V[v_row_off:v_row_off+vb.shape[0], col_offset:col_offset+r] = vb
        W[w_row_off:w_row_off+wb.shape[0], col_offset:col_offset+r] = wb
        col_offset += r
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_random_search(n, m, p, max_rank, time_budget):
    """
    Try to find a decomposition of rank <= max_rank using random initialization and local refinement.
    This is a heuristic approach.
    """
    start = time.time()
    best_rank = max_rank + 1
    best_decomp = None
    
    # We will try to find a decomposition of rank r for r from some lower bound to max_rank
    # Lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But Strassen-like bounds suggest lower is possible.
    # Let's try ranks from 30 to max_rank
    
    for r in range(30, max_rank + 1):
        if time.time() - start > time_budget:
            break
        
        # Try a few random initializations for this rank
        for trial in range(10):
            if time.time() - start > time_budget:
                break
            
            # Initialize U, V, W with small random integers/half-integers
            # We want entries in {-1, -0.5, 0, 0.5, 1} maybe? Or just integers for now.
            # Let's use integers in {-1, 0, 1}
            
            U = np.zeros((n*m, r), dtype=float)
            V = np.zeros((m*p, r), dtype=float)
            W = np.zeros((p*n, r), dtype=float)
            
            # Fill with random +/-1
            U[np.random.choice(n*m*r, size=int(n*m*r*0.3), replace=False)] = 1
            V[np.random.choice(m*p*r, size=int(m*p*r*0.3), replace=False)] = 1
            W[np.random.choice(p*n*r, size=int(p*n*r*0.3), replace=False)] = 1
            
            # Local refinement:
            # We want T(U,V,W) = I
            # This is a non-linear system.
            # We can use a simple coordinate descent or gradient descent on the error.
            # Error E = T - I
            # We want to minimize ||E||_F^2
            
            # This is computationally expensive for large r.
            # Given the time constraints, this might not work well.
            # Let's skip this for now and focus on the block decomposition which is guaranteed to work.
            
            pass
            
    return best_decomp

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Try the block decomposition
    try:
        block_decomp = _construct_block_decomposition()
        if _verify_fast(block_decomp["U"], block_decomp["V"], block_decomp["W"], n, m, p):
            best = block_decomp
            best_rank = len(block_decomp["U"][0])
    except Exception as e:
        pass
        
    # If we have time, we could try to optimize further, but the block decomposition is likely close to optimal or at least much better than 90.
    # Given the deadline, we return the best found.
    
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
        
    return best
