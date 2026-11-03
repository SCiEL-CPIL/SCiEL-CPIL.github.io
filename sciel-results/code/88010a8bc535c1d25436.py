"""Matrix multiplication 2,4,7 tensor decomposition using structured block decomposition + local search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as 4x7, 4x7, 4x7 arrays.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (c+d)e
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = 1.0
    
    # M3 = a(f-h)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = -1.0
    
    # M4 = d(g-e)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = -1.0; W[1, 3] = 1.0; W[2, 3] = 1.0; W[3, 3] = -1.0
    
    # M5 = (a+b)h
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0; W[3, 4] = 1.0
    
    # M6 = (c-d)(f+h)
    U[2, 5] = 1.0; U[3, 5] = -1.0
    V[1, 5] = 1.0; V[3, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (a-c)(e+g)
    U[0, 6] = 1.0; U[2, 6] = -1.0
    V[0, 6] = 1.0; V[2, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return the rank-4 schoolbook decomposition for 2x2 * 2x1.
    A is 2x2, B is 2x1, C is 2x1.
    U: 4x4, V: 2x4, W: 2x4
    """
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    
    # C[0] = a*e + b*f
    U[0, 0] = 1.0; V[0, 0] = 1.0; W[0, 0] = 1.0
    U[1, 1] = 1.0; V[1, 1] = 1.0; W[0, 1] = 1.0
    
    # C[1] = c*e + d*f
    U[2, 2] = 1.0; V[0, 2] = 1.0; W[1, 2] = 1.0
    U[3, 3] = 1.0; V[1, 3] = 1.0; W[1, 3] = 1.0
    
    return U, V, W

def _construct_block_decomposition():
    """
    Construct a block-structured decomposition for 2x4x7.
    
    Strategy: Split the middle dimension (4) into two parts of size 2.
    A is 2x4 = [A_left (2x2), A_right (2x2)]
    B is 4x7 = [B_top (2x7); B_bottom (2x7)]
    C = A_left * B_top + A_right * B_bottom
    
    Each term is 2x2 * 2x7. We can further split 2x7 into 2x3 + 2x4, but let's
    use a more efficient approach: treat 2x2 * 2x7 as a tensor problem.
    
    For 2x2 * 2x7:
    We can split the 7 columns into 2+2+2+1.
    2x2 * 2x2: rank 7 (Strassen)
    2x2 * 2x1: rank 4 (schoolbook)
    Total per 2x2 * 2x7: 7+7+7+4 = 25
    Two such terms: 25 + 25 = 50
    
    This gives rank 50.
    """
    n, m, p = 2, 4, 7
    
    # Split middle dimension into two 2s
    # A_left: rows 0-1, cols 0-1 of A (which is 2x4)
    # A_right: rows 0-1, cols 2-3 of A
    # B_top: rows 0-1 of B (which is 4x7)
    # B_bottom: rows 2-3 of B
    
    # For each 2x2 * 2x7 subproblem, split 7 into 2+2+2+1
    # Use Strassen for 2x2*2x2 and schoolbook for 2x2*2x1
    
    total_rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Process each of the two middle blocks
    for mid_block in range(2):
        # mid_block 0: A_left * B_top
        # mid_block 1: A_right * B_bottom
        
        # Split the 7 columns of B into 2+2+2+1
        col_splits = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for col_start, col_end in col_splits:
            col_width = col_end - col_start
            
            if col_width == 2:
                # Use Strassen for 2x2 * 2x2
                U_s, V_s, W_s = _strassen_2x2x2()
            else:  # col_width == 1
                # Use schoolbook for 2x2 * 2x1
                U_s, V_s, W_s = _schoolbook_2x2x1()
            
            # Now we need to embed these into the full 2x4x7 tensor
            # U is 4x4 for the 2x2 subproblem, but we need to place it in the 2x4 A matrix
            # For mid_block 0: A_left uses cols 0-1 of A
            # For mid_block 1: A_right uses cols 2-3 of A
            
            U_full = np.zeros((8, U_s.shape[1]), dtype=np.float64)  # 2*4 = 8 rows
            V_full = np.zeros((28, V_s.shape[1]), dtype=np.float64)  # 4*7 = 28 rows
            W_full = np.zeros((14, W_s.shape[1]), dtype=np.float64)  # 7*2 = 14 rows
            
            # Place U: A is 2x4, indexed as [i*4+j]
            # For mid_block 0: A cols 0-1 -> indices 0,1 (i=0) and 4,5 (i=1)
            # For mid_block 1: A cols 2-3 -> indices 2,3 (i=0) and 6,7 (i=1)
            if mid_block == 0:
                a_col_offset = 0  # cols 0,1
            else:
                a_col_offset = 2  # cols 2,3
            
            # U_s is 4x? where row index is i*2+j for i,j in 0..1
            for i in range(2):
                for j in range(2):
                    a_idx = i * 4 + (a_col_offset + j)
                    U_full[a_idx, :] = U_s[i * 2 + j, :]
            
            # Place V: B is 4x7, for this subproblem we use rows 0-1 (B_top) or 2-3 (B_bottom)
            # and cols col_start to col_end
            if mid_block == 0:
                b_row_offset = 0  # rows 0,1
            else:
                b_row_offset = 2  # rows 2,3
            
            # V_s is 2x? where row index is i*2+j for i in 0..1 (rows of B subblock), j in 0..col_width-1
            for i in range(2):
                for j in range(col_width):
                    b_idx = (b_row_offset + i) * 7 + (col_start + j)
                    V_full[b_idx, :] = V_s[i * col_width + j, :]
            
            # Place W: C is 7x2, indexed as [k*2+i]
            # W_s is 2x? where row index is i*col_width+j for i in 0..1 (rows of C), j in 0..col_width-1
            # But C is 2x7, so W[k*n+i, r] = W[k*2+i, r]
            for i in range(2):  # row of C
                for j in range(col_width):  # col of C
                    c_idx = (col_start + j) * 2 + i
                    W_full[c_idx, :] = W_s[i * col_width + j, :]
            
            U_blocks.append(U_full)
            V_blocks.append(V_full)
            W_blocks.append(W_full)
            total_rank += U_s.shape[1]
    
    # Concatenate all blocks
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return U, V, W, total_rank

def _local_search_reduce_rank(U, V, W, n, m, p, target_rank, max_time=30):
    """
    Try to reduce the rank from current to target_rank using local search.
    This is a heuristic approach.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # We'll try to find a lower-rank decomposition by optimizing
    # the entries of U, V, W to satisfy the tensor equation with fewer columns.
    
    # This is a hard problem, so we use a simple approach:
    # Try random initializations at the target rank and see if we can find a solution.
    
    best_U, best_V, best_W = None, None, None
    
    for trial in range(100):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization at target rank
        U_try = np.random.randn(n * m, target_rank)
        V_try = np.random.randn(m * p, target_rank)
        W_try = np.random.randn(p * n, target_rank)
        
        # Round to half-integers
        U_try = np.round(U_try * 2) / 2
        V_try = np.round(V_try * 2) / 2
        W_try = np.round(W_try * 2) / 2
        
        if _verify(U_try, V_try, W_try, n, m, p):
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Step 1: Construct the block decomposition (rank 50)
    U, V, W, rank = _construct_block_decomposition()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _construct_schoolbook(n, m, p)
        rank = n * m * p
    
    # Step 2: Try to reduce rank using local search
    if time.time() - start_time < DEADLINE - 20:
        for target in range(rank - 1, max(28, rank - 10), -1):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U_new, V_new, W_new = _local_search_reduce_rank(U, V, W, n, m, p, target, max_time=10)
            if U_new is not None:
                U, V, W = U_new, V_new, W_new
                rank = target
                break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
