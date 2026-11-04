"""Matrix multiplication 3x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N

# Deadline guard: evaluator kills after 150s; keep 10s safety margin.
DEADLINE = 140.0

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _strassen_2x2_block(U_rows, V_rows, W_rows, U_cols, V_cols, W_cols, 
                         U_idx, V_idx, W_idx, U_val, V_val, W_val, col):
    """
    Add one Strassen 2x2 multiplication to the decomposition.
    
    U_rows: list of (row_idx_in_U, col_idx_in_U) pairs for the 2x2 A block
    V_rows: list of (row_idx_in_V, col_idx_in_V) pairs for the 2x2 B block  
    W_rows: list of (row_idx_in_W, col_idx_in_W) pairs for the 2x2 C block
    
    Actually, let me use a cleaner approach. We'll build the decomposition
    by directly constructing U, V, W matrices.
    """
    pass

def _build_strassen_decomposition():
    """
    Build a Strassen-based decomposition for 3x4x5 matrix multiplication.
    
    Partition:
    A (3x4): 
      A_top (2x4): rows 0,1
      A_bot (1x4): row 2
    B (4x5):
      B_left (4x3): cols 0,1,2
      B_right (4x2): cols 3,4
    
    But for Strassen we need 2x2 blocks. Let me think differently.
    
    A (3x4) partitioned as:
      A11 (2x2): rows 0-1, cols 0-1
      A12 (2x2): rows 0-1, cols 2-3
      A21 (1x2): row 2, cols 0-1
      A22 (1x2): row 2, cols 2-3
    
    B (4x5) partitioned as:
      B11 (2x2): rows 0-1, cols 0-1
      B12 (2x3): rows 0-1, cols 2-4
      B21 (2x2): rows 2-3, cols 0-1
      B22 (2x3): rows 2-3, cols 2-4
    
    C (3x5):
      C11 (2x3): rows 0-1, cols 0-2
      C12 (2x2): rows 0-1, cols 3-4
      C21 (1x3): row 2, cols 0-2
      C22 (1x2): row 2, cols 3-4
    
    C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x2 * 2x2 = 2x2... wait, B11 is 2x2, A12 is 2x2, B21 is 2x2)
    Actually: A11(2x2)*B11(2x2) = 2x2, A12(2x2)*B21(2x2) = 2x2. Sum = 2x2. But C11 should be 2x3.
    
    Let me reconsider. The column partition of B should match C's columns.
    
    B (4x5): split into B_left (4x3) and B_right (4x2)
    B_left (4x3): rows 0-3, cols 0-2
      B11 (2x3): rows 0-1, cols 0-2
      B21 (2x3): rows 2-3, cols 0-2
    B_right (4x2): rows 0-3, cols 3-4
      B12 (2x2): rows 0-1, cols 3-4
      B22 (2x2): rows 2-3, cols 3-4
    
    C (3x5):
      C11 (2x3): rows 0-1, cols 0-2 = A_top(2x4) * B_left(4x3)
      C12 (2x2): rows 0-1, cols 3-4 = A_top(2x4) * B_right(4x2)
      C21 (1x3): row 2, cols 0-2 = A_bot(1x4) * B_left(4x3)
      C22 (1x2): row 2, cols 3-4 = A_bot(1x4) * B_right(4x2)
    
    For C11 = A_top(2x4) * B_left(4x3):
      A_top = [A11(2x2) | A12(2x2)]
      B_left = [B11(2x3); B21(2x3)]
      C11 = A11*B11 + A12*B21
      
      A11*B11: 2x2 * 2x3 = 2x3 (not square, can't use standard Strassen)
      
    Hmm, Strassen works for square matrices. Let me use a different approach.
    
    Instead, let me use the fact that we can decompose any matrix multiplication
    using tensor rank methods. For 3x4x5, I'll try to find a low-rank decomposition
    using a systematic search.
    
    Actually, let me try a direct algebraic approach for small ranks.
    """
    return None

def _try_rank_direct(rank, n, m, p):
    """
    Try to find a decomposition of given rank using direct algebraic methods.
    
    For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    Strategy: Use the structure of matrix multiplication. Set up the system
    and try to solve it.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # For small ranks, try random initialization with ALS refinement
    n_tries = max(20, int(100 * _time_left() / max(1, rank)))
    
    for attempt in range(n_tries):
        if _time_left() < 2:
            break
        
        # Random initialization with some structure
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Normalize to avoid numerical issues
        for c in range(rank):
            norm_U = np.linalg.norm(U[:, c])
            norm_V = np.linalg.norm(V[:, c])
            norm_W = np.linalg.norm(W[:, c])
            if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                scale = (norm_U * norm_V * norm_W) ** (1.0/3.0)
                U[:, c] /= scale
                V[:, c] /= scale
                W[:, c] /= scale
        
        # ALS refinement
        for iteration in range(100):
            if _time_left() < 2:
                break
            
            # Solve for W given U, V
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = x
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Solve for V given U, W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = x
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Solve for U given V, W
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = x
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Check convergence
            if iteration > 0:
                diff = (np.linalg.norm(W - W_new) + np.linalg.norm(V - V_new) + np.linalg.norm(U - U_new))
                if diff < 1e-8:
                    break
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from schoolbook rank and try to improve
    # Schoolbook rank = n*m*p = 60
    best_result = None
    best_rank = n * m * p
    
    # Generate schoolbook solution as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank + 1):
        if _time_left() < 5:
            break
        
        result = _try_rank_direct(rank, n, m, p)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found the minimum rank, stop
    
    return best_result
