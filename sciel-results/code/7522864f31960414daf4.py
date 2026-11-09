import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _residuals(x, n, m, p, rank):
    """
    Compute residuals for least_squares.
    x is a flat array of length (n*m + m*p + p*n) * rank.
    Residuals are the differences between the computed tensor entries and 1.
    """
    U_size = n * m * rank
    V_size = m * p * rank
    
    U_flat = x[:U_size]
    V_flat = x[U_size:U_size + V_size]
    W_flat = x[U_size + V_size:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                residuals[idx] = val - 1.0
                idx += 1
    return residuals

def _try_ranked_decomposition(n, m, p, target_rank, max_attempts=10, timeout=5.0):
    """
    Try to find an exact decomposition of rank target_rank using least_squares.
    """
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        x0 = np.random.randn((n * m + m * p + p * n) * target_rank)
        
        try:
            result = least_squares(
                _residuals, 
                x0, 
                args=(n, m, p, target_rank),
                method='lm', 
                max_nfev=200
            )
            
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                x_opt = result.x
                U_size = n * m * target_rank
                V_size = m * p * target_rank
                
                U = x_opt[:U_size].reshape(n * m, target_rank)
                V = x_opt[U_size:U_size + V_size].reshape(m * p, target_rank)
                W = x_opt[U_size + V_size:].reshape(p * n, target_rank)
                
                U_list = [[float(val) for val in row] for row in U]
                V_list = [[float(val) for val in row] for row in V]
                W_list = [[float(val) for val in row] for row in W]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
                    
        except Exception:
            continue
            
    return None

def _build_strassen_2x2():
    """
    Build the Strassen 2x2x2 decomposition.
    Returns U, V, W as numpy arrays of shape (4, 7) each.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen constants
    M1 = 1; M2 = 1; M3 = 1; M4 = 1; M5 = 1; M6 = -1; M7 = 1
    # C1 = M1, C2 = M2, C3 = M3, C4 = M4
    # C5 = M5, C6 = M6, C7 = M7
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # c=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # c=2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # c=3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # c=4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # c=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # c=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_4x4x6_strassen():
    """
    Build a rank-84 decomposition for 4x4x6 using the Strassen 2x2x2 building block.
    This uses the recursive Strassen approach: partition into 2x2 blocks.
    """
    n, m, p = 4, 4, 6
    
    # First, get the 2x2x2 Strassen decomposition.
    U2, V2, W2 = _build_strassen_2x2()
    
    # Now, construct the 4x4x6 decomposition.
    # U has shape (16, rank), V has shape (24, rank), W has shape (24, rank).
    # rank = 84 (12 terms * 7 rank-1 tensors per Strassen).
    rank = 84
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Terms: (a_row, a_col, b_row, b_col, c_row, c_col)
    # A is 4x4, B is 4x6, C is 4x6
    # Block indices: A has 2x2 blocks, B has 2x3 blocks, C has 2x3 blocks
    terms = [
        (0, 0, 0, 0, 0, 0), # A00*B00 -> C00
        (0, 1, 1, 0, 0, 0), # A01*B10 -> C00
        (0, 0, 0, 1, 0, 1), # A00*B01 -> C01
        (0, 1, 1, 1, 0, 1), # A01*B11 -> C01
        (0, 0, 0, 2, 0, 2), # A00*B02 -> C02
        (0, 1, 1, 2, 0, 2), # A01*B12 -> C02
        (1, 0, 0, 0, 1, 0), # A10*B00 -> C10
        (1, 1, 1, 0, 1, 0), # A11*B10 -> C10
        (1, 0, 0, 1, 1, 1), # A10*B01 -> C11
        (1, 1, 1, 1, 1, 1), # A11*B11 -> C11
        (1, 0, 0, 2, 1, 2), # A10*B02 -> C12
        (1, 1, 1, 2, 1, 2), # A11*B12 -> C12
    ]
    
    for t_idx, (ar, ac, br, bc, cr, cc) in enumerate(terms):
        for c in range(7):
            global_c = t_idx * 7 + c
            
            # U: A block at (ar, ac), local indices i_local, j_local
            for i_local in range(2):
                for j_local in range(2):
                    val = U2[i_local * 2 + j_local, c]
                    if val != 0:
                        global_row = ar * 2 + i_local
                        global_col = ac * 2 + j_local
                        U[global_row * 4 + global_col, global_c] += val
                        
            # V: B block at (br, bc), local indices j_local, k_local
            for j_local in range(2):
                for k_local in range(2):
                    val = V2[j_local * 2 + k_local, c]
                    if val != 0:
                        global_row = br * 2 + j_local
                        global_col = bc * 2 + k_local
                        V[global_row * 6 + global_col, global_c] += val
                        
            # W: C block at (cr, cc), local indices k_local, i_local
            for k_local in range(2):
                for i_local in range(2):
                    val = W2[k_local * 2 + i_local, c]
                    if val != 0:
                        global_row = cr * 2 + k_local
                        global_col = cc * 2 + i_local
                        W[global_row * 4 + global_col, global_c] += val
                        
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook()
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try the Strassen-based decomposition of rank 84
    U, V, W = _build_4x4x6_strassen()
    if _verify_decomposition(U, V, W, n, m, p):
        best_rank = 84
        best_result = {"U": U, "V": V, "W": W}
        
    # Try to find even lower rank decompositions using least_squares
    # Start from 50 up to best_rank - 1
    for target_rank in range(50, best_rank):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
            
        attempts = max(2, int(time_left / 5.0))
        
        result = _try_ranked_decomposition(n, m, p, target_rank, 
                                           max_attempts=attempts, 
                                           timeout=min(5.0, time_left - 1.0))
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break

    return best_result
# EVOLVE-BLOCK-END
