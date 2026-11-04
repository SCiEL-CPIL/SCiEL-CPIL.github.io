import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2x2 blocks.
    Rank = 42.
    """
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i * m + j, col] = U_s[i * 2 + j_local, c]
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        V[j * p + k, col] = V_s[j_local * 2 + k_local, c]
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        W[k * n + i, col] = W_s[k_local * 2 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _greedy_column_elimination(U, V, W, target_rank, tol=1e-9):
    """
    Systematic rank reduction: try eliminating each column and check if
    the decomposition remains valid. This exploits linear dependencies
    in the Strassen-based construction.
    """
    n, m, p = 2, 4, 6
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try each column for elimination in order
    for col_idx in range(U.shape[1]):
        if _deadline_reached():
            break
        if U.shape[1] <= target_rank:
            break
        
        # Create mask excluding this column
        mask = [i for i in range(U.shape[1]) if i != col_idx]
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check validity
        is_valid, error = _verify_decomposition(U_sub, V_sub, W_sub, tol)
        
        if is_valid:
            # Keep the elimination
            U = U_sub
            V = V_sub
            W = W_sub
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the guaranteed Rank 42 solution
    U42, V42, W42 = _build_2x4x6_decomposition()
    
    # Convert to lists for return
    U42_list = U42.tolist()
    V42_list = V42.tolist()
    W42_list = W42.tolist()
    
    # 2. Verify Rank 42
    is_valid_42, _ = _verify_decomposition(U42, V42, W42)
    
    if not is_valid_42:
        # Fallback to Schoolbook Rank 48 if Strassen fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 3. Try to reduce rank via column elimination (structural approach)
    U_reduced, V_reduced, W_reduced = _greedy_column_elimination(U42, V42, W42, target_rank=41)
    
    # 4. Verify the reduced solution
    is_valid_reduced, error_reduced = _verify_decomposition(U_reduced, V_reduced, W_reduced)
    
    if is_valid_reduced and U_reduced.shape[1] < 42:
        return {
            "U": U_reduced.tolist(),
            "V": V_reduced.tolist(),
            "W": W_reduced.tolist()
        }
    
    # 5. Return Rank 42
    return {
        "U": U42.tolist(),
        "V": V42.tolist(),
        "W": W42.tolist()
    }
