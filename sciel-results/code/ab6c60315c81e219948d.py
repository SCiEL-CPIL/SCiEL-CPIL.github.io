"""Fresh program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

n, m, p = DIMENSIONS

def build_target(n, m, p):
    """Build the target multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def compute_error(U, V, W, T):
    """Compute Frobenius norm of error between target and reconstructed tensor."""
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(T - T_rec)

def find_rank9_2x3x3(time_budget):
    """
    Search for a rank 9 decomposition of the 2x3x3 multiplication tensor.
    Returns U, V, W as numpy arrays of shape (6,9), (9,9), (6,9) if found, else None.
    """
    n_s, m_s, p_s = 2, 3, 3
    R = 9
    T_s = build_target(n_s, m_s, p_s)
    
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Try multiple restarts
    while time.time() < start_time + time_budget:
        # Initialize with small random integers
        # Heuristic: Many entries in optimal decompositions are 0, 1, or -1
        U = rng.integers(-1, 2, size=(n_s * m_s, R))
        V = rng.integers(-1, 2, size=(m_s * p_s, R))
        W = rng.integers(-1, 2, size=(p_s * n_s, R))
        
        err = compute_error(U, V, W, T_s)
        
        # Local Search: Coordinate Descent
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            # Iterate through all entries of U, V, W
            for mat_name, mat in [('U', U), ('V', V), ('W', W)]:
                for idx in range(mat.size):
                    if time.time() > start_time + time_budget:
                        break
                    row, col = np.unravel_index(idx, mat.shape)
                    old_val = mat[row, col]
                    # Try -1, 0, 1 (and maybe +/-2 if current is large, but start small)
                    for new_val in [-1, 0, 1]:
                        if new_val == old_val:
                            continue
                        mat[row, col] = new_val
                        new_err = compute_error(U, V, W, T_s)
                        if new_err < err:
                            err = new_err
                            improved = True
                            if err < 1e-9:
                                break
                        else:
                            mat[row, col] = old_val
                    if err < 1e-9:
                        break
                if err < 1e-9:
                    break
            if err < 1e-9:
                break
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-9:
                break
                
    if best_err < 1e-9:
        # Verify and return integers
        U_int = np.round(best_U).astype(int)
        V_int = np.round(best_V).astype(int)
        W_int = np.round(best_W).astype(int)
        T_rec = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.allclose(T_rec, T_s, atol=1e-9):
            return U_int, V_int, W_int
            
    return None

def construct():
    start_time = time.time()
    time_limit = 120  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Try to find rank 9 decomposition for 2x3x3.
    # 2. If found, construct rank 18 decomposition for 2x3x6 by duplication.
    # 3. If not found, fallback to rank 36 schoolbook.
    
    # Allocate 100s for the search
    search_budget = 100
    result = find_rank9_2x3x3(search_budget)
    
    if result is not None:
        U_s, V_s, W_s = result
        # Construct 2x3x6 rank 18
        n, m, p = 2, 3, 6
        R = 18
        
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        
        # The 2x3x6 problem is two independent 2x3x3 problems.
        # U is the same for both (A is shared).
        # We can just repeat the U columns?
        # Wait, the decomposition is T = sum_r u_r v_r w_r.
        # For 2x3x6:
        # Terms 0..8 correspond to B1 (k=0,1,2)
        # Terms 9..17 correspond to B2 (k=3,4,5)
        
        # U: (6, 18)
        # For r in 0..8, U[:, r] = U_s[:, r]
        # For r in 9..17, U[:, r] = U_s[:, r-9]
        for r in range(9):
            U[:, r] = U_s[:, r]
            U[:, r + 9] = U_s[:, r]
            
        # V: (18, 18)
        # Rows of V are indexed by (j, k) where j in 0..2, k in 0..5.
        # For r in 0..8 (first block, k=0..2):
        #   V[j*6 + k, r] = V_s[j*3 + k, r]
        # For r in 9..17 (second block, k=3..5):
        #   V[j*6 + (k+3), r] = V_s[j*3 + k, r-9]
        
        for j in range(3):
            for k in range(3):
                # First block (k=0,1,2)
                V[j * 6 + k, 0:9] = V_s[j * 3 + k, :]
                # Second block (k=3,4,5)
                V[j * 6 + (k + 3), 9:18] = V_s[j * 3 + k, :]
                
        # W: (12, 18)
        # Rows of W are indexed by (k, i) where k in 0..5, i in 0..1.
        # For r in 0..8 (first block, k=0..2):
        #   W[k*2 + i, r] = W_s[k*2 + i, r]
        # For r in 9..17 (second block, k=3..5):
        #   W[(k+3)*2 + i, r] = W_s[k*2 + i, r-9]
        
        for k in range(3):
            for i in range(2):
                # First block
                W[k * 2 + i, 0:9] = W_s[k * 2 + i, :]
                # Second block
                W[(k + 3) * 2 + i, 9:18] = W_s[k * 2 + i, :]
                
        # Verify the full 2x3x6 decomposition
        T_full = build_target(2, 3, 6)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_rec, T_full, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Should not happen if 2x3x3 was correct, but safety check
            pass

    # Fallback: Schoolbook rank 36
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

# EVOLVE-BLOCK-END
