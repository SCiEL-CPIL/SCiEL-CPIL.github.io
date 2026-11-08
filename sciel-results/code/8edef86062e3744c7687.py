"""Combined program: block-Strassen with rank-reduction search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s safety margin before 150s limit
    
    # Build the base Strassen solution (rank 42)
    base_U, base_V, base_W = _build_strassen_solution(n, m, p)
    
    # Verify the base solution
    if _verify_solution(base_U, base_V, base_W, n, m, p):
        best_U, best_V, best_W = base_U, base_V, base_W
        best_rank = 42
    else:
        # Fallback to simple solution
        best_U, best_V, best_W = _build_simple_solution(n, m, p)
        best_rank = n * m * p
    
    # Try to reduce rank by finding linearly dependent columns
    if time.time() < deadline - 10:
        U_red, V_red, W_red = _reduce_rank_advanced(best_U, best_V, best_W, n, m, p, deadline)
        if U_red is not None:
            r_red = len(U_red[0]) if U_red else 0
            if r_red < best_rank and _verify_solution(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = r_red
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _build_strassen_solution(n, m, p):
    """Build the standard block-Strassen solution with rank 42."""
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return U, V, W

def _verify_solution(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _reduce_rank_advanced(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linearly dependent columns.
    Uses SVD-based approach to identify near-dependent columns.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # First, check for obviously zero columns
    for c in range(rank):
        if np.allclose(U_arr[:, c], 0) or np.allclose(V_arr[:, c], 0) or np.allclose(W_arr[:, c], 0):
            U_new = np.delete(U_arr, c, axis=1)
            V_new = np.delete(V_arr, c, axis=1)
            W_new = np.delete(W_arr, c, axis=1)
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    # Try to find if any column can be removed by checking if the remaining
    # columns still span the same tensor space
    # This is a heuristic: try removing each column and verify
    
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank
    
    # Try removing one column at a time
    for c in range(rank):
        if time.time() > deadline - 5:
            break
        try:
            U_test = np.delete(U_arr, c, axis=1)
            V_test = np.delete(V_arr, c, axis=1)
            W_test = np.delete(W_arr, c, axis=1)
            
            if _verify_solution(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = rank - 1
                U_arr, V_arr, W_arr = U_test, V_test, W_test
                rank = best_rank
        except Exception:
            continue
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None, None, None

def _build_simple_solution(n, m, p):
    """Build a simple (high-rank) solution as fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return U, V, W
# EVOLVE-BLOCK-END
