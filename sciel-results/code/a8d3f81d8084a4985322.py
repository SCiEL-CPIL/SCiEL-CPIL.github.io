import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.allclose(T, 1.0, atol=tol)

def _snap_to_half_integer(mat, tol=1e-9):
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        out = []
        for row in snapped:
            out.append([float(x) for x in row])
        return out
    return [list(map(float, row)) for row in arr]

def _build_decomposition(n, m, p):
    # Same as before, rank 44
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1
            
    def _strassen_terms():
        return [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
        ]
        
    terms = _strassen_terms()
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in terms:
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * m + (ac + yi)][col] += ux
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * p + (bc + xi)][col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * n + (cr + zi)][col] += wz
            col += 1
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """Try to reduce rank from current to target_rank by finding linear dependencies."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    R = U_arr.shape[1]
    
    if R <= target_rank:
        return None
    
    # Try to find a subset of columns that still works
    # This is a combinatorial problem, so try heuristics
    
    # Heuristic 1: Try removing one column at a time and see if we can adjust
    # For each column c, try to express it as a combination of others
    # This is complex, so let's try a simpler approach:
    # Randomly project to lower rank and check
    
    # Actually, let's try a structured approach:
    # The rank-44 solution has 16 schoolbook terms + 28 Strassen terms
    # Maybe we can merge some terms
    
    # Try: can we reduce the 1×4×4 part by sharing with Strassen terms?
    # The 1×4×4 uses U[0:4, 0:16] = identity-like
    # The Strassen terms use U[4:12, 16:44]
    
    # Let's try a different decomposition: use 3×4×4 directly with a 
    # generalized Strassen approach
    
    # For now, return None (no improvement found)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Start with rank 44
    result = _build_decomposition(n, m, p)
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        
    best_rank = len(result["U"][0])
    best_result = result
    
    # Try to find lower rank using structured search
    # Key idea: try different block decompositions
    
    # Attempt 1: Try a 3×4×4 decomposition using three 1×4×4 with sharing
    # Not obvious how to share, so skip
    
    # Attempt 2: Try rank-43 by checking if any column is redundant
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    R = U_arr.shape[1]
    
    # Check if the solution has any linearly dependent columns
    # Form the tensor and check rank
    for R_try in range(best_rank - 1, max(best_rank - 5, 30), -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-R_try solution by random search
        # Use the existing solution as a starting point
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization near the current solution
            U_try = U_arr[:, :R_try].copy()
            V_try = V_arr[:, :R_try].copy()
            W_try = W_arr[:, :R_try].copy()
            
            # Add small random perturbation
            U_try += 0.01 * np.random.randn(n * m, R_try)
            V_try += 0.01 * np.random.randn(m * p, R_try)
            W_try += 0.01 * np.random.randn(p * n, R_try)
            
            # Verify
            if _verify_fast(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p, tol=1e-6):
                best_rank = R_try
                best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                break
    
    # Use the best result found
    result = best_result
    result["U"] = _snap_to_half_integer(result["U"])
    result["V"] = _snap_to_half_integer(result["V"])
    result["W"] = _snap_to_half_integer(result["W"])
    
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        
    return result
