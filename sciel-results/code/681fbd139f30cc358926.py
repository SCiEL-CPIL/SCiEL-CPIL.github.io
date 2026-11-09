import numpy as np
import time

DIMENSIONS = [3, 4, 8]
DEADLINE = 150.0

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_with_linear_solve(n, m, p, R, U, V, deadline):
    """
    Given U (n*m, R) and V (m*p, R), solve for W such that
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    Returns W if exact solution found and entries are half-integers, else None.
    """
    if time.time() > deadline:
        return None
    
    # Build linear system: A @ W_flat = b
    # Variables: W[k*n+i, r] for k in range(p), i in range(n), r in range(R)
    # Total variables: p * n * R
    num_vars = p * n * R
    num_eqs = n * m * p
    
    A_mat = np.zeros((num_eqs, num_vars), dtype=float)
    b_vec = np.ones(num_eqs, dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    # Variable index for W[k*n+i, r]
                    var_idx = (k * n + i) * R + r
                    coeff = U[i * m + j, r] * V[j * p + k, r]
                    if coeff != 0.0:
                        A_mat[idx, var_idx] = coeff
                idx += 1
    
    # Solve the linear system
    try:
        W_flat, residuals, rank_found, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except Exception:
        return None
    
    # Check if the solution is exact
    if not np.allclose(A_mat @ W_flat, b_vec, atol=1e-10):
        return None
    
    W = W_flat.reshape((p * n, R))
    
    # Try to snap to half-integers
    W_snapped = np.round(W * 2) / 2
    
    # Verify with snapped values
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s_val = 0.0
                for r in range(R):
                    s_val += U[i * m + j, r] * V[j * p + k, r] * W_snapped[k * n + i, r]
                if abs(s_val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Also verify U and V are half-integers
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    
    # Full verification with all snapped
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s_val = 0.0
                for r in range(R):
                    s_val += U_snapped[i * m + j, r] * V_snapped[j * p + k, r] * W_snapped[k * n + i, r]
                if abs(s_val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return U_snapped, V_snapped, W_snapped

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              U has shape (n*m, rank)
              V has shape (m*p, rank)
              W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - 10.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try lower ranks using random initialization + linear solve for W
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Also rank >= n + m + p - 2 = 13 (for all-ones tensor, but this is matmul tensor)
    # For matrix multiplication, better bounds exist but let's try a range
    
    # Candidate ranks to try, from low to high
    candidate_ranks = list(range(32, best_rank))
    candidate_ranks.sort()
    
    # For each candidate rank, try multiple random initializations
    np.random.seed(42)
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
        if R >= best_rank:
            break
        
        # Number of random trials for this rank
        # Use more trials for ranks closer to known bounds
        num_trials = 50 if R < 50 else 20 if R < 70 else 10
        
        found = False
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Randomly initialize U and V with small integer/half-integer values
            # Use values from {-1, 0, 1} to increase chance of exact solutions
            U = np.random.randint(-1, 2, (n * m, R)).astype(float)
            V = np.random.randint(-1, 2, (n * p, R)).astype(float)
            
            # Ensure some non-zero entries
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            result = _try_rank_with_linear_solve(n, m, p, R, U, V, deadline)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_rank = R
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break
        
        if found:
            # Continue searching for even lower ranks
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
