"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
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


def _validate(U, V, W, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor)
    
    # Build the tensor via einsum or loops
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _try_rank(r, deadline, best_result):
    """Try to find a rank-r decomposition. Returns (result, found) or (None, False)."""
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # We need to find U (6 x r), V (9 x r), W (6 x r) such that
    # for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of equations. Let's use a randomized approach with least squares.
    # Actually, let's think of it as: for each (i,j,k), we have one equation.
    # Total equations: n*m*p = 18
    # Unknowns: 3*r (entries of U, V, W)
    
    # For r = 18, it's trivially solvable (schoolbook).
    # For r < 18, we need to check if a solution exists.
    
    # Strategy: random initialization + iterative refinement
    # Or: use the fact that this is a tensor decomposition problem
    
    # Let's try a different approach: randomly generate U and V, then solve for W
    # For fixed U and V, the equations are linear in W.
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is linear in W[k*n+i, c] for fixed (i,j,k).
    
    # Actually, let's set up the full linear system.
    # Unknowns: all entries of U, V, W (3*r variables total)
    # But the equations are cubic (products of U, V, W entries), so it's nonlinear.
    
    # Alternative: fix two matrices, solve for the third.
    # Fix U and V. Then for each (i,j,k), we have:
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # Let A_{(i,j,k), c} = U[i*m+j,c] * V[j*p+k,c]
    # Then A_{(i,j,k), :} dot W[k*n+i, :] = 1
    # This gives n*m*p equations in dim_w * r unknowns (entries of W).
    # 18 equations in 6*r unknowns.
    
    # For r >= 3, 6*r >= 18, so the system is underdetermined and may have solutions.
    # Let's try: random U, V, then solve for W via least squares.
    
    start_time = time.time()
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Randomly initialize U and V
        U_rand = np.random.randn(dim_u, r)
        V_rand = np.random.randn(dim_v, r)
        
        # Build the linear system for W
        # For each (i,j,k), equation: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Let's index the unknowns of W: W has shape (dim_w, r), so dim_w*r unknowns.
        # The coefficient for W[k*n+i, c] in equation (i,j,k) is U[i*m+j,c]*V[j*p+k,c]
        
        # Number of equations: n*m*p = 18
        # Number of unknowns: dim_w * r = 6*r
        
        A = np.zeros((n * m * p, dim_w * r))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_rand[i * m + j, :]
                    v_row = V_rand[j * p + k, :]
                    w_row_idx = k * n + i
                    for c in range(r):
                        A[eq_idx, w_row_idx * r + c] = u_row[c] * v_row[c]
                    eq_idx += 1
        
        # Solve A * W_flat = b via least squares
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        W_rand = W_flat.reshape(dim_w, r)
        
        # Check if this solution actually works
        # Verify: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_rand[i * m + j, :]
                    v_row = V_rand[j * p + k, :]
                    w_row = W_rand[k * n + i, :]
                    val = np.dot(u_row, np.multiply(v_row, w_row))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-9:
            # Found a valid decomposition!
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            # Try to round to integers/half-integers
            U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
            
            # Check if rounded version still works
            if _validate(U_rounded, V_rounded, W_rounded, tol=1e-9):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}, True
            else:
                # Return unrounded (still valid as float)
                return {"U": U_list, "V": V_list, "W": W_list}, True
    
    return None, False


def construct():
    start = time.time()
    deadline = start + 120  # safety margin
    
    n, m, p = DIMENSIONS
    # Start from rank 18 and go down
    # Known lower bound: for 2x3x3 matmul, the rank is at least... 
    # The tensor rank of 2x3x3 matmul is known to be 14 (or similar). Let's try from 14 up to 18.
    
    # Actually, let's try ranks from 14 to 18 (since 18 is schoolbook)
    for r in range(14, 19):
        if time.time() > deadline:
            break
        
        result, found = _try_rank(r, deadline, None)
        if found:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
