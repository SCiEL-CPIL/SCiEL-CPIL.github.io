"""Matrix multiplication 2x4x4 rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
_DEADLINE = 135.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, time_limit):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    deadline = time.time() + time_limit
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    total_eqs = n * m * p
    
    # We need: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of bilinear equations in the entries.
    
    # Strategy: fix U and V randomly, solve for W via linear system, then check.
    # Actually, fix U and W, solve for V; or use a more clever approach.
    
    # Better: parameterize. For each column c, we have vectors u_c (n*m), v_c (m*p), w_c (p*n).
    # The constraint is: for all (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
    
    # Let's try: fix U to be a random matrix, then for each (i,j), the equation becomes:
    # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k.
    # This is linear in the products V[j*p+k,c]*W[k*n+i,c].
    
    # Alternative approach: use the fact that we can write the tensor as a sum of
    # rank-1 tensors. Let's try a structured ansatz.
    
    # For n=2, let's try to exploit the small size.
    # Let me try: U has 8 rows, V has 16 rows, W has 8 rows.
    
    # Random search: pick random U, V, solve for W.
    # For fixed U and V, the equations are linear in W:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Group by (k,i): for fixed (k,i), sum over j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Wait, no. For fixed (i,j,k), it's one equation. There are 32 equations and 8*r unknowns in W.
    # If r >= 4, we have 8*r >= 32 unknowns... wait, W has p*n = 8 rows and r columns, so 8*r unknowns.
    # 32 equations, 8*r unknowns. Need 8*r >= 32, so r >= 4.
    
    # But the equations are not all independent in W for fixed U,V.
    # Let's set up the linear system properly.
    
    # For fixed U (8 x r) and V (16 x r), the equations for W are:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is linear in W. The coefficient of W[k*n+i,c] in equation (i,j,k) is U[i*m+j,c]*V[j*p+k,c].
    
    # So we have a 32 x (8*r) linear system. Let's build it.
    
    for attempt in range(20000):
        if time.time() > deadline:
            break
        
        # Random U and V
        U = np.random.randn(n_rows, r)
        V = np.random.randn(m_rows, r)
        
        # Build linear system for W
        # A is 32 x (8*r), b is 32
        A = np.zeros((total_eqs, p_rows * r))
        b = np.ones(total_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        # W[k*n+i, c] has index (k*n+i)*r + c
                        w_idx = (k * n + i) * r + c
                        A[eq_idx, w_idx] = U[i * m + j, c] * V[j * p + k, c]
                    eq_idx += 1
        
        # Solve A @ W_flat = b
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_rows, r)
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            # Convert to integer/half-integer if possible (they should be exact)
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(8, 16, 8) = 16
    lower_bound = max(n * m, m * p, p * n)
    
    for r in range(lower_bound, best_rank):
        elapsed = time.time() - start_time
        if elapsed > 130:  # leave 5s margin
            break
        time_left = 130 - elapsed
        if time_left < 2:
            break
        
        result = _try_rank(r, n, m, p, time_left)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # found minimal rank
    
    return best
