import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # Target ranks to try, from low to high
    # Lower bound is max(nm, mp, pn) = 12.
    # Let's start from 13.
    min_rank = 13
    max_rank = 24
    
    best_result = None
    best_rank = 25
    
    # Precompute indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # For ALS, we need to map (i,j) -> U_row, (j,k) -> V_row, (k,i) -> W_row
    # And for solving, we need the coefficients.
    
    def als_step(U, V, W, r):
        # Update U
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                # A is p x r
                A = np.zeros((p, r))
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A[k, :] = V[v_idx, :] * W[w_idx, :]
                # Solve A @ x = 1
                # A is p x r. We want x of size r.
                # Least squares: min ||A x - 1||^2
                try:
                    x, residuals, rank, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[u_idx, :] = x
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                # B is n x r
                B = np.zeros((n, r))
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    B[i, :] = U[u_idx, :] * W[w_idx, :]
                try:
                    x, residuals, rank, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[v_idx, :] = x
                except np.linalg.LinAlgError:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # C is m x r
                C = np.zeros((m, r))
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    C[j, :] = U[u_idx, :] * V[v_idx, :]
                try:
                    x, residuals, rank, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[w_idx, :] = x
                except np.linalg.LinAlgError:
                    pass
                    
        return U, V, W

    def calculate_residual(U, V, W, r):
        res = 0.0
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res += (val - 1.0) ** 2
        return res

    def try_snap_and_verify(U, V, W, r):
        # Try to snap values to fractions with small denominators
        # Check if all values are close to k/d
        denominators = [1, 2, 3, 4, 6, 12, 24]
        for d in denominators:
            U_snap = np.zeros_like(U)
            V_snap = np.zeros_like(V)
            W_snap = np.zeros_like(W)
            valid = True
            for i in range(n*m):
                for c in range(r):
                    val = U[i, c]
                    nearest = round(val * d) / d
                    if abs(val - nearest) > 1e-6:
                        valid = False
                        break
                    U_snap[i, c] = nearest
                if not valid: break
            if not valid: continue
            
            for i in range(m*p):
                for c in range(r):
                    val = V[i, c]
                    nearest = round(val * d) / d
                    if abs(val - nearest) > 1e-6:
                        valid = False
                        break
                    V_snap[i, c] = nearest
                if not valid: break
            if not valid: continue
            
            for i in range(p*n):
                for c in range(r):
                    val = W[i, c]
                    nearest = round(val * d) / d
                    if abs(val - nearest) > 1e-6:
                        valid = False
                        break
                    W_snap[i, c] = nearest
                if not valid: break
            if not valid: continue
            
            # Verify exactly
            # Convert to Fraction for exact check

            # Actually, just use float verification with tight tolerance first, then fraction if needed?
            # Float verification with 1e-9 is usually sufficient if snapped to 1/d.
            # But the task asks for exact decomposition.
            # Let's verify with Fractions to be sure.
            
            # Re-convert snapped floats to Fractions
            U_f = [[Fraction(U_snap[i, c]).limit_denominator(d) for c in range(r)] for i in range(n*m)]
            V_f = [[Fraction(V_snap[i, c]).limit_denominator(d) for c in range(r)] for i in range(m*p)]
            W_f = [[Fraction(W_snap[i, c]).limit_denominator(d) for c in range(r)] for i in range(p*n)]
            
            is_exact = True
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = 0
                        for c in range(r):
                            s += U_f[u_idx][c] * V_f[v_idx][c] * W_f[w_idx][c]
                        if s != 1:
                            is_exact = False
                            break
                    if not is_exact: break
                if not is_exact: break
            if not is_exact: break
            
            if is_exact:
                return U_snap, V_snap, W_snap
                
        return None

    # Search loop
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Number of restarts
        # We can increase restarts for lower ranks if time permits, but let's keep it balanced.
        # 50 restarts per rank seems reasonable.
        num_restarts = 50
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Run ALS
            for it in range(200):
                U, V, W = als_step(U, V, W, r)
                # Check residual occasionally
                if it % 20 == 0:
                    res = calculate_residual(U, V, W, r)
                    if res < 1e-8:
                        break
            
            res = calculate_residual(U, V, W, r)
            if res < 1e-6:
                snapped = try_snap_and_verify(U, V, W, r)
                if snapped is not None:
                    U_snap, V_snap, W_snap = snapped
                    # Convert to list of lists
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # If we find a good residual but can't snap, we might store it, but we need exact.
            # We continue searching.
            
    # If no low rank found, return schoolbook
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
