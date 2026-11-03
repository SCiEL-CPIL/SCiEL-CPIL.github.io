"""Tensor decomposition search for 3x3x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p, atol=1e-10):
    """Check if U @ V^T reconstructs the matmul tensor exactly."""
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    T_approx = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_approx[i, j, k] = np.dot(Ur[i, j, :], Vr[j, k, :] * Wr[k, i, :])
    
    T_target = np.ones((n, m, p))
    return np.allclose(T_approx, T_target, atol=atol, rtol=atol), T_approx
    
def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(2.0 * x) / 2.0

def _round_to_int(x):
    """Round to nearest integer."""
    return np.round(x)

def _compute_error(Ur, Vr, Wr, n, m, p):
    """Compute max error of the tensor reconstruction."""
    T_approx = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_approx[i, j, k] = np.dot(Ur[i, j, :], Vr[j, k, :] * Wr[k, i, :])
    return np.max(np.abs(T_approx - 1.0))

def _refine(U, V, W, n, m, p, max_iter=300, tol=1e-12, deadline=None):
    """Alternating least squares refinement with deadline guard."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank).copy()
    Vr = V.reshape(m, p, rank).copy()
    Wr = W.reshape(p, n, rank).copy()
    
    prev_err = float('inf')
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                M = Vr[j, :, :] * Wr[:, i, :]  # (p, rank)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    Ur[i, j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                N = Ur[:, j, :] * Wr[k, :, :]  # (n, rank)
                try:
                    sol, _, _, _ = np.linalg.lstsq(N.T, np.ones(n), rcond=None)
                    Vr[j, k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                P = Ur[i, :, :] * Vr[:, k, :]  # (m, rank)
                try:
                    sol, _, _, _ = np.linalg.lstsq(P.T, np.ones(m), rcond=None)
                    Wr[k, i, :] = sol
                except:
                    pass
        
        # Check convergence
        err = _compute_error(Ur, Vr, Wr, n, m, p)
        if err < tol:
            break
        if err > prev_err * (1 + 1e-6):
            # Diverging, revert
            Ur = U.reshape(n, m, rank).copy()
            Vr = V.reshape(m, p, rank).copy()
            Wr = W.reshape(p, n, rank).copy()
            break
        prev_err = err
    
    return Ur.reshape(n*m, rank), Vr.reshape(m*p, rank), Wr.reshape(p*n, rank), err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, return schoolbook as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try ranks from low to high
    # Known: rank(3x3x3) = 23, so rank(3x3x4) >= 23
    # Let's try from 23 upward
    min_possible = 23
    max_rank_to_try = 35
    
    for rank in range(min_possible, min(max_rank_to_try + 1, best_rank)):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts with different strategies
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = hash((rank, restart, n, m, p)) & 0xFFFFFFFF
            np.random.seed(seed)
            
            # Different initialization strategies based on restart index
            if restart % 3 == 0:
                # Standard random
                scale = 0.5
                U_init = np.random.randn(n*m, rank) * scale
                V_init = np.random.randn(m*p, rank) * scale
                W_init = np.random.randn(p*n, rank) * scale
            elif restart % 3 == 1:
                # Sparse initialization - each row has a few nonzeros
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                for r in range(rank):
                    # Pick random non-zero positions
                    nz_u = np.random.choice(n*m, size=max(1, n*m//3), replace=False)
                    nz_v = np.random.choice(m*p, size=max(1, m*p//3), replace=False)
                    nz_w = np.random.choice(p*n, size=max(1, p*n//3), replace=False)
                    U_init[nz_u, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    V_init[nz_v, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    W_init[nz_w, r] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            else:
                # Structured: try to match the pattern of matmul
                # Initialize with some structure
                U_init = np.random.randn(n*m, rank) * 0.3
                V_init = np.random.randn(m*p, rank) * 0.3
                W_init = np.random.randn(p*n, rank) * 0.3
                # Add some diagonal-like structure
                for r in range(rank):
                    if r < n*m:
                        U_init[r % (n*m), r] += 1.0
                    if r < m*p:
                        V_init[r % (m*p), r] += 1.0
                    if r < p*n:
                        W_init[r % (p*n), r] += 1.0
            
            # Refine with deadline
            U_ref, V_ref, W_ref, err = _refine(U_init, V_init, W_init, n, m, p, max_iter=200, deadline=deadline - 5)
            
            if err < 1e-8:
                # Try various rounding strategies
                found = False
                for round_fn, label in [(_round_to_half, "half"), (_round_to_int, "int"), (lambda x: x, "none")]:
                    if found:
                        break
                    if label == "none":
                        U_cand = U_ref.copy()
                        V_cand = V_ref.copy()
                        W_cand = W_ref.copy()
                    else:
                        U_cand = round_fn(U_ref)
                        V_cand = round_fn(V_ref)
                        W_cand = round_fn(W_ref)
                    
                    valid, _ = _verify(U_cand, V_cand, W_cand, n, m, p, atol=1e-10)
                    if valid:
                        best_rank = rank
                        best_result = {
                            "U": U_cand.tolist(),
                            "V": V_cand.tolist(),
                            "W": W_cand.tolist()
                        }
                        found = True
                        break
                
                if found:
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return best_result
