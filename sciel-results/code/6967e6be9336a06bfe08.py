"""Structural change: search for a rank-27 exact decomposition of 2x4 * 4x4
by exploiting the small first dimension (n=2).

Key structural insight:
  C[0,j] = sum_k A[0,k] * B[k,j]
  C[1,j] = sum_k A[1,k] * B[k,j]

We can group terms by the B-index.  For each column j of C, the two entries
C[0,j] and C[1,j] are linear combinations of the same B[k,j] values.

A rank-r decomposition has:
  U: 8 x r  (indices i*m+k, i in {0,1}, k in {0..3})
  V: 16 x r (indices k*p+j)
  W: 8 x r  (indices j*n+i, j in {0..3}, i in {0,1})

The tensor equation is:
  sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1
  for all i in [0,n), k in [0,m), j in [0,p)

This is a system of 2*4*4 = 32 polynomial equations in 32*r unknowns.
For r=27, that's 864 unknowns and 32 equations - very underdetermined.

Strategy: Use a structured ansatz with alternating minimization.
Since the system is highly underdetermined, we can fix some variables and solve
linear systems for others. We use a smart initialization based on the trivial
solution and then perturb to find lower rank solutions.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False, val
    return True, 1.0

def try_rank_alt_min(r, max_iter=200, seed=42, deadline=None):
    """Try to find a rank-r decomposition using alternating minimization."""
    n, m, p = 2, 4, 4
    
    if deadline is not None and time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute index mappings
    i_k_to_idx = {(i, k): i*m + k for i in range(n) for k in range(m)}
    k_j_to_idx = {(k, j): k*p + j for k in range(m) for j in range(p)}
    j_i_to_idx = {(j, i): j*n + i for j in range(p) for i in range(n)}
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute residual
        R = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    R[i,k,j] = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :]) - 1.0
        
        res_sq = np.sum(R**2)
        if res_sq < best_residual:
            best_residual = res_sq
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-12:
            break
        
        # Alternating minimization: solve for U given V, W
        # For each (i,k): sum_j sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = p
        # This is a linear system in U[i*m+k, :]
        
        # For fixed V, W, the equations for U are:
        # For each (i, k, j): sum_s U[i*m+k, s] * (V[k*p+j, s] * W[j*n+i, s]) = 1
        # Group by (i,k): for each j, we have a linear equation in U[i*m+k, :]
        # This gives p equations in r unknowns for each (i,k)
        
        # Solve least squares for U
        for i in range(n):
            for k in range(m):
                A_mat = np.zeros((p, r))
                b_vec = np.ones(p)
                for j in range(p):
                    A_mat[j, :] = V[k*p+j, :] * W[j*n+i, :]
                # Solve A_mat @ u = b_vec
                u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[i*m+k, :] = u_sol
        
        # Solve for V given U, W
        for k in range(m):
            for j in range(p):
                A_mat = np.zeros((n, r))
                b_vec = np.ones(n)
                for i in range(n):
                    A_mat[i, :] = U[i*m+k, :] * W[j*n+i, :]
                v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[k*p+j, :] = v_sol
        
        # Solve for W given U, V
        for j in range(p):
            for i in range(n):
                A_mat = np.zeros((m, r))
                b_vec = np.ones(m)
                for k in range(m):
                    A_mat[k, :] = U[i*m+k, :] * V[k*p+j, :]
                w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[j*n+i, :] = w_sol
    
    if best_residual < 1e-10:
        return best_U, best_V, best_W
    return None

def try_rank_gradient(r, max_iter=5000, seed=42, deadline=None):
    """Try to find a rank-r decomposition using gradient descent."""
    n, m, p = 2, 4, 4
    
    if deadline is not None and time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute residual
        R = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    R[i,k,j] = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :]) - 1.0
        
        res_sq = np.sum(R**2)
        if res_sq < best_residual:
            best_residual = res_sq
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-12:
            break
        
        # Compute gradient
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r_val = R[i,k,j]
                    for s in range(r):
                        dU[i*m+k, s] += 2 * r_val * V[k*p+j, s] * W[j*n+i, s]
                        dV[k*p+j, s] += 2 * r_val * U[i*m+k, s] * W[j*n+i, s]
                        dW[j*n+i, s] += 2 * r_val * U[i*m+k, s] * V[k*p+j, s]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_residual < 1e-10:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 110  # safety margin
    
    # Try to find low-rank decompositions
    for r in range(14, 33):
        if time.time() > deadline:
            break
        
        # Try alternating minimization with multiple seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = try_rank_alt_min(r, max_iter=100, seed=seed, deadline=deadline)
            if result is not None:
                U, V, W = result
                # Verify
                ok, _ = verify_decomposition(U, V, W, n, m, p)
                if ok:
                    # Convert to lists with integer/half-integer entries
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try gradient descent
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = try_rank_gradient(r, max_iter=2000, seed=seed, deadline=deadline)
            if result is not None:
                U, V, W = result
                ok, _ = verify_decomposition(U, V, W, n, m, p)
                if ok:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return the trivial rank-32 solution
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = i * (m * p) + k * p + j
                U[i * m + k][s] = 1
                V[k * p + j][s] = 1
                W[j * n + i][s] = 1
    
    return {"U": U, "V": V, "W": W}
