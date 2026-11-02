import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

def _tensor_to_matrices(U, V, W):
    """
    Convert factor matrices to the tensor form for verification.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    n, m, p = DIMENSIONS
    R = U.shape[1]
    # T_{i, j, k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to check if this equals 1 for all i, j, k.
    # Efficient check:
    # For each r, the term is (U[:, r] * V[:, r]) reshaped? No.
    # Let's just do the direct sum for verification, it's small.
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # The term contributes to T[i, j, k] as u[i, j] * v[j, k] * w[k, i]
        # This is a sum over r.
        # We can compute this by iterating r.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T

def _verify(U, V, W, tol=1e-6):
    T = _tensor_to_matrices(U, V, W)
    # The target tensor T_target[i, j, k] = 1 for all i, j, k
    # Wait, is it 1?
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # The tensor T defines the coefficients.
    # T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # It is 1 if the indices are valid, 0 otherwise.
    # Since all indices are valid in the ranges, T_target is all 1s.
    return np.allclose(T, 1.0, atol=tol)

def _als_decompose(n, m, p, rank, max_iter=50, tol=1e-8):
    """
    Alternating Least Squares for CP decomposition of the all-ones tensor.
    """
    # Initialize
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize to prevent divergence
    for r in range(rank):
        norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
        if norm > 0:
            U[:, r] /= norm ** (1/3)
            V[:, r] /= norm ** (1/3)
            W[:, r] /= norm ** (1/3)
        else:
            U[:, r] = 1.0 / np.sqrt(n*m)
            V[:, r] = 1.0 / np.sqrt(m*p)
            W[:, r] = 1.0 / np.sqrt(p*n)

    prev_err = np.inf
    for it in range(max_iter):
        # Update U
        # We want to minimize || T - sum_r u_r v_r w_r ||^2
        # Fix V, W.
        # For each row of U (index a = i*m+j), we solve for u_a.
        # This is a linear system.
        
        # Construct the system for U:
        # Let M_U be a matrix of size (n*m) x (rank * (m*p) * (p*n)?) No, that's huge.
        # Better: Iterate over the "flattened" target.
        
        # Reshape V and W to facilitate multiplication.
        # V is (m*p, R). W is (p*n, R).
        # For a fixed i, j, the equation is:
        # 1 = sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] * W[k*n+i, r])
        # Let's define a matrix M where M[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
        # Then we have: 1 (vector of length n*m) = U^T M^T ?
        # Actually, let's just do gradient descent or simple least squares per row.
        
        # Efficient ALS:
        # Precompute V_reshaped: (m, p, R)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Target is 1 for this (i, j)
                # Coefficients for each r: c_r = sum_k V[j, k, r] * W[k, i, r]
                # c = np.einsum('kr,ir->r', V_3d[j], W_3d[:, i])
                c = np.sum(V_3d[j, :, :] * W_3d[:, i, :], axis=0)
                # We want u such that sum_r u_r c_r = 1
                # This is underdetermined if rank > 1.
                # We want the minimum norm solution? Or just one solution?
                # In ALS, we usually fix the other factors and solve for the best fit.
                # Since the target is a constant 1, and we have multiple (i,j) sharing the same U row?
                # No, each (i,j) has its own U entry.
                # So for each (i,j), we have a scalar equation: u[i*m+j, :] dot c = 1.
                # This is 1 equation, rank unknowns.
                # We can set u[i*m+j, :] = c / (c dot c) to satisfy c dot u = 1.
                # This is the minimum norm solution.
                denom = np.dot(c, c)
                if denom > 1e-12:
                    U_new[i*m+j, :] = c / denom
                else:
                    U_new[i*m+j, :] = 0
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Target 1
                # c_r = sum_i U[i, j, r] * W[k, i, r]
                # U_3d = U.reshape(n, m, rank)
                # W_3d = W.reshape(p, n, rank)
                U_3d = U_new.reshape(n, m, rank)
                # c = sum_i U_3d[i, j, :] * W_3d[k, i, :]
                c = np.sum(U_3d[:, j, :] * W_3d[k, :, :], axis=0)
                denom = np.dot(c, c)
                if denom > 1e-12:
                    V_new[j*p+k, :] = c / denom
                else:
                    V_new[j*p+k, :] = 0
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # Target 1
                # c_r = sum_j U[i, j, r] * V[j, k, r]
                U_3d = U_new.reshape(n, m, rank)
                V_3d_new = V_new.reshape(m, p, rank)
                # c = sum_j U_3d[i, j, :] * V_3d_new[j, k, :]
                c = np.sum(U_3d[i, :, :] * V_3d_new[:, k, :], axis=0)
                denom = np.dot(c, c)
                if denom > 1e-12:
                    W_new[k*n+i, :] = c / denom
                else:
                    W_new[k*n+i, :] = 0
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence (optional, just run max_iter)
        # err = _tensor_to_matrices(U, V, W)
        # if np.allclose(err, 1.0, atol=tol): break
        
    return U, V, W

def _round_to_half(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _try_rank(n, m, p, rank, deadline):
    """Try to find a valid decomposition of the given rank."""
    if time.time() > deadline:
        return None
    
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    # Try a few random initializations
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        U, V, W = _als_decompose(n, m, p, rank, max_iter=20, tol=1e-6)
        
        # Round
        U_r, V_r, W_r = _round_to_half(U, V, W)
        
        # Verify
        if _verify(U_r, V_r, W_r, tol=1e-6):
            return U_r, V_r, W_r
        
        # Keep track of best error for potential future use (not used here)
        # T = _tensor_to_matrices(U_r, V_r, W_r)
        # err = np.max(np.abs(T - 1.0))
        # if err < best_err:
        #     best_err = err
        #     best_U, best_V, best_W = U_r, V_r, W_r
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy: Try ranks from low to high.
    # Theoretical lower bound for 5x5 is 18 (unproven) or 20 (known).
    # For 3x5x5, if independent, 3*20=60.
    # But maybe there is a shared structure allowing lower rank.
    # Let's try 55, 56, 57, 58, 59, 60.
    
    # Fallback: Schoolbook (Rank 75)
    def _build_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a better rank
    for rank in range(50, 76):
        if time.time() > deadline - 10:
            break
        
        res = _try_rank(n, m, p, rank, deadline - 5)
        if res is not None:
            U, V, W = res
            # Final verification
            if _verify(U, V, W, tol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If all fails, return schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
