"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for 2x2 * 2x3."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrices(U, V, W, n, m, p, rank):
    """Convert factor matrices to tensor form and verify."""
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # Rank-1 term: T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T

def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition exactly reproduces the tensor."""
    n, m, p = DIMENSIONS
    T = _tensor_to_matrices(U, V, W, n, m, p, len(U[0]))
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    result = arr.copy()
    for idx in np.ndindex(result.shape):
        val = result[idx]
        nearest = np.round(val * 2) / 2.0
        if abs(val - nearest) < tol:
            result[idx] = nearest
    return result

def _try_rank(rank, max_iters=200, tol=1e-10):
    """Try to find a rank-`rank` decomposition using structured search."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    T_flat = T.flatten()
    total = n * m * p
    
    # Parameterization: each rank-1 term has (n*m)*(m*p)*(p*n) parameters
    # But we exploit structure: for term r, U_r is n×m, V_r is m×p, W_r is p×n
    # Total params per term: n*m + m*p + p*n = 4+6+6 = 16
    # Total params: rank * 16
    # Constraints: total = 12 equations
    # So for rank >= 2, we have more params than equations (overdetermined in a good way)
    
    # Strategy: use the bilinear structure to set up equations
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # This is trilinear in the variables. We'll use iterative linearization.
    
    # Initialize with schoolbook-like structure but with some merging
    # For rank r, we assign r "slots" and each slot covers a subset of (i,j,k)
    
    # Simple approach: start from rank-12 and try to zero out (rank-12-r) terms
    # via perturbation
    
    # Better: direct optimization
    # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Objective: ||tensor_from(U,V,W) - T||^2
    
    def objective_and_grad(params):
        """Compute objective and gradient."""
        # params is a flat array of size rank*(n*m + m*p + p*n)
        U = params[:rank*n*m].reshape(rank, n, m)
        V = params[rank*n*m:rank*(n*m+m*p)].reshape(rank, m, p)
        W = params[rank*(n*m+m*p):].reshape(rank, p, n)
        
        # Compute tensor
        T_comp = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_comp[i,j,k] += U[r,i,j] * V[r,j,k] * W[r,k,i]
        
        diff = T_comp - T
        obj = np.sum(diff**2)
        
        # Gradient: d(obj)/d(U[r,i,j]) = 2 * sum_{j',k'} diff[i,j',k'] * V[r,j',k'] * W[r,k',i]
        # Actually: d/dU[r,i,j] sum_{i',j',k'} diff[i',j',k']^2 
        #   = 2 * sum_{j',k'} diff[i,j',k'] * V[r,j',k'] * W[r,k',i]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for k in range(p):
                        s += diff[i,j,k] * V[r,j,k] * W[r,k,i]
                    grad_U[r,i,j] = 2.0 * s
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for i in range(n):
                        s += diff[i,j,k] * U[r,i,j] * W[r,k,i]
                    grad_V[r,j,k] = 2.0 * s
            for k in range(p):
                for i in range(n):
                    s = 0.0
                    for j in range(m):
                        s += diff[i,j,k] * U[r,i,j] * V[r,j,k]
                    grad_W[r,k,i] = 2.0 * s
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad
    
    # Initialize: use schoolbook for first min(rank, 12) terms, rest random small
    np.random.seed(42)
    params = np.zeros(rank * (n*m + m*p + p*n))
    
    # Fill with schoolbook for available slots
    schoolbook_count = min(rank, n*m*p)
    for col in range(schoolbook_count):
        i = col // (m*p)
        j = (col // p) % m
        k = col % p
        # U[r, i, j] = 1
        params[col * (n*m + m*p + p*n) + i*m + j] = 1.0
        # V[r, j, k] = 1
        params[col * (n*m + m*p + p*n) + n*m + j*p + k] = 1.0
        # W[r, k, i] = 1
        params[col * (n*m + m*p + p*n) + n*m + m*p + k*n + i] = 1.0
    
    # For extra slots (if rank > 12), add small random
    for col in range(schoolbook_count, rank):
        offset = col * (n*m + m*p + p*n)
        params[offset:offset + n*m] = np.random.randn(n*m) * 0.1
        params[offset + n*m:offset + n*m + m*p] = np.random.randn(m*p) * 0.1
        params[offset + n*m + m*p:] = np.random.randn(p*n) * 0.1
    
    # L-BFGS-like optimization with line search
    lr = 0.01
    best_obj = float('inf')
    best_params = params.copy()
    
    for it in range(max_iters):
        obj, grad = objective_and_grad(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
        
        if best_obj < tol:
            break
        
        # Gradient step with backtracking line search
        step = -lr * grad
        alpha = 1.0
        for _ in range(20):
            new_params = params + alpha * step
            new_obj, _ = objective_and_grad(new_params)
            if new_obj < obj:
                params = new_params
                break
            alpha *= 0.5
        else:
            # If no improvement, try smaller step
            params = params - 0.001 * grad
        
        # Periodic restart with different initialization
        if it % 50 == 49 and it > 0:
            # Try a different random init
            np.random.seed(it)
            params = np.random.randn(rank * (n*m + m*p + p*n)) * 0.5
            # Add schoolbook structure
            for col in range(schoolbook_count):
                i = col // (m*p)
                j = (col // p) % m
                k = col % p
                offset = col * (n*m + m*p + p*n)
                params[offset + i*m + j] = 1.0
                params[offset + n*m + j*p + k] = 1.0
                params[offset + n*m + m*p + k*n + i] = 1.0
    
    # Final check
    if best_obj < tol * 100:  # Allow some slack for snapping
        U = best_params[:rank*n*m].reshape(rank, n, m).T  # (n*m, rank)
        V = best_params[rank*n*m:rank*(n*m+m*p)].reshape(rank, m, p).T  # (m*p, rank)
        W = best_params[rank*(n*m+m*p):].reshape(rank, p, n).T  # (p*n, rank)
        
        # Snap to half-integers
        U = _snap_to_half_int(U)
        V = _snap_to_half_int(V)
        W = _snap_to_half_int(W)
        
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 12
    
    # First, get the schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for rank in range(schoolbook_rank - 1, 5, -1):  # Try 11, 10, 9, 8, 7, 6
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(rank, max_iters=100)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
