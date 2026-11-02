"""Matrix multiplication 3,4,7 rank-minimal decomposition."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools
import random

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W give exact matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against identity: T[i,j,k] = 1 if this is a valid multiplication term
    # Actually T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # Wait no. The tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Compute decomposition
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    if np.max(np.abs(T - T_decomp)) > tol:
        return False
    return True

def _try_rank(r, n, m, p, max_iters=200, seed=42, time_limit=30):
    """Try to find a rank-r decomposition."""
    np.random.seed(seed)
    random.seed(seed)
    start = time.time()
    
    best_loss = float('inf')
    best_params = None
    
    # Initial guess: use schoolbook and try to zero out some components
    # Or use random initialization
    for attempt in range(5):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to avoid scaling issues
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 1e-10:
                scale = norm ** (1/3)
                U[:,c] /= scale
                V[:,c] /= scale
                W[:,c] /= scale
        
        # Define loss function
        def loss(params):
            U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r:n*m*r+m*p*r], params[n*m*r+m*p*r:]
            U_r = U_flat.reshape(n*m, r)
            V_r = V_flat.reshape(m*p, r)
            W_r = W_flat.reshape(p*n, r)
            
            T_decomp = np.zeros((n, m, p))
            for rr in range(r):
                u = U_r[:, rr].reshape(n, m)
                v = V_r[:, rr].reshape(m, p)
                w = W_r[:, rr].reshape(p, n)
                T_decomp += np.einsum('ij,jk,ki->ijk', u, v, w)
            
            # Target: all ones
            target = np.ones((n, m, p))
            return np.sum((T_decomp - target)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try L-BFGS
        res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'maxfun': max_iters*10})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
        
        if best_loss < 1e-12:
            break
    
    if best_params is None:
        return None
    
    if best_loss < 1e-9:
        U = best_params[:n*m*r].reshape(n*m, r)
        V = best_params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W = best_params[n*m*r+m*p*r:].reshape(p*n, r)
        return U, V, W
    
    return None

def _try_structured_rank(r, n, m, p, time_limit=20):
    """Try structured approaches for rank r."""
    start = time.time()
    
    # Approach 1: Use the fact that for matrix multiplication, we can use
    # the identity: sum over j of A[i,j]B[j,k]
    # Try to find a low-rank factorization using algebraic identities
    
    # For small cases, try to use known results
    # The rank of the 2x2x2 tensor is 7 (Strassen)
    # For 3x4x7, we can try to embed smaller results
    
    # Try: use a basis of polynomials or other structured vectors
    # This is heuristic - try random but structured initializations
    
    for attempt in range(10):
        if time.time() - start > time_limit:
            break
        
        np.random.seed(42 + attempt * 1000)
        
        # Try with some zero patterns
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Apply random sparsification
        for c in range(r):
            mask_u = np.random.rand(n*m) > 0.5
            mask_v = np.random.rand(m*p) > 0.5
            mask_w = np.random.rand(p*n) > 0.5
            U[mask_u, c] = 0
            V[mask_v, c] = 0
            W[mask_w, c] = 0
        
        def loss(params):
            U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r:n*m*r+m*p*r], params[n*m*r+m*p*r:]
            U_r = U_flat.reshape(n*m, r)
            V_r = V_flat.reshape(m*p, r)
            W_r = W_flat.reshape(p*n, r)
            
            T_decomp = np.zeros((n, m, p))
            for rr in range(r):
                u = U_r[:, rr].reshape(n, m)
                v = V_r[:, rr].reshape(m, p)
                w = W_r[:, rr].reshape(p, n)
                T_decomp += np.einsum('ij,jk,ki->ijk', u, v, w)
            
            target = np.ones((n, m, p))
            return np.sum((T_decomp - target)**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
        
        if res.fun < 1e-9:
            U = res.x[:n*m*r].reshape(n*m, r)
            V = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W = res.x[n*m*r:].reshape(p*n, r)
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank
    # Lower bound: at least max(n, m, p) and at least (n*m*p)^(1/3) roughly
    # Try from a reasonable lower bound upward
    lower_bound = max(n, m, p)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(r, n, m, p, max_iters=150, seed=42, time_limit=20)
        if result is not None:
            U, V, W = result
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        # Also try structured approach
        if time.time() > deadline - 30:
            break
        result2 = _try_structured_rank(r, n, m, p, time_limit=15)
        if result2 is not None:
            U, V, W = result2
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best_result
