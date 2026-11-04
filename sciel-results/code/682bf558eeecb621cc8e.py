"""Optimized tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    The tensor T has indices (i, j, k) where i in [0,2), j in [0,4), k in [0,7).
    We seek rank-r decomposition: T = sum_c U[:,c] V[:,c] W[:,c]
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    
    The constraint is: for all i,j,k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of polynomial equations in the entries of U, V, W.
    
    Strategy: Use the fact that n=2 is small. For each column c, the vectors
    U_c (length 8), V_c (length 28), W_c (length 14) must satisfy the
    multiplicative constraint. We can fix a gauge and search for solutions.
    
    Known lower bound: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    The schoolbook rank is 56. We try to find rank 28-35 decompositions.
    
    Approach: Randomized search with local refinement.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # and check it equals 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-10):
        """Faster verification using numpy."""
        r = U.shape[1]
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can vectorize over c
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def make_schoolbook():
        """Standard schoolbook decomposition of rank n*m*p = 56."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = make_schoolbook()
    
    # Try to find lower rank decompositions
    # The minimum possible rank is at least max(nm, mp, pn) = 28.
    # Let's try ranks from 28 upward.
    
    best_rank = n * m * p
    best_result = (U_fb, V_fb, W_fb)
    
    # Strategy: For small n=2, we can exploit structure.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # 
    # Consider the "border rank" approach. For matrix multiplication,
    # the tensor is T = sum_{i,j,k} e_i^A e_j^B e_k^C.
    #
    # A key insight: if we can find rank-r decomposition, then for any
    # linear combination, we get a bilinear identity.
    #
    # Let's try a structured random search:
    # 1. Randomly initialize U, V, W with small entries
    # 2. Use gradient-based or iterative refinement to satisfy constraints
    # 3. Or use the fact that this is a system of polynomial equations
    
    # Alternative approach: use known results.
    # For 2x4 * 4x7, the tensor rank is at most 56 (schoolbook).
    # It's at least 28.
    # 
    # Let's try a different strategy: exploit the 2-row structure.
    # For each column c, let u_c = (u_c^0, u_c^1) where u_c^i is a 4-vector (row i of U_c)
    # v_c is a 28-vector, w_c is a 14-vector.
    #
    # The constraint is: for all i,j,k: sum_c u_c^i[j] * v_c[j*p+k] * w_c[k*n+i] = 1
    #
    # This is equivalent to: for each i, the sum over c of (u_c^i outer v_c) contracted with w_c appropriately = 1.
    
    # Let's try random search with local optimization for rank r.
    # We'll use a simple approach: random initialization + Newton-like steps.
    
    def try_rank(r, max_iter=500, seed=42):
        """Try to find a rank-r decomposition using random restarts and local search."""
        if time_left() < 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        for attempt in range(20):
            if time_left() < 5:
                break
            
            # Random initialization
            U = rng.randn(nm, r) * 0.1
            V = rng.randn(mp, r) * 0.1
            W = rng.randn(pn, r) * 0.1
            
            # Normalize to avoid trivial solutions
            for c in range(r):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    # Scale so that product of norms is ~1
                    scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                    U[:, c] /= scale
                    V[:, c] /= scale
                    W[:, c] /= scale
            
            # Iterative refinement: for each (i,j,k), the residual is
            # r_{i,j,k} = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
            # We want to minimize sum_{i,j,k} r_{i,j,k}^2
            
            # Use a simple gradient descent on the objective
            # F = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            lr = 0.01
            for it in range(max_iter):
                if time_left() < 10:
                    break
                
                # Compute residuals
                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                
                obj = np.sum(res**2)
                if obj < 1e-12:
                    if verify_fast(U, V, W, 1e-10):
                        return U, V, W
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_ijk = res[i,j,k]
                            # dF/dU[i*m+j,c] = 2*r_ijk * V[j*p+k,c] * W[k*n+i,c]
                            # dF/dV[j*p+k,c] = 2*r_ijk * U[i*m+j,c] * W[k*n+i,c]
                            # dF/dW[k*n+i,c] = 2*r_ijk * U[i*m+j,c] * V[j*p+k,c]
                            dU[i*m+j] += 2.0 * r_ijk * (V[j*p+k] * W[k*n+i])
                            dV[j*p+k] += 2.0 * r_ijk * (U[i*m+j] * W[k*n+i])
                            dW[k*n+i] += 2.0 * r_ijk * (U[i*m+j] * V[j*p+k])
                
                # Gradient step
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                new_obj = np.sum(res**2)  # Approximate; recompute would be more accurate
                # Simple adaptation
                if obj > 0 and new_obj > obj * 1.1:
                    lr *= 0.5
                elif obj > 0 and new_obj < obj * 0.9:
                    lr = min(lr * 1.2, 0.1)
            
            # Check if we converged
            res_final = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res_final[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
            if np.max(np.abs(res_final)) < 1e-10:
                if verify_fast(U, V, W, 1e-10):
                    return U, V, W
        
        return None
    
    # Try ranks from 28 upward
    for r in range(28, best_rank + 1):
        if time_left() < 15:
            break
        
        result = try_rank(r, max_iter=300, seed=r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break  # Found the minimum rank we can achieve
    
    # If we didn't find anything better, return schoolbook
    if best_rank == n * m * p:
        U, V, W = U_fb, V_fb, W_fb
    else:
        U, V, W = best_result
    
    # Final verification
    if not verify_fast(U, V, W, 1e-10):
        # Fall back to schoolbook
        U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
