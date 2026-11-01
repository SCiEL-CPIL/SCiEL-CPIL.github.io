"""Tensor decomposition search for 2x3 * 3x9 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the trivial rank-54 decomposition."""
    n, m, p = 2, 3, 9
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, tol=1e-8):
    """Check if U, V, W form a valid tensor decomposition."""
    n, m, p = 2, 3, 9
    r = U.shape[1]
    # For each output (i, k), check: sum_c U[i*m + j, c] * V[j*p + k, c] == W[k*n + i, c]
    # Actually the convention: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and T[i, j, k] should be 1 if this is the matmul tensor (A[i,j]*B[j,k] -> C[i,k])
    # The matmul tensor has T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, let me reconsider. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # is T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) entry of C
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    
    # Efficient check: build the tensor and compare
    # T_decomp[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can compute this as: for each c, the outer product contribution
    # T_decomp = sum_c (U_c reshaped) * (V_c reshaped) * (W_c reshaped) where each is a 3D slice
    
    # Alternative: check via einsum
    # T_decomp[i,j,k] = einsum('ic,jc,kc->ijk', U, V, W)
    # But U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Let's just do it directly for small sizes
    T_decomp = np.zeros((n, m, p))
    for c in range(r):
        u = U[:, c]  # shape (n*m,)
        v = V[:, c]  # shape (m*p,)
        w = W[:, c]  # shape (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    T_target = np.ones((n, m, p))
    return np.max(np.abs(T_decomp - T_target)) < tol

def _verify_fast(U, V, W, tol=1e-8):
    """Faster verification using numpy operations."""
    n, m, p = 2, 3, 9
    r = U.shape[1]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # T_decomp[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_decomp = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    return np.max(np.abs(T_decomp - T_target)) < tol

def _try_rank(r, time_limit=60.0):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    n, m, p = 2, 3, 9
    start_time = time.time()
    
    # Random initialization approach with iterative refinement
    # We use the fact that if we fix U and V, W is determined (if possible)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: random init, then use alternating minimization / gradient descent
    # But this is a non-convex problem. Let's try a simpler approach.
    
    # Approach: Use the structure. For each column c, we have three vectors:
    # u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # such that sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
    
    # This is equivalent to: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c where
    # u_c is indexed by (i,j), v_c by (j,k), w_c by (k,i)
    
    # Let's try random search with local improvement
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Iterative refinement: alternate between updating W given U,V and vice versa
        # Given U and V, for each (i,k), we need:
        # sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in W: sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
        # Let A_ik[c] = sum_j U[i*m+j,c] * V[j*p+k,c]
        # Then A_ik @ W[k*n+i, :] = 1 for all i,k
        
        # Actually this is a system of equations in W.
        # For fixed U,V: the constraint is that for all i,j,k:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Group by (i,k): sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
        # So for each (i,k): M_ik @ w_ik = 1 where M_ik[c] = sum_j U[i*m+j,c]*V[j*p+k,c]
        # and w_ik = W[k*n+i, :]
        
        # This is an overdetermined system (r equations for each of n*p = 18 outputs, but each output has r unknowns)
        # Actually for each (i,k) pair, we have r unknowns W[k*n+i, 0..r-1] and the equation is:
        # sum_c M_ik[c] * W[k*n+i, c] = 1
        # That's 1 equation in r unknowns. Underdetermined if r > 1.
        
        # Similarly for U and V.
        
        # Let's use a different approach: least squares / iterative
        # Initialize W, then update U and V to minimize error
        
        # Actually, let's just do a simple random search with many iterations
        # and check if we can get close to zero error
        
        # Better approach: use the bilinear structure
        # Fix W, then the problem becomes: find U, V such that
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # For fixed W, this is still bilinear in U and V.
        
        # Let's try: random U, V, then solve for W via least squares
        # For each (i,k): minimize ||M_ik @ w_ik - 1||^2
        # where M_ik[c] = sum_j U[i*m+j,c]*V[j*p+k,c]
        
        # This gives w_ik = M_ik^T / (M_ik @ M_ik^T) if M_ik @ M_ik^T != 0
        # (least squares solution to M_ik^T @ w = 1)
        
        for iteration in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Update W given U, V
            for i in range(n):
                for k in range(p):
                    # M[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros(r)
                    for j in range(m):
                        M += U[i * m + j, :] * V[j * p + k, :]
                    # Solve M^T @ w = 1 (least squares)
                    if np.linalg.norm(M) > 1e-12:
                        W[k * n + i, :] = M / np.dot(M, M)
                    else:
                        W[k * n + i, :] = 0
            
            # Update U given V, W
            for i in range(n):
                for j in range(m):
                    # For this (i,j), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # Let N[k] = sum_c V[j*p+k,c]*W[k*n+i,c] ... no that's not right
                    # The equation is: for all k: sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
                    # So U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
                    # This is: B @ u = 1 where B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve B @ u = 1 (least squares)
                    if np.linalg.norm(B) > 1e-12:
                        U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    else:
                        U[i * m + j, :] = 0
            
            # Update V given U, W
            for j in range(m):
                for k in range(p):
                    # For this (j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
                    # V[j*p+k, :] dot (U[i*m+j, :] * W[k*n+i, :]) = 1 for all i
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    if np.linalg.norm(B) > 1e-12:
                        V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    else:
                        V[j * p + k, :] = 0
            
            # Check convergence
            err = _verify_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                break
        
        if best_err < 1e-8:
            return best_U, best_V, best_W
    
    return None

def _verify_error(U, V, W):
    """Compute the max error of the decomposition."""
    n, m, p = 2, 3, 9
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_decomp = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    return float(np.max(np.abs(T_decomp - T_target)))

def construct():
    """Find a low-rank tensor decomposition for 2x3 * 3x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 9
    schoolbook_rank = n * m * p  # 54
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(6, 27, 18) = 27
    # But actual minimum might be higher.
    
    # Try a range of ranks
    ranks_to_try = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 45, 48, 50, 52, 53, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Time budget per rank
        time_budget = min(20.0, max(5.0, deadline - time.time() - 5.0))
        
        result = _try_rank(r, time_limit=time_budget)
        if result is not None:
            U, V, W = result
            # Verify once more
            if _verify_fast(U, V, W, tol=1e-8):
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fall back to schoolbook
    U, V, W = _schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
