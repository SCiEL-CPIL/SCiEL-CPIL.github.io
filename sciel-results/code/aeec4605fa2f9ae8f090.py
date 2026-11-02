"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] where T[i,j,k]=1 if this is a valid (i,j,k) triple."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrices(T, U, V, W, rank):
    """Convert tensor T to factor matrices. U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]."""
    n, m, p = T.shape
    U_mat = np.zeros((n * m, rank))
    V_mat = np.zeros((m * p, rank))
    W_mat = np.zeros((p * n, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U_mat[i * m + j, r] = T[i, j, r]
        for j in range(m):
            for k in range(p):
                V_mat[j * p + k, r] = T[j, k, r]
        for k in range(p):
            for i in range(n):
                W_mat[k * n + i, r] = T[i, k, r]
    return U_mat, V_mat, W_mat

def _reconstruct_from_factors(U, V, W, n, m, p, rank):
    """Reconstruct the tensor from factors."""
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return T

def _als_step(T, U, V, W, n, m, p, rank, max_iter=50, tol=1e-10):
    """Alternating least squares to fit T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    for it in range(max_iter):
        # Update U: for each r, solve for U[:,r] given V[:,r] and W[:,r]
        for r in range(rank):
            # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Fix V[:,r] and W[:,r], solve for U[:,r]
            # This is a linear system: for each (i,j), sum over k: T[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # So U[i*m+j,r] = sum_k T[i,j,k] / (V[j*p+k,r] * W[k*n+i,r]) ... but this isn't quite right for ALS
            # Better: treat it as a linear least squares problem
            # Reshape: for fixed r, we have T[i,j,k] ≈ U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
            # Let A[i*m+j, j*p+k] = V[j*p+k, r] * W[k*n+i, r] (but this depends on i too through W)
            # Actually this is more complex. Let me use a different approach.
            pass
        
        # Simpler approach: use the fact that for each r, the rank-1 tensor is u_r ⊗ v_r ⊗ w_r
        # where u_r = U[:,r], v_r = V[:,r], w_r = W[:,r]
        # We want sum_r u_r ⊗ v_r ⊗ w_r ≈ T
        # ALS: fix two factors, solve for the third
        
        # Update U: fix V and W
        for r in range(rank):
            # For each (i,j), U[i*m+j, r] should satisfy:
            # sum_k T[i,j,k] = U[i*m+j,r] * sum_k V[j*p+k,r] * W[k*n+i,r] ... NO
            # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This means for fixed r, the contribution to T[i,j,k] from rank r is:
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So T[i,j,k] - sum_{r'≠r} U[i*m+j, r']*V[j*p+k, r']*W[k*n+i, r'] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is still coupled. Let me use a proper ALS formulation.
            pass
        
        # Let me use a cleaner ALS: for each r, the rank-1 component is outer(u_r, v_r, w_r)
        # where u_r[i*m+j] = U[i*m+j, r], etc.
        # For ALS, fix V and W, solve for U:
        # For each (i,j), we want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ≈ T[i,j,k] for all k
        # This is: U[i*m+j, :] @ diag(V[j*p+k, :]) @ W[k*n+i, :] ... no
        # Let me define for fixed (i,j): 
        #   b[k] = T[i,j,k]
        #   A[k, r] = V[j*p+k, r] * W[k*n+i, r]
        #   Then b ≈ A @ U[i*m+j, :]
        # So U[i*m+j, :] = pinv(A) @ b
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                b = T[i, j, :]
                # Solve A @ x = b
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(rank)
                U[i * m + j, :] = x
        
        # Update V: fix U and W
        for j in range(m):
            for k in range(p):
                A = np.zeros((m, rank))  # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                b = np.zeros(n)
                for i in range(n):
                    b[i] = T[i, j, k]
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(rank)
                V[j * p + k, :] = x
        
        # Update W: fix U and V
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))  # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                b = np.zeros(m)
                for j in range(m):
                    b[j] = T[i, j, k]
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    x = np.zeros(rank)
                W[k * n + i, :] = x
        
        # Check convergence
        T_recon = _reconstruct_from_factors(U, V, W, n, m, p, rank)
        err = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
        if err < tol:
            break
    
    return U, V, W, err

def _to_rational_approx(mat, denom_max=1000):
    """Convert a float matrix to half-integers (multiples of 0.5) or integers."""
    mat_r = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            # Try to find the closest half-integer
            x = mat[i, j]
            # Round to nearest 0.5
            r = round(x * 2) / 2.0
            mat_r[i, j] = r
    return mat_r

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify that the factors exactly decompose the tensor."""
    T_recon = _reconstruct_from_factors(U, V, W, n, m, p, rank)
    T_target = _build_tensor(n, m, p)
    return np.linalg.norm(T_recon - T_target) < tol

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition as a dict."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try various rank reductions
    # Known lower bounds: for 4x4x7, the rank is at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But the actual minimal rank is likely higher. Let's try to find a good one.
    
    # Strategy: start with random rank-r decomposition and refine with ALS
    random.seed(42)
    
    for target_rank in [best_rank - 1, best_rank - 2, best_rank - 5, best_rank - 10, best_rank - 20]:
        if target_rank < max(n * m, m * p, p * n):
            continue
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random factors
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Run ALS
            U, V, W, err = _als_step(T, U, V, W, n, m, p, target_rank, max_iter=30, tol=1e-8)
            
            if err < 1e-6:
                # Try to convert to rational (half-integer) values
                U_r = _to_rational_approx(U)
                V_r = _to_rational_approx(V)
                W_r = _to_rational_approx(W)
                
                # Verify
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, target_rank, tol=1e-6):
                    # Convert to lists of half-integers
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    
                    # Check if all values are half-integers
                    valid = True
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for val in row:
                                if abs(val * 2 - round(val * 2)) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
        
        if best_rank <= max(n * m, m * p, p * n):
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
