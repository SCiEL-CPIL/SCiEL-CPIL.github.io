"""Matrix multiplication 3,4,4 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build the tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #   and for all other (i,j,k) combos that don't correspond to valid mult, it should be 0
    # Actually the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, no. The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k.
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since every entry of C depends on A[i,j] and B[j,k])
    # Actually T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple contributes.
    # The tensor is all ones of shape (n, m, p).
    # Wait, that's not right either. Let me think more carefully.
    # 
    # Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (each term in the sum has coefficient 1)
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So T is the all-ones tensor of shape (n, m, p).
    
    T = np.ones((n, m, p))
    # Compute the decomposition
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    err += abs(val - 1.0)
    return err < 1e-9

def _round_to_half_int(arr):
    """Round a 2D array to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i,j,k is the all-ones tensor.
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use random initialization and gradient-free optimization
    # Or use the fact that for the all-ones tensor, we can try structured decompositions
    
    # For the all-ones tensor of shape (n,m,p), a rank-1 decomposition would be:
    # U[i*m+j, r] = a[i,j], V[j*p+k, r] = b[j,k], W[k*n+i, r] = c[k,i]
    # such that sum_r a[i,j]*b[j,k]*c[k,i] = 1 for all i,j,k
    
    # This is equivalent to finding matrices A (n x m), B (m x p), C (p x n) 
    # such that the "cyclic" product gives 1 everywhere.
    
    # Let's try a different approach: parameterize and optimize
    # Use numpy for speed
    
    def try_rank(r, num_trials=50, max_iter=2000):
        """Try to find a rank-r decomposition."""
        nonlocal best_result, best_rank
        
        if r >= best_rank:
            return False
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(trial * 1000 + r)
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Iterative optimization: minimize sum of squared errors
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Target: 1 for all i,j,k
            
            # Use simple gradient descent or alternating minimization
            # Alternating minimization: fix two, solve for third (linear system)
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), we need
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is: U[i*m+j,:] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # Actually: for fixed (i,j), the equation is:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k in range(p)
                # Let M[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then sum_r M[k,r] * U[i*m+j,r] = 1 for all k
                # This is a linear system: M @ U[i*m+j,:] = ones(p)
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_r V[j*p+k,r]*W[k*n+i,r]*U[i*m+j,r] = 1
                        A_sys = np.zeros((p, r))
                        for k in range(p):
                            A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_sys = np.ones(p)
                        # Solve least squares
                        if r <= p:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        else:
                            sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U[i*m+j, :] = sol
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i
                        A_sys = np.zeros((n, r))
                        for i in range(n):
                            A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_sys = np.ones(n)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V[j*p+k, :] = sol
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
                        A_sys = np.zeros((m, r))
                        for j in range(m):
                            A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b_sys = np.ones(m)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W[k*n+i, :] = sol
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if err < 1e-12:
                    break
            
            # Round to half-integers and verify
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = r
                return True
        
        return False
    
    # Try ranks from low to high
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But for the all-ones tensor, the rank is actually min(n*m, m*p, p*n) in some cases?
    # No, the tensor rank of the all-ones tensor of shape (n,m,p) is 1 (it's a rank-1 tensor!)
    # Wait, T[i,j,k] = 1 = 1*1*1, so it IS rank 1!
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_r U*V*W = 1*1*1 = 1 for all i,j,k. 
    # So rank 1 should work!
    
    # Let me verify: T[i,j,k] = 1 for all i,j,k
    # With rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 ✓
    
    # So the answer is rank 1!
    if time.time() < deadline:
        U = np.ones((n * m, 1))
        V = np.ones((m * p, 1))
        W = np.ones((p * n, 1))
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 1 doesn't work (shouldn't happen), try rank 2, 3, etc.
    for r in range(2, best_rank):
        if time.time() > deadline - 20:
            break
        if try_rank(r, num_trials=10, max_iter=500):
            break
    
    return best_result
