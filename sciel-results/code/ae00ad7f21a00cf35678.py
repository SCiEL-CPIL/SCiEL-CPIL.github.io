"""Search for a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The multiplication tensor should be 1 for all (i,j,k)
    # Actually, the standard decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = 1 for all valid indices
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape for efficient computation
    # T[i,j,k] = sum_r U_2d[i*m+j, r] * V_2d[j*p+k, r] * W_2d[k*n+i, r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Use einsum: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with schoolbook (rank 18)
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
    
    best = {'U': U, 'V': V, 'W': W}
    best_rank = rank
    
    # Try to reduce rank by searching for lower-rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # Known results suggest rank might be around 12-15 for this problem
    
    # Strategy: Use numerical optimization to find low-rank decompositions
    # Then try to snap to rational/half-integer values
    
    def try_rank(target_rank, max_iters=5000, seed=42):
        """Try to find a decomposition of given rank using optimization."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # We want T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of n*m*p equations with 3*rank*(n*m + m*p + p*n) unknowns
        
        # Use gradient descent on the error
        def error_and_grad():
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = T - 1.0
            total_err = np.sum(err**2)
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = err[i,j,k]
                        for r in range(target_rank):
                            dU[i*m+j, r] += 2*e*V[j*p+k, r]*W[k*n+i, r]
                            dV[j*p+k, r] += 2*e*U[i*m+j, r]*W[k*n+i, r]
                            dW[k*n+i, r] += 2*e*U[i*m+j, r]*V[j*p+k, r]
            
            return total_err, (dU, dV, dW)
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        t = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            err_val, (gU, gV, gW) = error_and_grad()
            t += 1
            bc1 = 1 - beta1**t
            bc2 = 1 - beta2**t
            
            mU = beta1*mU + (1-beta1)*gU; vU = beta2*vU + (1-beta2)*gU**2
            mV = beta1*mV + (1-beta1)*gV; vV = beta2*vV + (1-beta2)*gV**2
            mW = beta1*mW + (1-beta1)*gW; vW = beta2*vW + (1-beta2)*gW**2
            
            mUh = mU/bc1; vUh = vU/bc2
            mVh = mV/bc1; vVh = vV/bc2
            mWh = mW/bc1; vWh = vW/bc2
            
            U -= lr*mUh/(np.sqrt(vUh)+eps)
            V -= lr*mVh/(np.sqrt(vVh)+eps)
            W -= lr*mWh/(np.sqrt(vWh)+eps)
            
            if it % 500 == 0:
                if err_val < 1e-6:
                    break
        
        # Check if we found a good solution
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err < 1e-4:
            # Try to snap to rational values
            def snap_matrix(M, denom=2):
                """Snap values to multiples of 1/denom."""
                snapped = np.round(M * denom) / denom
                return snapped
            
            U_s = snap_matrix(U, 2)
            V_s = snap_matrix(V, 2)
            W_s = snap_matrix(W, 2)
            
            # Verify snapped version
            if _verify_fast(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, 1e-9):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
            
            # Try other denominators
            for denom in [1, 3, 4, 5, 6]:
                U_s = snap_matrix(U, denom)
                V_s = snap_matrix(V, denom)
                W_s = snap_matrix(W, denom)
                if _verify_fast(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, 1e-9):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            
            # If snapping fails, return the numerical solution (may not be accepted)
            # But the task requires integer/half-integer entries
            return None
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            U_l, V_l, W_l = result
            if _verify_fast(U_l, V_l, W_l, n, m, p):
                best = {'U': U_l, 'V': V_l, 'W': W_l}
                best_rank = target_rank
                break  # Found minimum, stop
    
    return best

# EVOLVE-BLOCK-END
