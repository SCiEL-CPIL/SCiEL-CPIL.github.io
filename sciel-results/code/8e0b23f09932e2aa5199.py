import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build schoolbook decomposition (rank 56) as guaranteed fallback
    r_school = n * m * p
    U_school = np.zeros((n * m, r_school))
    V_school = np.zeros((m * p, r_school))
    W_school = np.zeros((p * n, r_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = r_school
    
    def verify(U_r, V_r, W_r, r):
        """Check if decomposition is exact."""
        # T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For matmul, T[i,j,k] = 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U_r[i*m+j, rr] * V_r[j*p+k, rr] * W_r[k*n+i, rr]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U_r, V_r, W_r, r):
        """Vectorized verification."""
        rec = np.zeros((n, m, p))
        for rr in range(r):
            U_mat = U_r[:, rr].reshape(n, m)
            V_mat = V_r[:, rr].reshape(m, p)
            W_mat = W_r[:, rr].reshape(p, n)
            rec += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
        return np.allclose(rec, 1.0, atol=1e-9)
    
    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2.0) / 2.0
    
    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, np) = max(8, 28, 14) = 28
    # Try ranks from 28 to 55
    
    for r in range(28, min(best_rank, 56)):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts with different strategies
        n_restarts = 3 if r < 40 else 2
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Strategy 1: Initialize from a subset of schoolbook + perturbation
            np.random.seed(42 + r * 100 + restart)
            
            if restart == 0:
                # Start from schoolbook columns, randomly select r of them
                idx = np.random.choice(r_school, size=r, replace=False)
                U_r = U_school[:, idx].copy()
                V_r = V_school[:, idx].copy()
                W_r = W_school[:, idx].copy()
            elif restart == 1:
                # Random half-integer initialization
                U_r = np.round(np.random.randn(n * m, r) * 0.5) / 2
                V_r = np.round(np.random.randn(m * p, r) * 0.5) / 2
                W_r = np.round(np.random.randn(p * n, r) * 0.5) / 2
            else:
                # Sparse initialization
                U_r = np.zeros((n * m, r))
                V_r = np.zeros((m * p, r))
                W_r = np.zeros((p * n, r))
                for rr in range(r):
                    # Each rank-1 component touches a few entries
                    u_idx = np.random.choice(n * m, size=max(1, n*m//4), replace=False)
                    v_idx = np.random.choice(m * p, size=max(1, m*p//4), replace=False)
                    w_idx = np.random.choice(p * n, size=max(1, p*n//4), replace=False)
                    U_r[u_idx, rr] = 1.0
                    V_r[v_idx, rr] = 1.0
                    W_r[w_idx, rr] = 1.0
            
            # Alternating least squares with half-integer constraint
            lr = 0.1
            prev_loss = np.inf
            
            for step in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Compute reconstruction
                rec = np.zeros((n, m, p))
                for rr in range(r):
                    U_mat = U_r[:, rr].reshape(n, m)
                    V_mat = V_r[:, rr].reshape(m, p)
                    W_mat = W_r[:, rr].reshape(p, n)
                    rec += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
                
                error = 1.0 - rec
                loss = np.sum(error ** 2)
                
                if loss < 1e-14:
                    break
                
                if loss > prev_loss * 1.05:
                    lr *= 0.5
                elif step % 20 == 0 and loss < prev_loss * 0.95:
                    lr = min(lr * 1.2, 0.5)
                prev_loss = loss
                
                # Gradient descent step
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for rr in range(r):
                    U_mat = U_r[:, rr].reshape(n, m)
                    V_mat = V_r[:, rr].reshape(m, p)
                    W_mat = W_r[:, rr].reshape(p, n)
                    
                    dU_mat = -2.0 * np.einsum('ijk,jk,ki->ij', error, V_mat, W_mat)
                    dV_mat = -2.0 * np.einsum('ijk,ij,ki->jk', error, U_mat, W_mat)
                    dW_mat = -2.0 * np.einsum('ijk,ij,jk->ki', error, U_mat, V_mat)
                    
                    dU[:, rr] = dU_mat.flatten()
                    dV[:, rr] = dV_mat.flatten()
                    dW[:, rr] = dW_mat.flatten()
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
            
            # Final check with half-integer rounding
            U_rounded = round_to_half_int(U_r)
            V_rounded = round_to_half_int(V_r)
            W_rounded = round_to_half_int(W_r)
            
            if verify_fast(U_rounded, V_rounded, W_rounded, r):
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = r
                break  # Found valid solution for this rank
        
        if r < best_rank:
            break  # Found solution, try lower ranks
    
    return best_result
