"""
Matrix multiplication 3x3 * 3x5 tensor decomposition.
Uses structured search for low-rank decompositions.
"""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build the target tensor T with indices T[a,b,c] where:
    # a = i*m + j (input index for A)
    # b = j*p + k (input index for B)  
    # c = k*n + i (output index for C, following W[k*n+i, r] convention)
    # T[a,b,c] = 1 if a,b,c correspond to valid (i,j,k) triple
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    schoolbook_rank = n * m * p  # 45
    
    # Schoolbook decomposition as fallback
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sch.tolist(), V_sch.tolist(), W_sch.tolist())
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        approx = np.zeros_like(T)
        for r in range(rank):
            approx += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T, approx, atol=tol)
    
    def try_rank(target_rank, max_attempts=3):
        """Try to find a decomposition of given rank using structured approach."""
        if time.time() > deadline - 15:
            return None
        
        # For matrix multiplication, we can use the fact that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is sum over j of (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
        # We need to find rank-r decomposition.
        
        # Strategy: Use the known structure. The tensor for (n,m,p) matrix mult
        # has a natural structure. For small dimensions, we can try to find
        # decompositions using integer/half-integer entries.
        
        # Approach: Try to express the tensor as sum of rank-1 tensors
        # where each rank-1 tensor corresponds to a "term" in a matrix multiplication algorithm.
        
        # For 3x3 * 3x5, the schoolbook is 45 terms.
        # We know that for square matrices, Strassen gives 7 for 2x2.
        # For rectangular, there are known bounds.
        
        # Let's try a direct search for small ranks using the structure.
        
        # Key insight: The tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We want U[a,r], V[b,r], W[c,r] such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # For a fixed (a,b), T[a,b,c] is nonzero only for specific c values.
        # a = i*m+j, b = j'*p+k. For T[a,b,c] to be 1, we need j=j' and c=k*n+i.
        
        # So for each (a,b), there's at most one c where T[a,b,c]=1.
        # This means the tensor is very sparse.
        
        # Let's try ranks from the theoretical lower bound up.
        # Lower bound for matrix multiplication tensor rank is at least max(n*m, m*p, p*n)
        # For 3,3,5: max(9, 15, 15) = 15
        
        lower_bound = max(n*m, m*p, p*n)
        
        if target_rank < lower_bound:
            return None
        
        # Try to find a decomposition by solving a system of equations
        # For each rank r, we have r unknown columns in U, V, W
        # The constraint is that for all (a,b,c):
        #   sum_{r'} U[a,r']*V[b,r']*W[c,r'] = T[a,b,c]
        
        # This is a system of nonlinear equations. We can use a greedy approach:
        # 1. Pick a nonzero entry of T
        # 2. Assign it to a rank-1 term
        # 3. Subtract and continue
        
        # Actually, for exact integer decompositions, let's try a different approach.
        # We'll use the fact that for matrix multiplication, there are known decompositions.
        
        # For now, let's try to use optimization with better initialization.
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize using a structured approach
            if attempt == 0:
                # Use the first target_rank terms of schoolbook
                U_init = U_sch[:, :target_rank].copy()
                V_init = V_sch[:, :target_rank].copy()
                W_init = W_sch[:, :target_rank].copy()
            elif attempt == 1:
                # Random initialization with small values
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
            else:
                # Hybrid: some schoolbook, some random
                half = target_rank // 2
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                if half > 0:
                    U_init[:, :half] = U_sch[:, :half]
                    V_init[:, :half] = V_sch[:, :half]
                    W_init[:, :half] = W_sch[:, :half]
                if target_rank - half > 0:
                    U_init[:, half:] = np.random.randn(n*m, target_rank - half) * 0.1
                    V_init[:, half:] = np.random.randn(m*p, target_rank - half) * 0.1
                    W_init[:, half:] = np.random.randn(p*n, target_rank - half) * 0.1
            
            # Define the objective function
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute sum of squared errors
                # T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                approx = np.zeros_like(T)
                for r in range(target_rank):
                    approx += np.einsum('a,b,c->abc', U_r[:,r], V_r[:,r], W_r[:,r])
                
                return np.sum((T - approx)**2)
            
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, params_init, method='L-BFGS-B',
                                 options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12})
                
                if result.fun < 1e-8:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    
                    # Try rounding to integers only
                    U_int = np.round(U_r)
                    V_int = np.round(V_r)
                    W_int = np.round(W_r)
                    if verify_decomposition(U_int, V_int, W_int):
                        return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                        
            except Exception:
                pass
        
        return None
    
    # Search for lower rank decompositions
    # Start from a reasonable lower bound
    # For 3x3 * 3x5, known results suggest rank around 15-20 might be achievable
    # Let's try from 15 up to 30
    
    for target_rank in range(max(n*m, m*p, p*n), min(schoolbook_rank, 35)):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, max_attempts=2)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue searching for even lower rank
            continue
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
