import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build Rank 36 Decomposition
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        terms = [
            ({(i0,j0):1, (i0+1,j0+1):1}, {(j0,k0):1, (j0+1,k0+1):1}, {(i0,k0):1, (i0+1,k0+1):1}),
            ({(i0+1,j0):1, (i0+1,j0+1):1}, {(j0,k0):1}, {(i0+1,k0):1, (i0+1,k0+1):-1}),
            ({(i0,j0):1}, {(j0,k0+1):1, (j0+1,k0+1):-1}, {(i0,k0+1):1, (i0+1,k0+1):1}),
            ({(i0+1,j0+1):1}, {(j0+1,k0):1, (j0,k0):-1}, {(i0,k0):1, (i0+1,k0):1}),
            ({(i0,j0):1, (i0,j0+1):1}, {(j0+1,k0+1):1}, {(i0,k0):-1, (i0,k0+1):1}),
            ({(i0+1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k0+1):1}, {(i0+1,k0+1):1}),
            ({(i0,j0+1):1, (i0+1,j0+1):-1}, {(j0+1,k0):1, (j0+1,k0+1):1}, {(i0,k0):1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val*sign for k, val in u.items()},
                {k: val*sign for k, val in v.items()},
                {k: val*sign for k, val in w.items()}
            ))
            
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        for di in range(2):
            for dj in range(2):
                rank_terms.append((
                    {(i0+di, j0+dj): sign},
                    {(j0+dj, k0): sign},
                    {(i0+di, k0): sign}
                ))
                
    # Split m=4 into 2+2, p=5 into 2+2+1 for each m-block
    # Block 1 (j=0,1):
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    add_schoolbook_2x2x1(0, 0, 4)
    # Block 2 (j=2,3):
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 2, 2)
    add_schoolbook_2x2x1(0, 2, 4)
    
    R0 = len(rank_terms) # 36
    
    def terms_to_matrices(terms):
        R = len(terms)
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        for col, (u_d, v_d, w_d) in enumerate(terms):
            for (i,j), c in u_d.items():
                U[i*m + j, col] = c
            for (j,k), c in v_d.items():
                V[j*p + k, col] = c
            for (i,k), c in w_d.items():
                W[k*n + i, col] = c
        return U, V, W
        
    U0, V0, W0 = terms_to_matrices(rank_terms)
    
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        # Tensor check: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        # Vectorized check
        # U is (nm, R), V is (mp, R), W is (pn, R)
        # We need to check all i, j, k
        # This is a bit slow if done naively, but nm=8, mp=20, pn=10.
        # 8*20*10 = 1600 checks. Each check is a dot product of length R.
        # 1600 * 36 operations is small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U0, V0, W0):
        # Fallback to schoolbook 40
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((p*n, n*m*p))
        for r, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U_s[i*m+j, r] = 1
            V_s[j*p+k, r] = 1
            W_s[k*n+i, r] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_rank = R0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Optimization to reduce rank
    # We try to minimize the L2 norm of the coefficients while keeping the tensor exact.
    # If a coefficient vector goes to zero, we can remove that term.
    
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        # Perturb to escape local minima
        R = best_rank
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        
        # Small noise
        noise_scale = 0.1
        U += np.random.randn(n*m, R) * noise_scale
        V += np.random.randn(m*p, R) * noise_scale
        W += np.random.randn(p*n, R) * noise_scale
        
        # Optimize
        # We use a simple gradient descent on the loss:
        # L = ||T - T(U,V,W)||^2 + lambda * (||U||_F^2 + ||V||_F^2 + ||W||_F^2)
        # If lambda is large, it pushes norms down.
        
        lambda_val = 1.0
        lr = 0.01
        max_iters = 1000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute Gradient
            # err_{ijk} = 1 - sum_r U_{ij,r} V_{jk,r} W_{ik,r}
            # dL/dU_{ij,r} = -2 * err_{ijk} * V_{jk,r} * W_{ik,r} + 2 * lambda * U_{ij,r}
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Precompute products for efficiency?
            # R is small (~36).
            # We can loop over i,j,k
            
            # To speed up, we can vectorize the error computation
            # But for simplicity and safety with time, we loop.
            # 1600 tuples * 36 dims = ~60k ops per iteration. 1000 iters = 60M ops.
            # Python is slow. 60M ops might take 10-20s.
            # We should vectorize.
            
            # Vectorized Error Calculation:
            # U: (nm, R), V: (mp, R), W: (pn, R)
            # We need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # This is a tensor contraction.
            # Let's construct the tensor T_calc
            # This is hard to vectorize directly without high memory.
            # However, we can compute the gradient using matrix multiplications.
            
            # For fixed W, the term involving U is:
            # sum_k W[k*n+i, r] * (sum_j U[i*m+j, r] V[j*p+k, r])
            # This looks like a product of U and V weighted by W.
            
            # Let's stick to the loop but optimize it.
            # Actually, 1600 * 36 is very small. The loop overhead in Python is the issue.
            # Let's try to batch the operations.
            
            # We can compute the error for all (i,j,k) at once if we reshape.
            # But the indices are coupled.
            
            # Alternative: Use `np.einsum`?
            # U is (nm, R), V is (mp, R), W is (pn, R)
            # We want T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # Let's just use the loop but break early if loss is small.
            
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        
                        u_vec = U[u_idx, :]
                        v_vec = V[v_idx, :]
                        w_vec = W[w_idx, :]
                        
                        prod = u_vec * v_vec * w_vec
                        val = np.sum(prod)
                        err = 1.0 - val
                        loss += err * err
                        
                        grad_U[u_idx, :] += err * v_vec * w_vec
                        grad_V[v_idx, :] += err * u_vec * w_vec
                        grad_W[w_idx, :] += err * u_vec * v_vec
                        
            # Add regularization
            grad_U += 2 * lambda_val * U
            grad_V += 2 * lambda_val * V
            grad_W += 2 * lambda_val * W
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check for zero columns
            col_norms = np.min(np.min(np.linalg.norm(U, axis=0), np.linalg.norm(V, axis=0)), np.linalg.norm(W, axis=0))
            if np.min(col_norms) < 0.05:
                break
        
        # After optimization, check if we can remove columns
        col_norms_U = np.linalg.norm(U, axis=0)
        col_norms_V = np.linalg.norm(V, axis=0)
        col_norms_W = np.linalg.norm(W, axis=0)
        combined_norms = np.min(np.min(col_norms_U, col_norms_V), col_norms_W)
        
        # Sort by norm
        order = np.argsort(combined_norms)
        
        # Try removing columns
        for num_remove in range(1, 10):
            if time.time() > deadline - 5:
                break
            keep_cols = order[num_remove:]
            if len(keep_cols) < 10:
                break
                
            U_test = U[:, keep_cols]
            V_test = V[:, keep_cols]
            W_test = W[:, keep_cols]
            
            # Verify with tight tolerance
            if verify(U_test, V_test, W_test, tol=1e-6):
                new_rank = len(keep_cols)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U = U_test.copy()
                    best_V = V_test.copy()
                    best_W = W_test.copy()
                break # Found valid reduction, stop trying to remove more in this attempt
    
    # Final step: Round to integers/half-integers and verify
    def round_to_half(x):
        return np.round(x * 2) / 2.0
        
    U_rounded = round_to_half(best_U)
    V_rounded = round_to_half(best_V)
    W_rounded = round_to_half(best_W)
    
    if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    else:
        # Try integer rounding
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        if verify(U_int, V_int, W_int, tol=1e-9):
            best_U, best_V, best_W = U_int, V_int, W_int
        else:
            # If rounding fails, we might have a numerical artifact.
            # We should probably return the original Strassen 36 if the optimized one is not exact.
            # But the task requires exact decomposition.
            # If the numerical solution is not close to integers, it's invalid.
            # We check the original 36.
            if not verify(best_U, best_V, best_W, tol=1e-6):
                 best_U, best_V, best_W = U0, V0, W0
                 best_rank = R0

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
