"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build the rank-36 Strassen decomposition as starting point
    # A: 2x4 (i in 0..1, j in 0..3)
    # B: 4x5 (j in 0..3, k in 0..4)
    # C: 2x5 (i in 0..1, k in 0..4)
    # W index: k*n + i = k*2 + i
    
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        """Add 7 Strassen terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        terms = [
            # m1 = (a00+a11)*(b00+b11) -> c00 += 1, c11 += 1
            ({(i0,j0):1, (i0+1,j0+1):1}, {(j0,k0):1, (j0+1,k0+1):1}, {(i0,k0):1, (i0+1,k0+1):1}),
            # m2 = (a10+a11)*b00 -> c10 += 1, c11 -= 1
            ({(i0+1,j0):1, (i0+1,j0+1):1}, {(j0,k0):1}, {(i0+1,k0):1, (i0+1,k0+1):-1}),
            # m3 = a00*(b01-b11) -> c01 += 1, c11 += 1
            ({(i0,j0):1}, {(j0,k0+1):1, (j0+1,k0+1):-1}, {(i0,k0+1):1, (i0+1,k0+1):1}),
            # m4 = a11*(b10-b00) -> c00 += 1, c10 += 1
            ({(i0+1,j0+1):1}, {(j0+1,k0):1, (j0,k0):-1}, {(i0,k0):1, (i0+1,k0):1}),
            # m5 = (a00+a01)*b11 -> c00 -= 1, c01 += 1
            ({(i0,j0):1, (i0,j0+1):1}, {(j0+1,k0+1):1}, {(i0,k0):-1, (i0,k0+1):1}),
            # m6 = (a10-a00)*(b00+b01) -> c11 += 1
            ({(i0+1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k0+1):1}, {(i0+1,k0+1):1}),
            # m7 = (a01-a11)*(b10+b11) -> c00 += 1
            ({(i0,j0+1):1, (i0+1,j0+1):-1}, {(j0+1,k0):1, (j0+1,k0+1):1}, {(i0,k0):1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val*sign for k, val in u.items()},
                {k: val*sign for k, val in v.items()},
                {k: val*sign for k, val in w.items()}
            ))
    
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        """Add 4 schoolbook terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        for di in range(2):
            for dj in range(2):
                rank_terms.append((
                    {(i0+di, j0+dj): sign},
                    {(j0+dj, k0): sign},
                    {(i0+di, k0): sign}
                ))
    
    # Build rank-36: split middle dim 4 as 2+2
    # For each A 2x2 block (cols 0-1 and cols 2-3), multiply with B 2x5 block
    # B 2x5 split as 2x2 + 2x2 + 2x1
    
    # A block 1: rows 0-1, cols 0-1
    add_strassen_2x2x2(0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 0, 2)   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_schoolbook_2x2x1(0, 0, 4) # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    
    # A block 2: rows 0-1, cols 2-3
    add_strassen_2x2x2(0, 2, 0)   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 2, 2)   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    add_schoolbook_2x2x1(0, 2, 4) # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    
    R0 = len(rank_terms)  # should be 36
    
    # Convert to matrices
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
    
    # Verify correctness of rank-36 decomposition
    # The multiplication tensor T[i,j,k] = delta_{ik} for the relevant indices
    # Actually: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor: T[i,j,k] where the decomposition is sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if this is the "1" entry
    # Actually the tensor for matrix mult is: T[i,j,k] = 1 (it's just the identity in j contracted)
    # Wait, the tensor is: for each (i,j,k), the coefficient is 1 (since (AB)_{ik} = sum_j A_{ij}B_{jk})
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        # Check all (i,j,k) tuples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U0, V0, W0):
        # Fallback: return schoolbook rank 40
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((p*n, n*m*p))
        for r, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U_s[i*m+j, r] = 1
            V_s[j*p+k, r] = 1
            W_s[k*n+i, r] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Now try to reduce rank using numerical optimization
    # Strategy: perturb the rank-36 solution and run ALS to minimize a "sparsity" objective
    # that encourages some columns to become zero (i.e., rank reduction)
    
    best_rank = R0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try multiple random perturbations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed version
        R = R0
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        # Add small random perturbations
        eps = 0.01 * (attempt + 1)
        U += np.random.randn(n*m, R) * eps
        V += np.random.randn(m*p, R) * eps
        W += np.random.randn(p*n, R) * eps
        
        # Run alternating least squares with a sparsity-promoting objective
        # Objective: minimize ||T - sum_r U_r V_r W_r||^2 + lambda * sum_r ||col_r||_1
        # where col_r is the combined column
        
        lambda_val = 0.1
        max_iters = 50
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each column r, minimize over U[:,r]
            # sum_r U[i,j,r] V[j,k,r] W[i,k,r] should equal 1
            # For fixed V, W, the optimal U[:,r] satisfies:
            # (V * W^T)_{(i,j),(j,k)} ... this is complex, let's use gradient descent instead
            
            # Use a simpler approach: gradient descent on the loss
            # L = sum_{i,j,k} (1 - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])^2
            #     + lambda * sum_r (||U[:,r]||^2 + ||V[:,r]||^2 + ||W[:,r]||^2)
            
            lr = 0.001
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute current value
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = 1.0 - val
                        
                        # Gradient w.r.t. U[i*m+j, r]
                        grad_U[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                        # Gradient w.r.t. V[j*p+k, r]
                        grad_V[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                        # Gradient w.r.t. W[k*n+i, r]
                        grad_W[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
            
            # Add regularization to encourage sparsity
            grad_U += 2 * lambda_val * U
            grad_V += 2 * lambda_val * V
            grad_W += 2 * lambda_val * W
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check if any columns have become small (potential rank reduction)
            col_norms_U = np.linalg.norm(U, axis=0)
            col_norms_V = np.linalg.norm(V, axis=0)
            col_norms_W = np.linalg.norm(W, axis=0)
            min_col_norm = np.min(np.minimum(np.minimum(col_norms_U, col_norms_V), col_norms_W))
            
            if min_col_norm < 0.01:
                # Try to remove the smallest column
                break
        
        # After optimization, check if we can remove columns
        col_norms_U = np.linalg.norm(U, axis=0)
        col_norms_V = np.linalg.norm(V, axis=0)
        col_norms_W = np.linalg.norm(W, axis=0)
        combined_norms = np.minimum(np.minimum(col_norms_U, col_norms_V), col_norms_W)
        
        # Sort columns by combined norm
        order = np.argsort(combined_norms)
        
        # Try removing columns from smallest to largest
        for num_remove in range(1, 6):  # Try removing up to 5 columns
            if time.time() > deadline - 5:
                break
            
            keep_cols = order[num_remove:]
            if len(keep_cols) < 10:
                break
            
            U_test = U[:, keep_cols]
            V_test = V[:, keep_cols]
            W_test = W[:, keep_cols]
            
            if verify(U_test, V_test, W_test, tol=1e-6):
                new_rank = len(keep_cols)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U = U_test.copy()
                    best_V = V_test.copy()
                    best_W = W_test.copy()
                break  # Found a valid reduced rank, stop trying to remove more
    
    # Convert best solution to integer/half-integer if possible
    # Round to nearest half-integer
    def round_to_half(x):
        return np.round(x * 2) / 2.0
    
    # Try to round the best solution
    U_rounded = round_to_half(best_U)
    V_rounded = round_to_half(best_V)
    W_rounded = round_to_half(best_W)
    
    if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        # Fallback to original rank-36
        best_U, best_V, best_W = U0, V0, W0
        best_rank = R0
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
