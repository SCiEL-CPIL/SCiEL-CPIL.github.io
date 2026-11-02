import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = 145.0

    # Dimensions
    dim_U = n * m  # 6
    dim_V = m * p  # 18
    dim_W = p * n  # 12

    def verify(U, V, W, atol=1e-8):
        r = U.shape[1]
        # Check all combinations
        # To speed up, we can check a subset first, then all
        # But 2*3*3*6*2*6 = 1296 checks. Each sum is r terms.
        # If r=20, 25920 ops. Very fast.
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b_idx in range(m):
                        for bp in range(m):
                            for c in range(p):
                                expected = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                s = 0.0
                                for l in range(r):
                                    s += W[k * n + i, l] * U[a * m + b_idx, l] * V[bp * p + c, l]
                                if abs(s - expected) > atol:
                                    return False
        return True

    # Schoolbook baseline (rank 36)
    r_sb = n * m * p
    U_sb = np.zeros((dim_U, r_sb))
    V_sb = np.zeros((dim_V, r_sb))
    W_sb = np.zeros((dim_W, r_sb))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_sb[ii * m + jj, col] = 1
                V_sb[jj * p + kk, col] = 1
                W_sb[kk * n + ii, col] = 1
                col += 1

    best_rank = r_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def try_rank(r):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False
        if time.time() - start_time > deadline - 10:
            return False
            
        # Build the linear system matrix A and vector b for the CP decomposition
        # Equation: sum_l U[a,b,l] V[bp,c,l] W[i,k,l] = delta(a,i) delta(b,bp) delta(c,k)
        # We can view this as: for fixed U, V, solve for W.
        # Or use ALS.
        
        # To make ALS fast, we precompute the "flattened" tensors.
        # Let's use a simple ALS with random restarts.
        
        def als_decompose(r, max_iters=100, tol=1e-10):
            # Initialize
            U = np.random.randn(dim_U, r) * 0.1
            V = np.random.randn(dim_V, r) * 0.1
            W = np.random.randn(dim_W, r) * 0.1
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() - start_time > deadline - 5:
                    return None, float('inf')
                
                # Solve for U
                # For each l, we want U[:, l] to minimize error.
                # This is equivalent to solving a linear system for each column of U?
                # No, the error is sum_{i,j,bp,c} (T_{i,j,bp,c} - sum_l ...)^2
                # This is a quadratic in U[:, l] for fixed V, W.
                
                # Efficient ALS:
                # Residual R = T - sum_{l' != l} U_{l'} V_{l'} W_{l'}
                # We want to find U_l such that <U_l, R_flat> is maximized?
                # Actually, we want to project R onto the span of V_l and W_l.
                
                # Let's compute the "effective" target for U_l.
                # T_{a,b,bp,c,i,k}
                # Term for l: U[a,b,l] V[bp,c,l] W[i,k,l]
                
                # We can compute the matrix M_U of size (dim_U * dim_W) x (dim_U) ? No.
                
                # Let's use the standard ALS update formula.
                # Fix V, W.
                # For each l, we want to update U[:, l].
                # The contribution of l to the tensor is U_l (outer) V_l (outer) W_l.
                # The residual is T - sum_{l'!=l} ...
                # We want to find U_l that best fits the residual in the subspace defined by V_l and W_l.
                
                # This is a bit complex to implement from scratch correctly and fast.
                # Instead, I will use `scipy.optimize.least_squares` on the full flattened system for small ranks?
                # No, too many variables.
                
                # Let's stick to the "solve for W given U, V" approach, but do it for all W at once?
                # The system is linear in W.
                # For each (a,b,bp,c,i,k), we have:
                # sum_l W[i,k,l] * (U[a,b,l] * V[bp,c,l]) = T_{a,b,bp,c,i,k}
                
                # Let X[i,k,a,b,bp,c] be a tensor where X[i,k,a,b,bp,c] = sum_l U[a,b,l] V[bp,c,l] W[i,k,l]
                # We want X to equal T.
                
                # Let's build the matrix A for the linear system in W.
                # Unknowns: W[i,k,l] for i in 0..1, k in 0..5, l in 0..r-1. Total dim_W * r.
                # Equations: for each a,b,bp,c,i,k. Total dim_U * dim_V * dim_W? No.
                # The tensor T has non-zeros only at specific indices.
                # Actually, T is sparse in the "equation" sense?
                # No, T is the identity tensor.
                # The number of equations is dim_U * dim_V * dim_W?
                # No, the sum is over l. The LHS is a function of 6 indices.
                # The RHS is 1 if (a=i, b=bp, c=k), else 0.
                
                # Let's just use `np.linalg.lstsq` on the full system.
                # Number of equations: n * m * m * p * n * p = 2 * 3 * 3 * 6 * 2 * 6 = 1296.
                # Number of unknowns: dim_W * r = 12 * r.
                # If r=21, unknowns = 252.
                # 1296 x 252 matrix. This is very small.
                
                # Build A and b once?
                # A depends on U and V.
                # A[eq, col] = U[a,b,l] * V[bp,c,l]
                # col = (i * p + k) * r + l ? No, W is (p*n, r). Index is (k*n + i, l).
                # So col = (k * n + i) * r + l.
                
                A = np.zeros((1296, dim_W * r))
                b = np.zeros(1296)
                
                eq = 0
                for i in range(n):
                    for k in range(p):
                        for a in range(n):
                            for b_idx in range(m):
                                for bp in range(m):
                                    for c in range(p):
                                        b[eq] = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                        for l in range(r):
                                            col = (k * n + i) * r + l
                                            A[eq, col] = U[a * m + b_idx, l] * V[bp * p + c, l]
                                        eq += 1
                
                W_flat, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(dim_W, r)
                
                if residuals and len(residuals) > 0:
                    err = residuals[0]
                else:
                    err = np.linalg.norm(A @ W_flat - b)
                
                return W, err

        # Random restarts
        restarts = 50
        for r_idx in range(r, best_rank - 1):
            if time.time() - start_time > deadline - 5:
                break
            found = False
            for restart in range(restarts):
                if time.time() - start_time > deadline - 2:
                    break
                
                # Initialize U, V randomly
                U = np.random.randn(dim_U, r_idx) * 0.5
                V = np.random.randn(dim_V, r_idx) * 0.5
                
                # Coordinate descent on U and V
                for cd_iter in range(20):
                    if time.time() - start_time > deadline - 1:
                        break
                    
                    # Update U
                    for l in range(r_idx):
                        U_new = U.copy()
                        U_new[:, l] = np.random.randn(dim_U) * 0.5
                        W_trial, err_trial = als_decompose(r_idx, max_iters=1) # Just solve for W
                        if err_trial < 1e-8:
                            return True
                        if err_trial < 1e-10 * (1 + err_trial): # Relative improvement check?
                            pass # We don't track prev_err easily here, let's just keep best
                        # Actually, let's just keep the best U, V, W found so far
                        # This is getting messy. Let's use a simpler heuristic.
                        # If we can't find a good solution quickly, skip.
                        break
                    
                    # Update V
                    for l in range(r_idx):
                        V_new = V.copy()
                        V_new[:, l] = np.random.randn(dim_V) * 0.5
                        W_trial, err_trial = als_decompose(r_idx, max_iters=1)
                        if err_trial < 1e-8:
                            # Verify
                            if verify(U, V, W_trial):
                                best_rank = r_idx
                                best_U, best_V, best_W = U, V, W_trial
                                return True
                        break

    # Try ranks 21, 20, 19
    for r in [21, 20, 19]:
        if try_rank(r):
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
