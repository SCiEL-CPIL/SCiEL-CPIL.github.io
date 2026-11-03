import numpy as np
import scipy.optimize as opt

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Precompute indices for the non-zero entries of the tensor
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # We need to compute the full tensor T_full = sum_r U[a,r] V[b,r] W[c,r]
    # and compare with T_target which is 1 at those indices and 0 elsewhere.
    
    # Target tensor is sparse, but we can represent the loss efficiently.
    # Loss = || T_full - T_target ||_F^2
    # T_full[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # To speed up gradient computation, we can precompute some contractions.
    # But for R ~ 60 and dimensions 16, 20, 20, direct einsum is fast enough.
    
    def loss_and_grad(params):
        R = params.size // (nm + mp + pn)
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        # Compute full tensor
        # T_full shape (nm, mp, pn)
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor
        # Instead of creating a full (16, 20, 20) array every time, 
        # we can compute the residual only where T_target is non-zero?
        # No, the loss includes zeros too. 
        # T_target is zero everywhere except 80 entries.
        # So T_full - T_target is T_full everywhere, except at 80 entries where it is T_full - 1.
        
        # Let's compute the squared error:
        # Sum_{a,b,c} (T_full[a,b,c] - T_target[a,b,c])^2
        # = Sum_{all} T_full^2 - 2 * Sum_{nonzero} T_full + 80 * 1^2
        # This is faster!
        
        # Sum of squares of T_full
        sum_sq = np.sum(T_full**2)
        
        # Sum of T_full at non-zero positions
        # We need the indices of the non-zero entries in T_target
        # a = i*m+j, b = j*p+k, c = k*n+i
        # We can precompute these indices outside.
        
        prod_nonzero = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :], axis=1)
        sum_nonzero = np.sum(prod_nonzero)
        
        loss = sum_sq - 2 * sum_nonzero + 80
        
        # Gradients
        # dL/dU[a,r] = 2 * ( Sum_{b,c} T_full[a,b,c] V[b,r] W[c,r] - (if a,b,c in nonzero) V[b,r] W[c,r] )
        # Let M = T_full. 
        # dL/dU = 2 * ( M * V * W - Target * V * W )
        # M * V * W is einsum('abc,br,cr->ar', T_full, V, W)
        # Target * V * W is sparse.
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', T_full, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', T_full, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', T_full, U, V)
        
        # Subtract the target contribution
        # For U: sum over b,c where (a,b,c) is in target.
        # The target is 1 at (idx_a[i], idx_b[i], idx_c[i]).
        # So for each i, we subtract V[idx_b[i], r] * W[idx_c[i], r] from grad_U[idx_a[i], r].
        # We can do this efficiently by accumulating.
        
        # Create a zero array for the target gradients
        # But we can just do a loop or use np.add.at if indices are not unique.
        # Indices are unique for each (i,j,k).
        
        # Vectorized update for gradients:
        # grad_U[idx_a, :] -= V[idx_b, :] * W[idx_c, :]
        # This works if idx_a has no duplicates. 
        # a = i*m+j. i in 0..3, j in 0..3. Unique.
        # So we can do:
        grad_U[idx_a, :] -= V[idx_b, :] * W[idx_c, :]
        grad_V[idx_b, :] -= U[idx_a, :] * W[idx_c, :]
        grad_W[idx_c, :] -= U[idx_a, :] * V[idx_b, :]
        
        # Flatten gradients
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    # Precompute indices
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)

    # Try ranks
    for rank in range(60, 75):
        np.random.seed(rank) # Deterministic but different for each rank
        
        # Multiple restarts
        for restart in range(5):
            np.random.seed(rank * 100 + restart)
            
            # Initialize with small random values
            U0 = np.random.randn(nm, rank) * 0.1
            V0 = np.random.randn(mp, rank) * 0.1
            W0 = np.random.randn(pn, rank) * 0.1
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use L-BFGS-B
            try:
                res = opt.minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.success:
                x_opt = res.x
                U_opt = x_opt[:nm*rank].reshape(nm, rank)
                V_opt = x_opt[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_opt = x_opt[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Snap to half-integers
                U_half = np.round(U_opt * 2) / 2
                V_half = np.round(V_opt * 2) / 2
                W_half = np.round(W_opt * 2) / 2
                
                # Verify
                T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                
                # Check if it matches target
                # Target is 1 at idx, 0 elsewhere.
                # We check max error.
                # We only need to check the non-zero entries and a few random zero entries?
                # Better: check the full tensor error.
                # But we can be smarter.
                # Error = || T_half - T_target ||_F
                # We know T_target.
                
                # Compute error efficiently
                # T_half is (16, 20, 20)
                # T_target is 1 at 80 spots, 0 elsewhere.
                
                # Sum of squares of T_half
                sum_sq = np.sum(T_half**2)
                prod_nonzero = np.sum(U_half[idx_a, :] * V_half[idx_b, :] * W_half[idx_c, :], axis=1)
                sum_nonzero = np.sum(prod_nonzero)
                
                # Error^2 = sum_sq - 2*sum_nonzero + 80
                error_sq = sum_sq - 2 * sum_nonzero + 80
                
                if error_sq < 1e-6:
                    # Found a valid decomposition
                    return {
                        "U": U_half.tolist(),
                        "V": V_half.tolist(),
                        "W": W_half.tolist()
                    }
    
    # Fallback to schoolbook if no low rank found
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
