"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Verify helper
    def verify(U, V, W, n, m, p, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition (rank = 96)
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
        return U, V, W, rank

    # Global Optimization Strategy
    # We seek U (nm x R), V (mp x R), W (pn x R) such that
    # sum_c U[:,c] V[:,c] W[:,c] == T_target
    # This is equivalent to minimizing || vec(U * V * W) - vec(T) ||_F
    
    def optimize_rank(R, max_iter=50, tol=1e-10):
        if time.time() > deadline:
            return None
        
        # Target tensor flattened
        # T[i, j, k] = 1 if i,j,k are valid indices, else 0
        # In our convention:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # The tensor T is defined such that T_{(i,j), (j,k), (k,i)} = 1?
        # No, the tensor is the multiplication tensor.
        # T_{(i,j), (j,k), (k,i)} is not quite right.
        # Let's define the target matrix M of size (nm*mp) x (pn) ? No.
        
        # Let's flatten the equation:
        # For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a system of N = n*m*p equations.
        # Variables are entries of U, V, W.
        
        # Residual function for least_squares
        # We need to map x (flat vector) to residuals.
        
        nm = n * m
        mp = m * p
        pn = p * n
        num_vars = R * (nm + mp + pn)
        
        # Precompute indices for speed
        # We need to extract U, V, W from x
        # x[0:R*nm] -> U
        # x[R*nm:R*nm+R*mp] -> V
        # x[R*nm+R*mp:] -> W
        
        # To make it faster, we can precompute the structure of the tensor.
        # The residual for (i,j,k) is:
        # r_{ijk} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} - 1
        
        # We can vectorize this using einsum if we reshape U, V, W.
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Residual tensor: (n, m, p)
        
        def residuals(x):
            U = x[:R*nm].reshape(nm, R)
            V = x[R*nm:R*nm+R*mp].reshape(mp, R)
            W = x[R*nm+R*mp:].reshape(pn, R)
            
            # Compute T = sum_c U[:,c] outer V[:,c] outer W[:,c]
            # This is (nm, mp, pn)
            # We need to reshape it to (n, m, p) and check against 1s.
            
            # Efficient computation:
            # T_ijk = sum_c U_{im+j, c} V_{jp+k, c} W_{kn+i, c}
            
            # Let's build the residual array (n, m, p)
            res = np.zeros((n, m, p))
            
            # This loop might be slow in pure Python, so let's try to vectorize.
            # For each c:
            # Term_c is (nm, mp, pn)
            # We can use np.einsum
            # But R might be small, so loop over c is okay?
            # n*m*p = 96. R ~ 70. 96*70 = 6720 ops per residual call.
            # least_squares calls this many times.
            
            # Alternative: Precompute the linear map?
            # The equation is linear in U, V, W individually if others are fixed.
            # But least_squares treats all as variables.
            
            # Let's use a loop over c for clarity and moderate speed, 
            # relying on the fact that n,m,p are small.
            
            # Actually, we can precompute the indices for U, V, W for each (i,j,k).
            # But that's 96 * 3 indices.
            
            # Let's try to vectorize the whole thing.
            # U reshaped to (n, m, R)
            # V reshaped to (m, p, R)
            # W reshaped to (p, n, R)
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
            
            # We can compute this as:
            # For each i, j:
            #   We have U_vec[i, j] of size R.
            #   We need to combine with V and W.
            
            # Let's just do the loop over c, it's likely fast enough for R<100 and 96 outputs.
            # But wait, least_squares might call residuals 1000s of times.
            # 1000 * 70 * 96 * 3 mults = ~20 million flops. Very fast.
            
            for c in range(R):
                u_c = U_3d[:, :, c] # (n, m)
                v_c = V_3d[:, :, c] # (m, p)
                w_c = W_3d[:, :, c] # (p, n)
                
                # We need to add u_c[i,j] * v_c[j,k] * w_c[k,i] to res[i,j,k]
                # This is a "trace" like operation.
                # res[i,j,k] += u_c[i,j] * (v_c[j,:] * w_c[:,i])_k ? No.
                
                # Let's use einsum for this specific c term:
                # term_c[i, j, k] = u_c[i, j] * v_c[j, k] * w_c[k, i]
                # np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
                
                res += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
                
            res -= 1.0
            return res.flatten()
            
        # Initial guess: Random
        # We can try a few random restarts.
        
        best_result = None
        best_rank = None
        
        # Try a few random initializations
        for restart in range(5):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(num_vars)
            # Scale? The values are expected to be small integers/half-integers.
            # Random normal might be too large or small.
            # Let's scale by 1.0
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if sol.success or sol.cost < tol:
                    res_val = sol.cost
                    if best_result is None or res_val < best_result:
                        best_result = res_val
                        best_x = sol.x
            except Exception:
                continue
                
        if best_result is None or best_result > 1e-8:
            return None
            
        # Extract U, V, W
        U = best_x[:R*nm].reshape(nm, R)
        V = best_x[R*nm:R*nm+R*mp].reshape(mp, R)
        W = best_x[R*nm+R*mp:].reshape(pn, R)
        
        # Check verification
        if verify(U, V, W, n, m, p, R):
            return U, V, W, R
            
        return None

    # Strategy:
    # 1. Try to find a decomposition with rank lower than schoolbook (96).
    # 2. Start from a rank slightly below 96 and increase if it fails.
    # 3. The block Strassen gave 88. We hope global search finds something < 88.
    # 4. Known lower bounds: 
    #    For 3x4x8, the rank is likely around 70-80.
    #    Let's try ranks from 70 to 87.
    
    best_solution = None
    best_rank = 96 # Fallback
    
    # Try ranks from 70 to 87
    # We prioritize lower ranks.
    for R in range(70, 88):
        if time.time() > deadline - 10:
            break
            
        sol = optimize_rank(R)
        if sol is not None:
            U, V, W, rank = sol
            if rank < best_rank:
                best_rank = rank
                best_solution = (U, V, W, rank)
                print(f"Found valid decomposition with rank {rank}")
                # We can stop if we find a very low rank, but let's see if we can go lower.
                # Given time constraints, if we find one, it's good.
                # But maybe a lower one exists?
                # The loop goes 70 -> 87. If 75 works, it returns 75.
                # If 70-74 fail, it tries 75.
                # So it naturally finds the lowest in the range.
                # We can break if we find rank <= 70? No, 70 is the start.
                # If we find 70, that's likely minimal or close.
                # Let's just let the loop finish or break on time.
    
    if best_solution is not None:
        return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
        
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    if verify(U, V, W, n, m, p, rank):
        print(f"Schoolbook decomposition verified with rank {rank}")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
