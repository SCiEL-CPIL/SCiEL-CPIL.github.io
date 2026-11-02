import numpy as np
import time
import random

def _build_tensor(n, m, p):
    """
    Build the dense tensor T of shape (n*m, m*p, p*n) 
    where T[a, b, c] = 1 if the indices correspond to a multiplication term.
    a = i*m + j
    b = j*p + k
    c = k*n + i  (Note: task says W[k*n+i, r], so c index is k*n + i)
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """
    Verify if U, V, W exactly reconstruct T.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    # Reconstruct T from U, V, W
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # This is a batched matrix multiplication / einsum
    # np.einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    # Check if T_recon matches T
    if not np.allclose(T_recon, T, atol=tol, rtol=0):
        return False
    return True

def _round_to_half_integers(mat):
    """
    Round a matrix to the nearest integer or half-integer.
    """
    # Multiply by 2, round to integer, divide by 2
    return np.round(mat * 2) / 2.0

def _try_rank_r(n, m, p, r, T, time_limit):
    """
    Try to find a decomposition of rank r.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize U, V, W
        # Use random initialization, scaled appropriately
        # The entries of T are 0 or 1. 
        # If we use random values in [-1, 1], the sum of r terms might be small.
        # Better to initialize with values that can sum to 1.
        # Let's try random values from {-1, 0, 1} or similar, or just normal.
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to avoid huge/small values
        # Not strictly necessary for ALS, but helps convergence.
        
        # ALS iterations
        for it in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Update U: U[:, s] = argmin_U || T - sum_s u_s v_s^T w_s^T ||
            # For fixed V, W, the term for column s is u_s * (V[:, s] * W[:, s] flattened?)
            # T is 3D. Let's reshape T to (n*m, m*p*p*n) ? No.
            # T[a, b, c] = sum_s U[a, s] * (V[b, s] * W[c, s])
            # For a fixed s, let M_s[b, c] = V[b, s] * W[c, s].
            # Then T[a, :, :] = sum_s U[a, s] M_s.
            # This is a matrix equation for each s? No, U[a, s] is a scalar.
            # We can solve for U[:, s] given V, W.
            # T_flat[a, b, c] -> T_2d[a, b*c]
            # M_s_flat[b, c] -> M_s_2d[b, c]
            # T_2d = U @ M_s_2d ? No.
            # T[a, b, c] = sum_s U[a, s] V[b, s] W[c, s]
            # Let's fix s. The contribution to T from term s is U[:, s] * (V[:, s] * W[:, s])?
            # No, it's an outer product of U[:, s], V[:, s], W[:, s].
            # We can vectorize the last two dimensions.
            # Let T_vec have shape (n*m, m*p * p*n).
            # Let V_W_s have shape (m*p, p*n) where (V_W_s)[b, c] = V[b, s] * W[c, s].
            # Then T_vec[:, :] = sum_s U[:, s] * V_W_s ?
            # No, T_vec[a, b, c] = sum_s U[a, s] V[b, s] W[c, s].
            # This is not a standard matrix product.
            # However, for a fixed s, if we know V[:, s] and W[:, s], we want to find U[:, s] such that
            # U[:, s] approximates the residual / (V[:, s] * W[:, s])?
            # Actually, ALS for CP decomposition:
            # To update U, we fix V and W.
            # The term involving U[a, s] is U[a, s] * sum_{b,c} T[a,b,c] V[b,s] W[c,s] / sum_{b,c} (V[b,s] W[c,s])^2 ?
            # Yes, this is the standard ALS update.
            
            # Precompute V_W_products for efficiency?
            # For each s, compute M_s = V[:, s] * W[:, s] (element-wise product, shape m*p x p*n? No, V is m*p, W is p*n. 
            # Wait, V index is b, W index is c. 
            # The "matrix" M_s has entries M_s[b, c] = V[b, s] * W[c, s].
            # We need to solve for U[:, s].
            # Let R = T - sum_{s' != s} U[:, s'] * M_{s'}.
            # Then U[:, s] = R / M_s (element-wise division, but with least squares if M_s has zeros).
            # Since M_s[b, c] = V[b, s] W[c, s], it can be zero.
            
            # Efficient update:
            # Let's compute the contribution of all other terms.
            # This is O(r * n*m * m*p * p*n) which is large.
            # n*m=6, m*p=21, p*n=14. r~40.
            # 40 * 6 * 21 * 14 = 70,560 operations per update step.
            # 3 updates per iteration (U, V, W).
            # 50 iterations * 3 * 70k = 10M operations. Very fast.
            
            # Let's implement the update for U.
            # We need to compute the "effective" right-hand side for U.
            # Let's use the formula:
            # U[:, s] = (T * (V[:, s] * W[:, s])) / ((V[:, s] * W[:, s]) ** 2)
            # where * is element-wise multiplication over the b,c dimensions, summed over b,c?
            # No, the least squares solution for x in A x = b is x = A^T b / A^T A.
            # Here, for each a, we have a system.
            # Actually, the update is:
            # U[a, s] = sum_{b,c} T[a,b,c] V[b,s] W[c,s] / sum_{b,c} (V[b,s] W[c,s])^2
            
            # Let's compute this.
            # V_W = V * W[:, None, :] ? No, V is (m*p, r), W is (p*n, r).
            # We need V[b, s] * W[c, s].
            # Let's create an array VW of shape (m*p, p*n, r) where VW[b, c, s] = V[b, s] * W[c, s].
            # This is memory heavy? 21 * 14 * 40 = 11760 floats. Fine.
            
            VW = np.einsum('bs,cs->bcs', V, W)
            
            # Numerator for U: sum_{b,c} T[a,b,c] VW[b,c,s]
            # This is a contraction over b, c.
            # Num_U = np.einsum('abc,bcs->as', T, VW)
            # Denom_U = np.einsum('bcs,bcs->s', VW, VW)
            
            Num_U = np.einsum('abc,bcs->as', T, VW)
            Denom_U = np.einsum('bcs,bcs->s', VW, VW)
            
            # Avoid division by zero
            Denom_U[Denom_U == 0] = 1e-10
            U_new = Num_U / Denom_U
            
            # Update V
            # VW2 = U * W? No.
            # V[b, s] = sum_{a,c} T[a,b,c] U[a,s] W[c,s] / sum_{a,c} (U[a,s] W[c,s])^2
            # Let UW = np.einsum('as,cs->acs', U, W) # Shape (n*m, p*n, r)
            UW = np.einsum('as,cs->acs', U, W)
            Num_V = np.einsum('abc,acs->bs', T, UW)
            Denom_V = np.einsum('acs,acs->s', UW, UW)
            Denom_V[Denom_V == 0] = 1e-10
            V_new = Num_V / Denom_V
            
            # Update W
            # UV = np.einsum('as,bs->abs', U, V) # Shape (n*m, m*p, r)
            UV = np.einsum('as,bs->abs', U, V)
            Num_W = np.einsum('abc,abs->cs', T, UV)
            Denom_W = np.einsum('abs,abs->s', UV, UV)
            Denom_W[Denom_W == 0] = 1e-10
            W_new = Num_W / Denom_W
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence / residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_recon)
            if residual < 1e-8:
                # Try to snap to half-integers
                U_rounded = _round_to_half_integers(U)
                V_rounded = _round_to_half_integers(V)
                W_rounded = _round_to_half_integers(W)
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                    # Convert to lists for return
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                else:
                    # Try rounding to integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-6):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
                # If rounding fails, we can still return the float solution if it's "exact enough"?
                # The task requires integer/half-integer entries.
                # So we must find one that snaps.
                # If it doesn't snap, this restart failed.
                pass

    return None

def _schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 2, 3, 7
    T = _build_tensor(n, m, p)
    
    # Deadline guard
    start_time = time.time()
    max_time = 110.0 # Leave 10s margin for 120s limit
    
    # Try ranks from 41 down to 30
    # If we find a lower rank, return it.
    # If not, return schoolbook.
    
    best_solution = None
    best_rank = n * m * p
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to improve
    for r in range(best_rank - 1, 20, -1):
        if time.time() - start_time > max_time:
            break
            
        # Allocate time for this rank
        # If we have 50 ranks to try, give each ~2 seconds?
        # Or just use the global deadline.
        sol = _try_rank_r(n, m, p, r, T, time.time() - start_time + 5.0) # Allow 5s per rank?
        # Better: pass remaining time.
        
        # Let's restructure: _try_rank_r should take a time limit.
        # I'll modify _try_rank_r to take a deadline.
        
        # Actually, let's just call it with a small time slice.
        # If r=41, try for 10s. If not found, r=40, try for 10s.
        # Total time ~ 110s.
        
        # Let's use a fixed time slice per rank to ensure we try many ranks.
        time_slice = 2.0
        sol = _try_rank_r_with_deadline(n, m, p, r, T, time.time() + time_slice)
        
        if sol is not None:
            U, V, W = sol
            best_solution = (U, V, W)
            best_rank = r
            # We can continue to try for even lower ranks
            # But if we found one, we should try to find a lower one.
            # However, the probability of finding a lower one decreases.
            # Let's continue the loop.
            
    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

def _try_rank_r_with_deadline(n, m, p, r, T, deadline):
    """
    Try to find a decomposition of rank r before deadline.
    """
    start_time = time.time()
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # ALS iterations
        for it in range(20): # Fewer iterations per restart to allow more restarts
            if time.time() > deadline:
                break
                
            # Update U
            VW = np.einsum('bs,cs->bcs', V, W)
            Num_U = np.einsum('abc,bcs->as', T, VW)
            Denom_U = np.einsum('bcs,bcs->s', VW, VW)
            Denom_U[Denom_U == 0] = 1e-10
            U = Num_U / Denom_U
            
            # Update V
            UW = np.einsum('as,cs->acs', U, W)
            Num_V = np.einsum('abc,acs->bs', T, UW)
            Denom_V = np.einsum('acs,acs->s', UW, UW)
            Denom_V[Denom_V == 0] = 1e-10
            V = Num_V / Denom_V
            
            # Update W
            UV = np.einsum('as,bs->abs', U, V)
            Num_W = np.einsum('abc,abs->cs', T, UV)
            Denom_W = np.einsum('abs,abs->s', UV, UV)
            Denom_W[Denom_W == 0] = 1e-10
            W = Num_W / Denom_W
            
            # Check residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_recon)
            if residual < 1e-8:
                U_rounded = _round_to_half_integers(U)
                V_rounded = _round_to_half_integers(V)
                W_rounded = _round_to_half_integers(W)
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-6):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
    return None
