import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=int)
    V_fb = np.zeros((m*p, rank_fb), dtype=int)
    W_fb = np.zeros((p*n, rank_fb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb

    def loss_func(x, rank):
        # x is flattened U, V, W
        size_u = n*m
        size_v = m*p
        size_w = p*n
        u_end = size_u * rank
        v_end = u_end + size_v * rank
        w_end = v_end + size_w * rank
        
        U = x[:u_end].reshape(size_u, rank)
        V = x[u_end:v_end].reshape(size_v, rank)
        W = x[v_end:w_end].reshape(size_w, rank)
        
        # Compute T_decomp
        # T_decomp[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        diff = T - T_decomp
        return np.sum(diff * diff)

    def snap_and_check(U, V, W, rank):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T_decomp = np.einsum('at,bt,ct->abc', U_s, V_s, W_s, optimize=True)
        if np.allclose(T_decomp, T, atol=1e-9):
            # Convert to list of lists
            return U_s, V_s, W_s
        return None

    def polish_and_snap(U, V, W, rank):
        """High-precision polish: do a final optimization step with tighter tolerance,
        then try snapping to half-integers."""
        if time.time() > deadline - 3:
            return None
        
        # Flatten for optimization
        size_u = n*m
        size_v = m*p
        size_w = p*n
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(loss_func, x0, args=(rank,), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e-8:
                u_end = size_u * rank
                v_end = u_end + size_v * rank
                w_end = v_end + size_w * rank
                
                U_pol = res.x[:u_end].reshape(size_u, rank)
                V_pol = res.x[u_end:v_end].reshape(size_v, rank)
                W_pol = res.x[v_end:w_end].reshape(size_w, rank)
                
                snapped = snap_and_check(U_pol, V_pol, W_pol, rank)
                if snapped:
                    return snapped
        except Exception:
            pass
        
        # Also try direct snap on the input
        return snap_and_check(U, V, W, rank)

    def tt_search(rank, max_tries=30):
        """Tensor Train decomposition search: represent the multiplication tensor
        as a TT with small bond dimensions, then recover exact rational cores."""
        if time.time() > deadline - 5:
            return None
        
        # TT format: T[a,b,c] = sum_{r1,r2} G1[a,r1] * G2[r1,b,r2] * G3[r2,c]
        # For a rank-R decomposition, we can think of it as TT with bond dim R
        # But let's try a more structured approach:
        # Group indices to form a 2D matrix and do SVD-based TT
        
        # Reshape T as a matrix: M[i,j,k, a,b,c] -> but that's 6D
        # Alternative: use the fact that T is sparse (only 36 non-zero entries)
        
        # Try TT with bond dimensions d1, d2 where d1*d2 >= rank
        # For simplicity, try d1=d2=sqrt(rank) rounded
        
        for d1 in range(1, int(np.sqrt(rank)) + 2):
            for d2 in range(1, int(np.sqrt(rank)) + 2):
                if d1 * d2 < rank:
                    continue
                if time.time() > deadline - 5:
                    return None
                
                # TT cores: G1 shape (n*m, 1, d1), G2 shape (d1, m*p, d2), G3 shape (d2, p*n, 1)
                # T[a,b,c] = sum_{r1,r2} G1[a,0,r1] * G2[r1,b,r2] * G3[r2,c,0]
                
                # This is equivalent to a rank-(d1*d2) decomposition where:
                # U[a, (r1,r2)] = G1[a,0,r1] (but this doesn't work directly)
                
                # Let's use a different approach: random TT initialization
                for attempt in range(max_tries // 4):
                    if time.time() > deadline - 5:
                        return None
                    
                    # Initialize TT cores with small random values
                    G1 = np.random.randn(n*m, 1, d1) * 0.5
                    G2 = np.random.randn(d1, m*p, d2) * 0.5
                    G3 = np.random.randn(d2, p*n, 1) * 0.5
                    
                    # Convert to U, V, W for standard rank decomposition
                    # U[a, t] where t = r1*d2 + r2
                    # But we need U[a,t]*V[b,t]*W[c,t] = sum_{r1,r2} G1[a,0,r1]*G2[r1,b,r2]*G3[r2,c,0]
                    # So U[a, r1*d2+r2] = G1[a,0,r1] * (1/sqrt(d2))
                    # V[b, r1*d2+r2] = G2[r1,b,r2] * (1/sqrt(d1*d2))
                    # W[c, r1*d2+r2] = G3[r2,c,0] * (1/sqrt(d1))
                    # This doesn't quite work. Let me think differently.
                    
                    # Actually, the TT representation IS a rank decomposition:
                    # T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
                    # where t indexes (r1, r2) pairs, so rank = d1*d2
                    # U[a, r1*d2+r2] should satisfy:
                    # U[a, r1*d2+r2] * V[b, r1*d2+r2] * W[c, r1*d2+r2] = G1[a,0,r1]*G2[r1,b,r2]*G3[r2,c,0]
                    
                    # One way: U[a, r1*d2+r2] = G1[a,0,r1]
                    #           V[b, r1*d2+r2] = G2[r1,b,r2]  
                    #           W[c, r1*d2+r2] = G3[r2,c,0]
                    # But then the product is G1[a,0,r1]*G2[r1,b,r2]*G3[r2,c,0] which is correct!
                    
                    U_tt = np.zeros((n*m, d1*d2))
                    V_tt = np.zeros((m*p, d1*d2))
                    W_tt = np.zeros((p*n, d1*d2))
                    
                    for r1 in range(d1):
                        for r2 in range(d2):
                            t = r1 * d2 + r2
                            U_tt[:, t] = G1[:, 0, r1]
                            V_tt[:, t] = G2[r1, :, r2]
                            W_tt[:, t] = G3[r2, :, 0]
                    
                    # Now optimize U_tt, V_tt, W_tt
                    x0 = np.concatenate([U_tt.flatten(), V_tt.flatten(), W_tt.flatten()])
                    
                    try:
                        res = minimize(loss_func, x0, args=(d1*d2,), method='L-BFGS-B',
                                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                        
                        if res.fun < 1e-6:
                            u_end = n*m * (d1*d2)
                            v_end = u_end + m*p * (d1*d2)
                            w_end = v_end + p*n * (d1*d2)
                            
                            U_opt = res.x[:u_end].reshape(n*m, d1*d2)
                            V_opt = res.x[u_end:v_end].reshape(m*p, d1*d2)
                            W_opt = res.x[v_end:w_end].reshape(p*n, d1*d2)
                            
                            # Polish and snap
                            polished = polish_and_snap(U_opt, V_opt, W_opt, d1*d2)
                            if polished:
                                return polished
                    except Exception:
                        pass
        
        return None

    # Try ranks from 24 to 35
    # We prioritize lower ranks
    for rank in range(24, 36):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        # First try TT-based search
        if time.time() < deadline - 15:
            tt_result = tt_search(rank, max_tries=20)
            if tt_result:
                U_s, V_s, W_s = tt_result
                best_result = (U_s, V_s, W_s)
                best_rank = rank
                break  # Found solution for this rank, move to next (lower) rank
        
        # Number of restarts for direct optimization
        max_restarts = 40 if rank < 26 else 15
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Use values in {-1, -0.5, 0, 0.5, 1}
            # This helps the optimizer stay in the "right" basin
            init_vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m*rank + m*p*rank + p*n*rank))
            
            # Optimization
            try:
                res = minimize(loss_func, init_vals, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-8:
                    size_u = n*m
                    size_v = m*p
                    size_w = p*n
                    u_end = size_u * rank
                    v_end = u_end + size_v * rank
                    w_end = v_end + size_w * rank
                    
                    U_opt = res.x[:u_end].reshape(size_u, rank)
                    V_opt = res.x[u_end:v_end].reshape(size_v, rank)
                    W_opt = res.x[v_end:w_end].reshape(size_w, rank)
                    
                    # Use polish and snap
                    snapped = polish_and_snap(U_opt, V_opt, W_opt, rank)
                    if snapped:
                        U_s, V_s, W_s = snapped
                        best_result = (U_s, V_s, W_s)
                        best_rank = rank
                        break
            except Exception:
                pass

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
