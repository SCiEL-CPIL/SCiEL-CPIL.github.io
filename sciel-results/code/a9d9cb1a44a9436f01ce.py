import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Flattened sizes
    N_A = n * m  # 9
    N_B = m * p  # 21
    N_C = p * n  # 21 (index k*n + i)
    
    # Construct the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') and i=i', j=j', k=k'
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((N_A, N_B, N_C), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We search for rank R
    # Lower bound: max(n*m, m*p, p*n) = 21
    # Upper bound: n*m*p = 63
    
    best_solution = None
    
    # Define a function to check if a solution is valid and snap to half-integers
    def verify_and_snap(U, V, W, R):
        # Check residual
        # C_calc[c] = sum_r W[c,r] * (U @ V^T)[c, r] ? 
        # Let M = U @ V.T (N_A x R) * ... no.
        # Term is sum_r U[a,r] V[b,r] W[c,r]
        # We can compute for each r: u_r (N_A), v_r (N_B), w_r (N_C)
        # Contribution to T is u_r[:, None] * v_r[None, :] broadcast to N_C? No.
        # It's an outer product of u_r and v_r, but we need to match with w_r.
        # Actually, the sum is:
        # Res[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
        
        # Efficient calculation:
        # Let M = U @ V.T  (N_A x N_B) ... no, U is N_A x R, V is N_B x R.
        # U @ V.T is N_A x N_B.
        # Then we need to multiply this by W for each c?
        # Res[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
        # Let Z = U * V.T (broadcast) -> N_A x N_B x R
        # Res = np.einsum('abr,cr->abc', Z, W)
        
        # This might be memory heavy if R is large, but R <= 63, sizes are small.
        Z = U[:, np.newaxis, :] * V[np.newaxis, :, :] # N_A x N_B x R
        Res = np.einsum('abr,cr->abc', Z, W)
        
        err = np.linalg.norm(Res - T)
        if err > 1e-5:
            return None
            
        # Snap to half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Verify snapped solution
        Z_s = U_snapped[:, np.newaxis, :] * V_snapped[np.newaxis, :, :]
        Res_s = np.einsum('abr,cr->abc', Z_s, W_snapped)
        
        if np.allclose(Res_s, T, atol=1e-9):
            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
        else:
            # Check if integer solution exists? The prompt allows half-integers.
            # If snap fails, maybe it's not a half-integer solution.
            return None

    def run_als(R, init_U=None, init_V=None, init_W=None, max_iter=200, tol=1e-8):
        # Initialize
        if init_U is None:
            U = np.random.randn(N_A, R)
        else:
            U = init_U.copy()
        if init_V is None:
            V = np.random.randn(N_B, R)
        else:
            V = init_V.copy()
        if init_W is None:
            W = np.random.randn(N_C, R)
        else:
            W = init_W.copy()
            
        prev_err = np.inf
        for i in range(max_iter):
            if time.time() > deadline:
                return U, V, W
                
            # Update U
            # Fixed V, W. Minimize sum_{a,b,c} (sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
            # This is linear in U.
            # For each a, sum_{b,c} ( ... )^2
            # It can be shown that U = T @ (V * W.T)^T * (V^T @ (W * V.T) ... )
            # Simpler: 
            # Let A_r = V[:, r] * W[:, r] ? No.
            # The term is U[a,r] * (sum_{b,c} V[b,r] W[c,r] ... )
            # Actually, the gradient w.r.t U[a,r] involves T.
            
            # Efficient ALS step:
            # 1. Update U:
            #    U = np.zeros_like(U)
            #    For each a:
            #       U[a, :] = ...
            
            # Let's use the standard formulation:
            # U = (T_flat @ (V * W.T)) * inv((V.T @ V) * (W.T @ W)?) No.
            
            # Let's derive:
            # Res[a,b,c] = Sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c]
            # Squared error E = Sum_{a,b,c} (Sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c])^2
            # Derivative w.r.t U[a', r']:
            # 2 Sum_{b,c} (Sum_r U[a',r] V[b,r] W[c,r] - T[a',b,c]) V[b,r'] W[c,r']
            # Let Y[a', r'] = Sum_{b,c} T[a',b,c] V[b,r'] W[c,r']
            # Let H[r, r'] = Sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
            # Then H @ U[a', :] = Y[a', :]
            
            # Calculate H (R x R)
            # H[r, r'] = Sum_b V[b,r] V[b,r'] * Sum_c W[c,r] W[c,r']
            # Let GV = V.T @ V (R x R)
            # Let GW = W.T @ W (R x R)
            # H = GV * GW (element-wise)
            
            # Calculate Y (N_A x R)
            # Y[a, r'] = Sum_{b,c} T[a,b,c] V[b,r'] W[c,r']
            # We can compute this as:
            # Let T_a = T[a, :, :] (N_B x N_C)
            # Y[a, r'] = Sum_b (Sum_c T[a,b,c] W[c,r']) V[b,r']
            # Let M_T_W[a, c, r'] = T[a,b,c] W[c,r']? No, sum over b.
            
            # Efficiently:
            # Temp[b, r'] = Sum_c T[a,b,c] W[c,r']? This depends on a.
            
            # Alternative:
            # Y[a, r'] = Sum_b V[b,r'] (Sum_c T[a,b,c] W[c,r'])
            # Let P[a, b, r'] = Sum_c T[a,b,c] W[c,r']
            # This is T (N_A x N_B x N_C) contracted with W (N_C x R) along axis 2.
            # P = np.einsum('abc,cr->abr', T, W)
            # Then Y = P @ V ? No.
            # Y[a, r'] = Sum_b P[a, b, r'] V[b, r']
            # Y = np.einsum('abr,br->ar', P, V)
            
            # Check time
            if time.time() > deadline:
                return U, V, W

            GV = V.T @ V
            GW = W.T @ W
            H = GV * GW
            
            P = np.einsum('abc,cr->abr', T, W)
            Y = np.einsum('abr,br->ar', P, V)
            
            # Solve H @ U.T = Y.T
            try:
                U = np.linalg.solve(H, Y.T).T
            except np.linalg.LinAlgError:
                U = np.linalg.lstsq(H, Y.T, rcond=None)[0].T
                
            # Update V
            # Similar logic
            # H_v[r, r'] = Sum_a U[a,r] U[a,r'] * Sum_c W[c,r] W[c,r']
            # Y_v[b, r'] = Sum_{a,c} T[a,b,c] U[a,r'] W[c,r']
            
            GU = U.T @ U
            GW = W.T @ W
            H_v = GU * GW
            
            Q = np.einsum('abc,ar->bcr', T, U) # Q[b,c,r] = Sum_a T[a,b,c] U[a,r]
            Y_v = np.einsum('bcr,cr->br', Q, W)
            
            try:
                V = np.linalg.solve(H_v, Y_v.T).T
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(H_v, Y_v.T, rcond=None)[0].T
                
            # Update W
            # H_w[r, r'] = Sum_a U[a,r] U[a,r'] * Sum_b V[b,r] V[b,r']
            # Y_w[c, r'] = Sum_{a,b} T[a,b,c] U[a,r'] V[b,r']
            
            GU = U.T @ U
            GV = V.T @ V
            H_w = GU * GV
            
            R_mat = np.einsum('abc,ar,br->cr', T, U, V) # R_mat[c,r] = Sum_{a,b} T[a,b,c] U[a,r] V[b,r]
            # Y_w is basically R_mat.T ?
            # Y_w[c, r'] = Sum_{a,b} T[a,b,c] U[a,r'] V[b,r']
            # Yes, Y_w = R_mat.T
            
            try:
                W = np.linalg.solve(H_w, R_mat.T).T
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(H_w, R_mat.T, rcond=None)[0].T
                
            # Calculate error
            Z = U[:, np.newaxis, :] * V[np.newaxis, :, :]
            Res = np.einsum('abr,cr->abc', Z, W)
            err = np.linalg.norm(Res - T)
            
            if err < tol:
                break
            if err < prev_err * (1 - 1e-4):
                prev_err = err
            else:
                # Stagnation, maybe break
                pass
                
        return U, V, W

    # Search loop
    # We prioritize ranks that are likely to be optimal.
    # Known: R(3,3,3)=21.
    # R(3,3,7) is likely in [21, 40].
    
    # Try a few ranks
    ranks_to_try = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 62, 63]
    # Add some random ranks in between if time permits?
    # For now, just iterate.
    
    # To ensure we return *something*, we start with the known valid solution (schoolbook) if search fails.
    # But we want the *best* (lowest rank).
    
    # We should also try to reuse the schoolbook structure?
    # No, random init is better for finding new structures.
    
    # Sort ranks to try? Usually optimal is low.
    # Let's try from low to high, but limit restarts per rank if time is tight.
    
    restarts_per_rank = 10
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        for restart in range(restarts_per_rank):
            if time.time() > deadline - 10:
                break
                
            U, V, W = run_als(R)
            sol = verify_and_snap(U, V, W, R)
            if sol is not None:
                # Found a valid solution for this R
                # Since we are trying increasing R (mostly), the first valid one is best?
                # We are not strictly increasing. We should check all and keep min.
                if best_solution is None or R < best_solution['rank']:
                    best_solution = {"U": sol["U"], "V": sol["V"], "W": sol["W"], "rank": R}
                    
                # If we found a very low rank, we can stop early?
                # Theoretical lower bound is 21. If we find 21, we can stop.
                if R == 21:
                    return best_solution
    
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to Schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
