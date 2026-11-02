import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 140s deadline, 10s safety margin before 150s kill

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Target tensor: T[a,b,c] = 1 where a=2i+j, b=6j+k, c=2k+i
    T_exp = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_exp[a, b, c] = 1.0

    # Strassen 2x2x2 steps
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    def build_strassen_block(g, U, V, W, col_offset):
        """Add a Strassen block for k=2g, 2g+1 at column offset."""
        for s in range(7):
            r = col_offset + s
            step = strassen_steps[s]
            k0 = 2 * g
            k1 = 2 * g + 1
            b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
            c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
            for l in range(4):
                U[l, r] = step['u'][l]
                V[b_map[l], r] = step['v'][l]
                W[c_map[l], r] = step['w'][l]

    def try_round_and_verify(U, V, W, T_target, scales=(1, 2, 4, 8)):
        """Try rounding to integers/half-integers at various scales."""
        for scale in scales:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_calc, T_target, atol=1e-7):
                return U_r, V_r, W_r
        # If no rounding works but values are already close
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_calc, T_target, atol=1e-7):
            return U, V, W
        return None

    def polish_als(U, V, W, T_target, max_iter=200, tol=1e-12, nm_=nm, mp_=mp, pn_=pn):
        """ALS polish: minimize ||sum_r u_r v_r w_r^T - T||^2."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            old_U, old_V, old_W = U.copy(), V.copy(), W.copy()
            
            # Solve for U: fix V, W
            M = np.zeros((mp_ * pn_, rank))
            for r in range(rank):
                M[:, r] = (V[:, r] * W[:, r]).flatten()
            T_flat = T_target.flatten()
            for a in range(nm_):
                T_a = T_flat[a*mp_*pn_:(a+1)*mp_*pn_]
                try:
                    U[a, :] = np.linalg.lstsq(M.T, T_a, rcond=None)[0]
                except:
                    pass
            
            # Solve for V: fix U, W
            N = np.zeros((nm_ * pn_, rank))
            for r in range(rank):
                N[:, r] = (U[:, r] * W[:, r]).flatten()
            for b in range(mp_):
                T_b = T_target[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(N.T, T_b, rcond=None)[0]
                except:
                    pass
            
            # Solve for W: fix U, V
            P = np.zeros((nm_ * mp_, rank))
            for r in range(rank):
                P[:, r] = (U[:, r] * V[:, r]).flatten()
            for c in range(pn_):
                T_c = T_target[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(P.T, T_c, rcond=None)[0]
                except:
                    pass
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T_target)
            if err < tol:
                break
            if np.linalg.norm(U - old_U) + np.linalg.norm(V - old_V) + np.linalg.norm(W - old_W) < 1e-14:
                break
        
        return U, V, W

    def try_rank_local(R, T_target, time_budget=30.0, n_restarts=10, seed=42, dims=(4,8,8)):
        """Try to find rank-R decomposition for a local tensor."""
        nm_l, mp_l, pn_l = dims
        np.random.seed(seed)
        best_err = np.inf
        best_result = None
        
        def objective(params):
            U = params[:R*nm_l].reshape(nm_l, R)
            V = params[R*nm_l:R*nm_l+R*mp_l].reshape(mp_l, R)
            W = params[R*nm_l+R*mp_l:R*nm_l+R*mp_l+R*pn_l].reshape(pn_l, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_calc - T_target).flatten()
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                x0 = np.random.randn(R * (nm_l + mp_l + pn_l)) * 0.5
            elif restart % 3 == 1:
                x0 = np.random.randn(R * (nm_l + mp_l + pn_l)) * 1.0
            else:
                x0 = np.random.randn(R * (nm_l + mp_l + pn_l)) * 0.2
            
            try:
                sol = opt.least_squares(objective, x0, method='lm', max_nfev=1000)
                if sol.cost < 1e-8:
                    U = sol.x[:R*nm_l].reshape(nm_l, R)
                    V = sol.x[R*nm_l:R*nm_l+R*mp_l].reshape(mp_l, R)
                    W = sol.x[R*nm_l+R*mp_l:R*nm_l+R*mp_l+R*pn_l].reshape(pn_l, R)
                    
                    # Polish
                    U, V, W = polish_als(U, V, W, T_target, max_iter=100, dims=dims)
                    
                    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(T_calc - T_target)
                    
                    if err < best_err:
                        best_err = err
                        best_result = (U, V, W)
                    
                    if err < 1e-10:
                        rounded = try_round_and_verify(U, V, W, T_target)
                        if rounded is not None:
                            return rounded
                        return (U, V, W)
            except Exception:
                pass
        
        return best_result

    # Strategy 1: Try rank 20 = 7 (Strassen for k=0,1) + 13 (for k=2,3,4,5)
    # Build Strassen for k=0,1
    U_s = np.zeros((nm, 7))
    V_s = np.zeros((mp, 7))
    W_s = np.zeros((pn, 7))
    build_strassen_block(0, U_s, V_s, W_s, 0)
    
    # Local tensor for k=2,3,4,5: shape (4, 8, 8)
    # a = 2i+j (i in 0,1; j in 0,1) -> 4 values
    # local b: j*4+kp (j in 0,1; kp in 0,1,2,3) -> 8 values
    # local c: i*4+kp (i in 0,1; kp in 0,1,2,3) -> 8 values
    T_local = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for kp in range(4):
                a = i * 2 + j
                lb = j * 4 + kp
                lc = i * 4 + kp
                T_local[a, lb, lc] = 1.0
    
    result13 = try_rank_local(13, T_local, time_budget=40, n_restarts=15, seed=77, dims=(4,8,8))
    if result13 is not None:
        U13, V13, W13 = result13
        # Assemble full rank-20
        U_full = np.zeros((nm, 20))
        V_full = np.zeros((mp, 20))
        W_full = np.zeros((pn, 20))
        
        U_full[:, :7] = U_s
        V_full[:, :7] = V_s
        W_full[:, :7] = W_s
        U_full[:, 7:20] = U13
        
        # Map V13 (local b = j*4+kp) to global V (b = j*6+k, k=2+kp)
        for r in range(13):
            for lb in range(8):
                val = V13[lb, r]
                if abs(val) > 1e-15:
                    j = lb // 4
                    kp = lb % 4
                    k = 2 + kp
                    gb = j * 6 + k
                    V_full[gb, 7+r] = val
        
        # Map W13 (local c = i*4+kp) to global W (c = k*2+i, k=2+kp)
        for r in range(13):
            for lc in range(8):
                val = W13[lc, r]
                if abs(val) > 1e-15:
                    i = lc // 4
                    kp = lc % 4
                    k = 2 + kp
                    gc = k * 2 + i
                    W_full[gc, 7+r] = val
        
        T_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        if np.allclose(T_full, T_exp, atol=1e-6):
            rounded = try_round_and_verify(U_full, V_full, W_full, T_exp)
            if rounded is not None:
                U_full, V_full, W_full = rounded
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Strategy 2: Try rank 19 = 7 + 7 + 5 (two Strassen blocks + rank 5 for last 2 cols)
    if time.time() < DEADLINE - 30:
        U_s2 = np.zeros((nm, 14))
        V_s2 = np.zeros((mp, 14))
        W_s2 = np.zeros((pn, 14))
        build_strassen_block(0, U_s2, V_s2, W_s2, 0)
        build_strassen_block(1, U_s2, V_s2, W_s2, 7)
        
        # Local tensor for k=4,5: shape (4, 4, 4)
        T_local5 = np.zeros((4, 4, 4))
        for i in range(2):
            for j in range(2):
                for kp in range(2):
                    a = i * 2 + j
                    lb = j * 2 + kp
                    lc = i * 2 + kp
                    T_local5[a, lb, lc] = 1.0
        
        result5 = try_rank_local(5, T_local5, time_budget=20, n_restarts=10, seed=55, dims=(4,4,4))
        if result5 is not None:
            U5, V5, W5 = result5
            U_full = np.zeros((nm, 19))
            V_full = np.zeros((mp, 19))
            W_full = np.zeros((pn, 19))
            
            U_full[:, :14] = U_s2
            V_full[:, :14] = V_s2
            W_full[:, :14] = W_s2
            U_full[:, 14:19] = U5
            
            # Map V5 (local b = j*2+kp) to global V (b = j*6+k, k=4+kp)
            for r in range(5):
                for lb in range(4):
                    val = V5[lb, r]
                    if abs(val) > 1e-15:
                        j = lb // 2
                        kp = lb % 2
                        k = 4 + kp
                        gb = j * 6 + k
                        V_full[gb, 14+r] = val
            
            # Map W5 (local c = i*2+kp) to global W (c = k*2+i, k=4+kp)
            for r in range(5):
                for lc in range(4):
                    val = W5[lc, r]
                    if abs(val) > 1e-15:
                        i = lc // 2
                        kp = lc % 2
                        k = 4 + kp
                        gc = k * 2 + i
                        W_full[gc, 14+r] = val
            
            T_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
            if np.allclose(T_full, T_exp, atol=1e-6):
                rounded = try_round_and_verify(U_full, V_full, W_full, T_exp)
                if rounded is not None:
                    U_full, V_full, W_full = rounded
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Rank 21 via 3 Strassen blocks
    U = np.zeros((nm, 21))
    V = np.zeros((mp, 21))
    W = np.zeros((pn, 21))
    for g in range(3):
        build_strassen_block(g, U, V, W, 7 * g)
    
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_check, T_exp, atol=1e-6):
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
