import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    candidates = [18, 27, 30, 33, 36, 40, 45, 50, 54]
    
    time_alloc = {
        18: 3.0,
        27: 50.0,
        30: 25.0,
        33: 25.0,
        36: 15.0,
        40: 10.0
    }
    
    for R in candidates:
        if time.time() > deadline - 5:
            break
        t_budget = time_alloc.get(R, 5.0)
        if time.time() + t_budget > deadline - 2:
            t_budget = max(0, deadline - 2 - time.time())
        if t_budget <= 0:
            continue
            
        result = search_low_rank(n, m, p, R, t_budget)
        if result is not None:
            return result

    return construct_schoolbook(n, m, p)

def search_low_rank(n, m, p, target_rank, time_budget):
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.flatten()
    
    def full_reconstruction(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W, optimize=True).flatten()
    
    def compute_error(Res_flat):
        diff = Res_flat - T_flat
        return np.dot(diff, diff)
    
    def structured_init(U, V, W):
        """Create structured initializations based on tensor structure."""
        # Strategy 1: Random sparse
        U_s = np.zeros_like(U)
        V_s = np.zeros_like(V)
        W_s = np.zeros_like(W)
        for r in range(target_rank):
            # Pick random non-zero patterns
            u_idx = np.random.choice(num_u, size=min(3, num_u), replace=False)
            v_idx = np.random.choice(num_v, size=min(3, num_v), replace=False)
            w_idx = np.random.choice(num_w, size=min(3, num_w), replace=False)
            U_s[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
            V_s[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
            W_s[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
        return U_s, V_s, W_s
    
    def greedy_local_search(U, V, W, Res_flat, current_err, deadline):
        """Greedy local search with multiple delta options."""
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_Res_flat = Res_flat.copy()
        
        deltas = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            mat_idx = random.randint(0, 2)
            
            if mat_idx == 0:
                row = random.randint(0, num_u - 1)
                col = random.randint(0, target_rank - 1)
                old_val = U[row, col]
                delta = random.choice(deltas)
                U[row, col] = old_val + delta
                
                outer = np.outer(V[:, col], W[:, col])
                offset = row * num_v * num_w
                Res_slice = Res_flat[offset:offset+num_v*num_w].reshape(num_v, num_w)
                T_slice = T_flat[offset:offset+num_v*num_w].reshape(num_v, num_w)
                
                diff = Res_slice - T_slice
                delta_err = 2.0 * delta * np.sum(outer * diff) + delta * delta * np.sum(outer * outer)
                
                if delta_err < -1e-12:
                    Res_flat[offset:offset+num_v*num_w] = (Res_slice + delta * outer).flatten()
                    current_err += delta_err
                    no_improve_count = 0
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_Res_flat = Res_flat.copy()
                        if best_err < 1e-9:
                            return snap_and_verify(best_U, best_V, best_W, T_flat)
                else:
                    U[row, col] = old_val
                    no_improve_count += 1
                    
            elif mat_idx == 1:
                row = random.randint(0, num_v - 1)
                col = random.randint(0, target_rank - 1)
                old_val = V[row, col]
                delta = random.choice(deltas)
                V[row, col] = old_val + delta
                
                outer = np.outer(U[:, col], W[:, col])
                Res_3d = Res_flat.reshape(num_u, num_v, num_w)
                Res_slice = Res_3d[:, row, :].copy()
                T_slice = T_flat.reshape(num_u, num_v, num_w)[:, row, :].copy()
                
                diff = Res_slice - T_slice
                delta_err = 2.0 * delta * np.sum(outer * diff) + delta * delta * np.sum(outer * outer)
                
                if delta_err < -1e-12:
                    Res_3d[:, row, :] = Res_slice + delta * outer
                    Res_flat = Res_3d.flatten()
                    current_err += delta_err
                    no_improve_count = 0
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_Res_flat = Res_flat.copy()
                        if best_err < 1e-9:
                            return snap_and_verify(best_U, best_V, best_W, T_flat)
                else:
                    V[row, col] = old_val
                    no_improve_count += 1
                    
            else:
                row = random.randint(0, num_w - 1)
                col = random.randint(0, target_rank - 1)
                old_val = W[row, col]
                delta = random.choice(deltas)
                W[row, col] = old_val + delta
                
                outer = np.outer(U[:, col], V[:, col])
                Res_3d = Res_flat.reshape(num_u, num_v, num_w)
                Res_slice = Res_3d[:, :, row].copy()
                T_slice = T_flat.reshape(num_u, num_v, num_w)[:, :, row].copy()
                
                diff = Res_slice - T_slice
                delta_err = 2.0 * delta * np.sum(outer * diff) + delta * delta * np.sum(outer * outer)
                
                if delta_err < -1e-12:
                    Res_3d[:, :, row] = Res_slice + delta * outer
                    Res_flat = Res_3d.flatten()
                    current_err += delta_err
                    no_improve_count = 0
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_Res_flat = Res_flat.copy()
                        if best_err < 1e-9:
                            return snap_and_verify(best_U, best_V, best_W, T_flat)
                else:
                    W[row, col] = old_val
                    no_improve_count += 1
        
        return best_U, best_V, best_W, best_err, best_Res_flat
    
    def snap_and_verify(U, V, W, T_flat):
        """Try multiple snapping strategies."""
        strategies = [
            lambda x: np.round(x * 2) / 2,
            lambda x: np.round(x * 4) / 4,
            lambda x: np.round(x * 8) / 8,
        ]
        
        for snap_fn in strategies:
            bU = snap_fn(U)
            bV = snap_fn(V)
            bW = snap_fn(W)
            Res = np.einsum('ar,br,cr->abc', bU, bV, bW, optimize=True).flatten()
            snap_err = np.dot(Res - T_flat, Res - T_flat)
            if snap_err < 1e-9:
                return {
                    "U": bU.tolist(),
                    "V": bV.tolist(),
                    "W": bW.tolist()
                }
        
        # Also try rounding to nearest integer
        bU = np.round(U)
        bV = np.round(V)
        bW = np.round(W)
        Res = np.einsum('ar,br,cr->abc', bU, bV, bW, optimize=True).flatten()
        snap_err = np.dot(Res - T_flat, Res - T_flat)
        if snap_err < 1e-9:
            return {
                "U": bU.tolist(),
                "V": bV.tolist(),
                "W": bW.tolist()
            }
        
        return None
    
    random.seed(42)
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start = time.time()
    deadline = start + time_budget
    
    # Try multiple initialization strategies
    num_restarts = max(3, int(time_budget / 10))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Random integer init
            U = np.random.randint(-1, 2, size=(num_u, target_rank)).astype(float)
            V = np.random.randint(-1, 2, size=(num_v, target_rank)).astype(float)
            W = np.random.randint(-1, 2, size=(num_w, target_rank)).astype(float)
        elif restart == 1:
            # Sparse structured init
            U, V, W = structured_init(U, V, W)
        else:
            # Random with more zeros
            U = np.random.choice([-1, 0, 1], size=(num_u, target_rank)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(num_v, target_rank)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(num_w, target_rank)).astype(float)
        
        Res_flat = full_reconstruction(U, V, W)
        current_err = compute_error(Res_flat)
        
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-9:
                return snap_and_verify(best_U, best_V, best_W, T_flat)
        
        # Local search
        U, V, W, current_err, Res_flat = greedy_local_search(U, V, W, Res_flat, current_err, deadline)
        
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = U, V, W
            if best_err < 1e-9:
                return snap_and_verify(best_U, best_V, best_W, T_flat)
    
    # Final polish
    if best_err < 1.0 and time.time() < deadline - 2:
        U_pol = best_U.copy()
        V_pol = best_V.copy()
        W_pol = best_W.copy()
        err_pol = compute_error(full_reconstruction(U_pol, V_pol, W_pol))
        
        for _pass in range(4):
            improved = False
            for mat in [U_pol, V_pol, W_pol]:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        # Try snapping to half-integers
                        floor_val = np.floor(mat[i,j]*2)/2
                        ceil_val = np.ceil(mat[i,j]*2)/2
                        for cand in [floor_val, ceil_val]:
                            if abs(cand - mat[i,j]) < 1e-12:
                                continue
                            old = mat[i,j]
                            mat[i,j] = cand
                            new_err = compute_error(full_reconstruction(U_pol, V_pol, W_pol))
                            if new_err < err_pol - 1e-12:
                                err_pol = new_err
                                improved = True
                            else:
                                mat[i,j] = old
            if not improved:
                break
        
        if err_pol < best_err:
            best_err = err_pol
            best_U, best_V, best_W = U_pol, V_pol, W_pol
        
        if best_err < 1e-9:
            return snap_and_verify(best_U, best_V, best_W, T_flat)

    if best_err < 1e-6:
        return snap_and_verify(best_U, best_V, best_W, T_flat)

    return None

def construct_schoolbook(n, m, p):
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
