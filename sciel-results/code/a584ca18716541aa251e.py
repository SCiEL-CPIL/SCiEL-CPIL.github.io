import numpy as np
from scipy.optimize import least_squares, differential_evolution
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Target tensor T[i, j, j_prime, k] = delta_{j, j_prime}
    # Shape: (n, m, m, p)
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
                
    def compute_M(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        M = np.zeros((n, m, m, p))
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            term = np.einsum('ki,ij,ak->ijak', w_r, u_r, v_r)
            M += term
        return M
        
    def residuals(params, R):
        size_U = nm * R
        size_V = mp * R
        U = params[:size_U].reshape(nm, R)
        V = params[size_U:size_U + size_V].reshape(mp, R)
        W = params[size_U + size_V:].reshape(pn, R)
        
        M = compute_M(U, V, W, R)
        return (M - T).flatten()
        
    def try_snap_and_verify(U_opt, V_opt, W_opt, target_rank):
        """Try to snap to half-integers and verify"""
        # Try multiple snap strategies
        for scale in [2, 4, 8]:
            U_snap = np.round(U_opt * scale) / scale
            V_snap = np.round(V_opt * scale) / scale
            W_snap = np.round(W_opt * scale) / scale
            
            M_snap = compute_M(U_snap, V_snap, W_snap, target_rank)
            if np.allclose(M_snap, T, atol=1e-8):
                return U_snap, V_snap, W_snap, True
        
        # Also try integer snapping
        U_snap = np.round(U_opt)
        V_snap = np.round(V_opt)
        W_snap = np.round(W_opt)
        M_snap = compute_M(U_snap, V_snap, W_snap, target_rank)
        if np.allclose(M_snap, T, atol=1e-8):
            return U_snap.astype(float), V_snap.astype(float), W_snap.astype(float), True
            
        return None, None, None, False
        
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Focus on Rank 11
    target_rank = 11
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Random restarts with LM
    num_restarts = 30
    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Diverse initialization strategies
        strategy = r_idx % 4
        
        if strategy == 0:
            U0 = rng.normal(0, 1, (nm, target_rank))
            V0 = rng.normal(0, 1, (mp, target_rank))
            W0 = rng.normal(0, 1, (pn, target_rank))
        elif strategy == 1:
            # Sparse initialization
            U0 = rng.choice([-1, 0, 1], (nm, target_rank))
            V0 = rng.choice([-1, 0, 1], (mp, target_rank))
            W0 = rng.choice([-1, 0, 1], (pn, target_rank))
        elif strategy == 2:
            # Half-integer initialization
            U0 = rng.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], (nm, target_rank))
            V0 = rng.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], (mp, target_rank))
            W0 = rng.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], (pn, target_rank))
        else:
            # Small values
            U0 = rng.uniform(-0.5, 0.5, (nm, target_rank))
            V0 = rng.uniform(-0.5, 0.5, (mp, target_rank))
            W0 = rng.uniform(-0.5, 0.5, (pn, target_rank))
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, params0, args=(target_rank,), method='lm', max_nfev=300)
            
            if result.success or result.cost < 1e-4:
                U_opt = result.x[:nm * target_rank].reshape(nm, target_rank)
                V_opt = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_opt = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                
                M_opt = compute_M(U_opt, V_opt, W_opt, target_rank)
                err = np.linalg.norm(M_opt - T)
                
                if err < 1e-3:
                    U_snap, V_snap, W_snap, found = try_snap_and_verify(U_opt, V_opt, W_opt, target_rank)
                    if found:
                        best_rank = target_rank
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        break
                        
        except Exception:
            pass
    
    # If rank 11 not found, try rank 12 as fallback
    if best_rank > target_rank and time.time() < deadline - 30:
        target_rank = 12
        for r_idx in range(10):
            if time.time() > deadline:
                break
                
            U0 = rng.normal(0, 1, (nm, target_rank))
            V0 = rng.normal(0, 1, (mp, target_rank))
            W0 = rng.normal(0, 1, (pn, target_rank))
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, params0, args=(target_rank,), method='lm', max_nfev=300)
                
                if result.success or result.cost < 1e-4:
                    U_opt = result.x[:nm * target_rank].reshape(nm, target_rank)
                    V_opt = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                    W_opt = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                    
                    M_opt = compute_M(U_opt, V_opt, W_opt, target_rank)
                    err = np.linalg.norm(M_opt - T)
                    
                    if err < 1e-3:
                        U_snap, V_snap, W_snap, found = try_snap_and_verify(U_opt, V_opt, W_opt, target_rank)
                        if found:
                            best_rank = target_rank
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            break
                            
            except Exception:
                pass
    
    # Convert best solution to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
