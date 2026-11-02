"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-10):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def compute_residual(U, V, W):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(S - T)
    
    def fast_als(U, V, W, max_iters=200, tol=1e-10):
        r = U.shape[1]
        if r == 0:
            return None
        
        Ur = U.copy()
        Vr = V.copy()
        Wr = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(nm, mp * pn)
                vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
                vw_norm_sq = np.dot(vw, vw)
                if vw_norm_sq > 1e-15:
                    Ur_new = S_minus_mat @ vw / vw_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur_new, Vr[:, r_idx], Wr[:, r_idx])
                    Ur[:, r_idx] = Ur_new
            
            # Update V
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(mp, nm * pn)
                uw = np.outer(Ur[:, r_idx], Wr[:, r_idx]).flatten()
                uw_norm_sq = np.dot(uw, uw)
                if uw_norm_sq > 1e-15:
                    Vr_new = S_minus_mat @ uw / uw_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur[:, r_idx], Vr_new, Wr[:, r_idx])
                    Vr[:, r_idx] = Vr_new
            
            # Update W
            S_total = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
            for r_idx in range(r):
                term = np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr[:, r_idx])
                S_minus = S_total - term
                S_minus_mat = S_minus.reshape(pn, nm * mp)
                uv = np.outer(Ur[:, r_idx], Vr[:, r_idx]).flatten()
                uv_norm_sq = np.dot(uv, uv)
                if uv_norm_sq > 1e-15:
                    Wr_new = S_minus_mat @ uv / uv_norm_sq
                    S_total = S_minus + np.einsum('a,b,c->abc', Ur[:, r_idx], Vr[:, r_idx], Wr_new)
                    Wr[:, r_idx] = Wr_new
            
            if it % 3 == 0 or it == max_iters - 1:
                res = np.linalg.norm(S_total - T)
                if res < tol:
                    if verify(Ur, Vr, Wr):
                        return Ur, Vr, Wr
                        
        return None
    
    def try_rank(r, n_attempts=5, max_iters=150):
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                # Random Gaussian
                Ur = np.random.randn(nm, r) * 0.5
                Vr = np.random.randn(mp, r) * 0.5
                Wr = np.random.randn(pn, r) * 0.5
            elif attempt == 1:
                # Sparse binary
                Ur = (np.random.rand(nm, r) < 0.3).astype(float)
                Vr = (np.random.rand(mp, r) < 0.3).astype(float)
                Wr = (np.random.rand(pn, r) < 0.3).astype(float)
            elif attempt == 2:
                # Structured initialization based on index patterns
                Ur = np.zeros((nm, r))
                Vr = np.zeros((mp, r))
                Wr = np.zeros((pn, r))
                for col in range(r):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n*m)) % p
                    Ur[i*m+j, col] = 1.0
                    Vr[j*p+k, col] = 1.0
                    Wr[k*n+i, col] = 1.0
                # Add some randomness to break symmetry
                Ur += np.random.randn(nm, r) * 0.1
                Vr += np.random.randn(mp, r) * 0.1
                Wr += np.random.randn(pn, r) * 0.1
            else:
                # Mix
                Ur = np.random.randn(nm, r) * 0.3
                Vr = np.random.randn(mp, r) * 0.3
                Wr = np.random.randn(pn, r) * 0.3
                Ur[np.random.rand(nm, r) < 0.4] = 0
                Vr[np.random.rand(mp, r) < 0.4] = 0
                Wr[np.random.rand(pn, r) < 0.4] = 0
            
            result = fast_als(Ur, Vr, Wr, max_iters=max_iters)
            if result is not None:
                return result
        
        return None
    
    # Try ranks from lower bound upwards
    lower_bound = max(nm, mp, pn)  # 28
    
    # Try a range of ranks, prioritizing lower ones
    search_ranks = list(range(lower_bound, min(lower_bound + 15, best_rank)))
    
    for target_rank in search_ranks:
        if time.time() > deadline:
            break
        
        res = try_rank(target_rank, n_attempts=4, max_iters=100)
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
            if target_rank == lower_bound:
                break
            break
    
    # If we found something, try to polish with more iterations
    if best_rank < n * m * p and time.time() < deadline - 30:
        polished = fast_als(best_U, best_V, best_W, max_iters=300, tol=1e-12)
        if polished is not None:
            best_U, best_V, best_W = polished
    
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
