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
    
    # Schoolbook decomposition
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
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(S, T, atol=tol)
    
    def residual(U, V, W):
        """Compute ||S - T||_F where S = sum_r U_r V_r W_r"""
        S = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(S - T)
    
    def iterative_rank_reduction(U, V, W, target_rank, max_iters=200, tol=1e-10):
        """
        Use iterative refinement to reduce rank.
        Start with target_rank random columns, then optimize via alternating least squares.
        """
        if target_rank >= U.shape[1]:
            return None
        
        r = target_rank
        # Random initialization from existing decomposition
        cols = np.random.choice(U.shape[1], r, replace=False)
        Ur = U[:, cols].copy()
        Vr = V[:, cols].copy()
        Wr = W[:, cols].copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Precompute T reshapes
            T_mat_a = T.reshape(nm, mp * pn)
            T_mat_b = T.reshape(mp, nm * pn)
            T_mat_c = T.reshape(pn, nm * mp)
            
            # Alternating least squares with efficient S_other computation
            # For each mode, we can compute S_other more efficiently by maintaining a running sum
            
            # Fix V, W, solve for U
            # Precompute V_w outer products
            for r_idx in range(r):
                if time.time() > deadline:
                    return None
                
                # Compute S without r_idx efficiently
                # S_other[a,b,c] = sum_{s != r_idx} Ur[a,s] Vr[b,s] Wr[c,s]
                # We can compute this as: total_S - Ur[:,r_idx] Vr[:,r_idx] Wr[:,r_idx]
                # But total_S changes as we update, so let's just compute directly but more efficiently
                
                S_other = np.zeros((nm, mp, pn))
                for s in range(r):
                    if s != r_idx:
                        S_other += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                
                S_other_mat = S_other.reshape(nm, mp * pn)
                residual_mat = T_mat_a - S_other_mat
                
                # V_r W_r reshaped: (mp*pn,)
                vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
                
                # Solve: residual_mat = U_new[:, r_idx] * vw^T  (rank-1 approximation)
                # U_new[:, r_idx] = residual_mat @ vw / (vw @ vw)
                vw_norm_sq = np.dot(vw, vw)
                if vw_norm_sq < 1e-15:
                    continue
                Ur[:, r_idx] = residual_mat @ vw / vw_norm_sq
            
            # Now fix U, W, solve for V
            for r_idx in range(r):
                if time.time() > deadline:
                    return None
                
                S_other = np.zeros((nm, mp, pn))
                for s in range(r):
                    if s != r_idx:
                        S_other += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                
                S_other_mat = S_other.reshape(mp, nm * pn)
                residual_mat = T_mat_b - S_other_mat
                
                uw = np.outer(Ur[:, r_idx], Wr[:, r_idx]).flatten()
                uw_norm_sq = np.dot(uw, uw)
                if uw_norm_sq < 1e-15:
                    continue
                Vr[:, r_idx] = residual_mat @ uw / uw_norm_sq
            
            # Fix U, V, solve for W
            for r_idx in range(r):
                if time.time() > deadline:
                    return None
                
                S_other = np.zeros((nm, mp, pn))
                for s in range(r):
                    if s != r_idx:
                        S_other += np.einsum('a,b,c->abc', Ur[:, s], Vr[:, s], Wr[:, s])
                
                S_other_mat = S_other.reshape(pn, nm * mp)
                residual_mat = T_mat_c - S_other_mat
                
                uv = np.outer(Ur[:, r_idx], Vr[:, r_idx]).flatten()
                uv_norm_sq = np.dot(uv, uv)
                if uv_norm_sq < 1e-15:
                    continue
                Wr[:, r_idx] = residual_mat @ uv / uv_norm_sq
            
            # Check convergence
            if it % 5 == 0:
                res = residual(Ur, Vr, Wr)
                if res < tol:
                    if verify(Ur, Vr, Wr):
                        return Ur, Vr, Wr
        
        # Final check
        res = residual(Ur, Vr, Wr)
        if res < tol:
            if verify(Ur, Vr, Wr):
                return Ur, Vr, Wr
        
        return None
    
    def try_direct_rank(r, n_attempts=10, max_iters=300):
        """Try to find rank-r decomposition with random initializations."""
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # From schoolbook, pick r random columns
                cols = np.random.choice(n*m*p, r, replace=False)
                Ur = U_s[:, cols].copy()
                Vr = V_s[:, cols].copy()
                Wr = W_s[:, cols].copy()
            elif attempt == 1:
                # Sparse random
                Ur = np.random.choice([0, 1], (nm, r)) * (np.random.rand(nm, r) < 0.3)
                Vr = np.random.choice([0, 1], (mp, r)) * (np.random.rand(mp, r) < 0.3)
                Wr = np.random.choice([0, 1], (pn, r)) * (np.random.rand(pn, r) < 0.3)
            elif attempt == 2:
                # Sparse random with different density
                Ur = np.random.choice([0, 1], (nm, r)) * (np.random.rand(nm, r) < 0.5)
                Vr = np.random.choice([0, 1], (mp, r)) * (np.random.rand(mp, r) < 0.5)
                Wr = np.random.choice([0, 1], (pn, r)) * (np.random.rand(pn, r) < 0.5)
            else:
                # Small random
                scale = 1.0 / np.sqrt(r)
                Ur = np.random.randn(nm, r) * scale
                Vr = np.random.randn(mp, r) * scale
                Wr = np.random.randn(pn, r) * scale
            
            result = iterative_rank_reduction(Ur, Vr, Wr, r, max_iters=max_iters)
            if result is not None:
                return result
        
        return None
    
    # Try reducing from schoolbook rank downward
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Also rank >= n*p = 21 (from output dimension)
    # For 3x4x7, known results suggest rank around 28-35
    
    start_rank = max(28, best_rank - 5)
    for target_rank in range(start_rank, 27, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_direct_rank(target_rank, n_attempts=8, max_iters=250)
        if result is not None:
            Ur, Vr, Wr = result
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
            # Try to go lower immediately
            start_rank = target_rank - 1
    
    # Also try some specific ranks with more attempts
    for target_rank in [28, 29, 30, 31, 32]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_direct_rank(target_rank, n_attempts=12, max_iters=350)
        if result is not None:
            Ur, Vr, Wr = result
            best_rank = target_rank
            best_U, best_V, best_W = Ur, Vr, Wr
    
    # Convert to lists
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
