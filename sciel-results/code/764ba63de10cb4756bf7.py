"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute index mappings for efficient updates
    idx_ijk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_ijk.append((i * m + j, j * p + k, k * n + i))
                
    # Mask of 1s
    mask_1 = np.zeros(T.shape, dtype=bool)
    for (a, b, c) in idx_ijk:
        mask_1[a, b, c] = True

    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose T"""
        U = U.astype(np.float64)
        V = V.astype(np.float64)
        W = W.astype(np.float64)
        
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        
        # Check entries that should be 1
        if not np.allclose(T_recon[mask_1], 1.0, atol=tol):
            return False
        
        # Check entries that should be 0
        if not np.allclose(T_recon[~mask_1], 0.0, atol=tol):
            return False
            
        return True

    def compute_residual(U, V, W):
        """Compute Frobenius norm of residual"""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.linalg.norm(T_recon - T)

    def try_rank(r, seed=42, max_iters=800):
        if time.time() > deadline - 2:
            return None
        
        np.random.seed(seed)
        
        # Multiple initialization strategies
        init_strategies = [
            'random',       # standard normal
            'sparse',       # sparse random
            'structured',   # structured initialization
        ]
        
        for init_type in init_strategies:
            if time.time() > deadline - 2:
                return None
            
            if init_type == 'random':
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
            elif init_type == 'sparse':
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for d in range(r):
                    # Each rank-1 term has some structure
                    for a in range(n * m):
                        if np.random.random() < 0.3:
                            U[a, d] = np.random.choice([-1, 1])
                    for b in range(m * p):
                        if np.random.random() < 0.3:
                            V[b, d] = np.random.choice([-1, 1])
                    for c in range(p * n):
                        if np.random.random() < 0.3:
                            W[c, d] = np.random.choice([-1, 1])
            else:  # structured
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            
            # Damped Newton with line search
            prev_res = compute_residual(U, V, W)
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    return None
                
                # Compute updates for U, V, W using Gauss-Newton
                P = V[:, None, :] * W[None, :, :]  # (m*p, p*n, r)
                h_U = np.einsum('abc,bcd->ad', T, P)
                G_U = np.einsum('abc,bcd,bce->ade', T, P, P)
                
                # Damped solve for U
                U_new = U.copy()
                for a in range(n * m):
                    G = G_U[a] + 1e-6 * np.eye(r)  # damping for numerical stability
                    try:
                        U_new[a, :] = np.linalg.solve(G, h_U[a, :])
                    except np.linalg.LinAlgError:
                        U_new[a, :] = np.linalg.lstsq(G, h_U[a, :])[0]
                
                Q = U_new[:, None, :] * W[None, :, :]
                h_V = np.einsum('abc,acd->bd', T, Q)
                G_V = np.einsum('abc,acd,ace->bde', T, Q, Q)
                
                V_new = V.copy()
                for b in range(m * p):
                    G = G_V[b] + 1e-6 * np.eye(r)
                    try:
                        V_new[b, :] = np.linalg.solve(G, h_V[b, :])
                    except np.linalg.LinAlgError:
                        V_new[b, :] = np.linalg.lstsq(G, h_V[b, :])[0]
                
                R = U_new[:, None, :] * V_new[None, :, :]
                h_W = np.einsum('abc,abd->cd', T, R)
                G_W = np.einsum('abc,abd,abe->cde', T, R, R)
                
                W_new = W.copy()
                for c in range(p * n):
                    G = G_W[c] + 1e-6 * np.eye(r)
                    try:
                        W_new[c, :] = np.linalg.solve(G, h_W[c, :])
                    except np.linalg.LinAlgError:
                        W_new[c, :] = np.linalg.lstsq(G, h_W[c, :])[0]
                
                # Line search to ensure descent
                best_alpha = 1.0
                best_res = compute_residual(U_new, V_new, W_new)
                
                if best_res < prev_res:
                    U, V, W = U_new, V_new, W_new
                    prev_res = best_res
                else:
                    # Backtracking line search
                    alpha = 0.5
                    for _ in range(10):
                        U_trial = U + alpha * (U_new - U)
                        V_trial = V + alpha * (V_new - V)
                        W_trial = W + alpha * (W_new - W)
                        res_trial = compute_residual(U_trial, V_trial, W_trial)
                        if res_trial < prev_res:
                            U, V, W = U_trial, V_trial, W_trial
                            prev_res = res_trial
                            best_alpha = alpha
                            break
                        alpha *= 0.5
                    else:
                        # No improvement, try smaller step
                        U = U + 0.1 * (U_new - U)
                        V = V + 0.1 * (V_new - V)
                        W = W + 0.1 * (W_new - W)
                        prev_res = compute_residual(U, V, W)
                
                # Check for exactness periodically
                if it % 20 == 0:
                    for scale in [1, 2, 4, 8]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        if check_exact(U_r, V_r, W_r):
                            return (r, U_r, V_r, W_r)
                
                # Early exit if residual is very small
                if prev_res < 1e-12:
                    for scale in [1, 2, 4, 8, 16]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        if check_exact(U_r, V_r, W_r):
                            return (r, U_r, V_r, W_r)
                    break
        
        # Final check with multiple scales
        for scale in [1, 2, 4, 8, 16, 32]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_exact(U_r, V_r, W_r):
                return (r, U_r, V_r, W_r)
                
        return None

    # Try various ranks starting from lower bound
    # Known lower bound for 3x3x3x5 is around 20-25
    # Try from 22 upwards
    ranks_to_try = list(range(22, rank_sb))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        # Try multiple seeds with different initialization
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found a rank, try next rank
    
    return best_result
# EVOLVE-BLOCK-END
