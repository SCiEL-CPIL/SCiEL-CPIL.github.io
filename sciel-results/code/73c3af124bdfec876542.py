"""Program searching for low-rank exact tensor decomposition of 2x3 by 3x8 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    deadline = time.time() + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    base_rank = n * m * p

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-9, rtol=1e-9)

    def tensor_residual(U, V, W, T_target):
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T_target - computed
        return np.sum(diff * diff)

    # ---------- Schoolbook baseline ----------
    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)
    best = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist(), "rank": base_rank}

    # ---------- Strategy: Find low-rank decomposition for M_{2,3,2} and expand ----------
    # M_{2,3,2} tensor T2[i*3+j, j*2+k, k*2+i] = 1
    n2, m2, p2 = 2, 3, 2
    T2 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0

    def verify2(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T2, atol=1e-9, rtol=1e-9)

    def tensor_residual2(U, V, W):
        return tensor_residual(U, V, W, T2)

    def polish_small(U, V, W, T_target, max_iter=2000, tol=1e-14):
        """Polish with adaptive learning rate."""
        U = U.copy(); V = V.copy(); W = W.copy()
        prev_loss = tensor_residual(U, V, W, T_target)
        lr = 1e-3
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if prev_loss < tol:
                break
            computed = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = computed - T_target
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_loss = tensor_residual(U, V, W, T_target)
            if new_loss < prev_loss:
                lr = min(lr * 1.03, 1e-2)
            else:
                lr = max(lr * 0.85, 1e-8)
                U += lr * grad_U
                V += lr * grad_V
                W += lr * grad_W
            prev_loss = new_loss
        return U, V, W

    def find_low_rank_m232(max_rank=12, min_rank=8, max_iter=3000):
        """Search for a low-rank decomposition of M_{2,3,2} using multiple strategies."""
        rng = np.random.RandomState(SEED)
        best_result = None
        best_loss = float('inf')
        
        for r in range(min_rank, max_rank + 1):
            for restart in range(15):
                if time.time() > deadline:
                    return best_result
                
                # Different initialization strategies
                strategy = restart % 3
                if strategy == 0:
                    # Random normal
                    scale = 1.0 / np.sqrt(r)
                    U = rng.normal(size=(n2 * m2, r)) * scale
                    V = rng.normal(size=(m2 * p2, r)) * scale
                    W = rng.normal(size=(p2 * n2, r)) * scale
                elif strategy == 1:
                    # Sparse initialization
                    U = np.zeros((n2 * m2, r))
                    V = np.zeros((m2 * p2, r))
                    W = np.zeros((p2 * n2, r))
                    for c in range(r):
                        i, j = rng.randint(0, n2-1), rng.randint(0, m2-1)
                        U[i*m2+j, c] = rng.choice([-1, 1])
                        j2, k = rng.randint(0, m2-1), rng.randint(0, p2-1)
                        V[j2*p2+k, c] = rng.choice([-1, 1])
                        k2, i2 = rng.randint(0, p2-1), rng.randint(0, n2-1)
                        W[k2*n2+i2, c] = rng.choice([-1, 1])
                else:
                    # Structured initialization
                    U = np.zeros((n2 * m2, r))
                    V = np.zeros((m2 * p2, r))
                    W = np.zeros((p2 * n2, r))
                    for c in range(r):
                        i, j, k = rng.randint(0, n2-1), rng.randint(0, m2-1), rng.randint(0, p2-1)
                        U[i*m2+j, c] = 1.0
                        V[j*p2+k, c] = 1.0
                        W[k*n2+i, c] = 1.0
                
                # Gradient Descent with adaptive lr
                prev_loss = tensor_residual2(U, V, W)
                lr = 0.05
                for it in range(max_iter):
                    if time.time() > deadline:
                        break
                    if prev_loss < 1e-12:
                        break
                    computed = np.einsum('ic,jc,kc->ijk', U, V, W)
                    diff = computed - T2
                    grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
                    grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
                    grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    new_loss = tensor_residual2(U, V, W)
                    if new_loss < prev_loss * 0.999:
                        lr = min(lr * 1.05, 0.1)
                    else:
                        lr = max(lr * 0.95, 1e-6)
                    prev_loss = new_loss
                
                # Polish if close
                if prev_loss < 1e-8:
                    U, V, W = polish_small(U, V, W, T2, max_iter=1500)
                    final_loss = tensor_residual2(U, V, W)
                    if final_loss < best_loss:
                        best_loss = final_loss
                        if verify2(U, V, W):
                            return U, V, W
                        best_result = (U, V, W)
        
        return best_result

    # Try to find a decomposition for M_{2,3,2} with rank <= 12
    result232 = find_low_rank_m232(max_rank=12, min_rank=8)
    
    if result232 is not None:
        U_loc, V_loc, W_loc = result232
        r_loc = U_loc.shape[1]
        
        # Expand to M_{2,3,8}
        # p = 8, p2 = 2. Number of blocks = 4.
        num_blocks = p // p2
        r_total = r_loc * num_blocks
        
        U_full = np.zeros((n * m, r_total))
        V_full = np.zeros((m * p, r_total))
        W_full = np.zeros((p * n, r_total))
        
        # U is repeated for each block
        for b in range(num_blocks):
            U_full[:, b*r_loc:(b+1)*r_loc] = U_loc
        
        # V and W are block-diagonal
        for b in range(num_blocks):
            for j in range(m):
                for k_loc in range(p2):
                    global_row_v = j * p + b * p2 + k_loc
                    local_row_v = j * p2 + k_loc
                    V_full[global_row_v, b*r_loc:(b+1)*r_loc] = V_loc[local_row_v, :]
                    
            for k_loc in range(p2):
                for i in range(n):
                    global_row_w = (b * p2 + k_loc) * n + i
                    local_row_w = k_loc * n + i
                    W_full[global_row_w, b*r_loc:(b+1)*r_loc] = W_loc[local_row_w, :]
        
        # Verify the full decomposition
        if verify(U_full, V_full, W_full):
            # Round to half-integers if close
            U_r = np.round(U_full * 2) / 2
            V_r = np.round(V_full * 2) / 2
            W_r = np.round(W_full * 2) / 2
            if verify(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Try direct search on full tensor with rank reduction
    # Start from schoolbook and try to reduce rank via null space projection
    if time.time() < deadline - 30:
        rng_full = np.random.RandomState(SEED + 1)
        for target_rank in [40, 36, 32, 28]:
            if time.time() > deadline - 10:
                break
            
            # Initialize from schoolbook with random perturbations
            U_init = U0.copy()
            V_init = V0.copy()
            W_init = W0.copy()
            
            # Keep only target_rank columns
            col_indices = rng_full.choice(base_rank, size=target_rank, replace=False)
            U_trunc = U_init[:, col_indices]
            V_trunc = V_init[:, col_indices]
            W_trunc = W_init[:, col_indices]
            
            # Polish
            U_pol, V_pol, W_pol = polish_small(U_trunc, V_trunc, W_trunc, T, max_iter=2000)
            
            if verify(U_pol, V_pol, W_pol):
                U_r = np.round(U_pol * 2) / 2
                V_r = np.round(V_pol * 2) / 2
                W_r = np.round(W_pol * 2) / 2
                if verify(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
