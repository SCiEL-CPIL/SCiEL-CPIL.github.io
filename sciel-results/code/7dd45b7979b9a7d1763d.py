import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 5]
DEADLINE = 135.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _snap_to_half_integers(mat):
    """Snap values to nearest half-integer."""
    return np.round(mat * 2) / 2

def _try_rank_optim(n, m, p, target_rank, time_limit=8, n_restarts=10):
    """Try to find a decomposition of target_rank using optimization with multiple restarts."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    def objective_and_grad(params):
        U = params[:size_U].reshape(n*m, target_rank)
        V = params[size_U : size_U + size_V].reshape(m*p, target_rank)
        W = params[size_U + size_V :].reshape(p*n, target_rank)
        # T_recon[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        diff = T_recon - T
        loss = np.sum(diff ** 2)
        # Gradient
        grad_U = np.einsum('abc,bcd->acd', diff, V * W[None, :, :] * 1) # careful
        # More properly:
        # dL/dU[a,c] = 2 * sum_{b,c_idx} diff[a,b,c_idx] * V[b,c] * W[c_idx,c]
        VW = V * W[None, :, :]  # [b, c_idx, c]
        grad_U = 2.0 * np.einsum('abc,bcc_idx->ac', diff, np.einsum('bdc,cd_idx->bcd', V, W)) if False else None
        # Let's do it properly:
        # diff[a,b,c_idx] * V[b,d] * W[c_idx,d] summed over b,c_idx for each a,d
        # Actually: dL/dU[a,d] = 2 * sum_{b,c_idx} diff[a,b,c_idx] * V[b,d] * W[c_idx,d]
        tmp1 = np.einsum('abd,cd->abc', diff, V)  # no
        # Let me just use finite difference or autograd-like:
        # For simplicity and correctness:
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        # Using einsum:
        # grad_U[a,d] = 2 * sum_b,c diff[a,b,c] * V[b,d] * W[c,d]
        diff_bcd = np.einsum('abc,bdc->abd', diff, W)  # this isn't right either
        # Let me think carefully.
        # T_recon[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
        # d(T_recon[a,b,c])/dU[a0,d0] = delta(a,a0)*delta(d,d0)*V[b,d0]*W[c,d0]
        # So dL/dU[a0,d0] = 2 * sum_{b,c} diff[a0,b,c] * V[b,d0] * W[c,d0]
        # = 2 * (diff[:, :, :] * W[None, :, :]) contracted appropriately
        # Let me build it:
        VW = V[:, :, np.newaxis] * W[np.newaxis, :, :]  # [b, c_idx, d] -> V[b,d]*W[c,d] is not right
        # V[b,d] * W[c,d]: shape [b, c, d]
        VWd = np.einsum('bd,cd->bcd', V, W)  # VWd[b,c,d] = V[b,d]*W[c,d]
        grad_U = 2.0 * np.einsum('abc,bcd->ad', diff, VWd)
        # Similarly: dL/dV[b0,d0] = 2 * sum_{a,c} diff[a,b0,c] * U[a,d0] * W[c,d0]
        U_W = np.einsum('ad,cd->acd', U, W)  # U_W[a,c,d] = U[a,d]*W[c,d]
        grad_V = 2.0 * np.einsum('abc,acd->bd', diff, U_W)
        # dL/dW[c0,d0] = 2 * sum_{a,b} diff[a,b,c0] * U[a,d0] * V[b,d0]
        UV = np.einsum('ad,bd->abd', U, V)  # UV[a,b,d] = U[a,d]*V[b,d]
        grad_W = 2.0 * np.einsum('abc,abd->cd', diff, UV)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    best_loss = np.inf
    best_params = None
    
    for restart in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
        
        np.random.seed(restart * 137)
        if restart == 0:
            # Initialize near schoolbook-like: for each (i,j,k), set corresponding entries
            init = np.zeros(total_size)
            # For a sparse initialization, pick random subset
            pass
        
        # Random initialization
        scale = 0.5
        init = np.random.normal(0, scale, size=total_size)
        
        try:
            res = opt.minimize(objective_and_grad, init, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            if best_loss < 1e-8:
                break
        except Exception:
            continue
    
    if best_params is None or best_loss > 1e-6:
        return None
    
    U = best_params[:size_U].reshape(n*m, target_rank)
    V = best_params[size_U : size_U + size_V].reshape(m*p, target_rank)
    W = best_params[size_U + size_V :].reshape(p*n, target_rank)
    
    # Snap to half-integers
    U_snapped = _snap_to_half_integers(U)
    V_snapped = _snap_to_half_integers(V)
    W_snapped = _snap_to_half_integers(W)
    
    if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
        return U_snapped, V_snapped, W_snapped
    
    # Try without snapping
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to find rank 44 first (one less than schoolbook)
    # Then try lower if time permits
    for target_rank in range(best_rank - 1, 35, -1):
        if time.time() > DEADLINE - 15:
            break
        
        remaining_time = DEADLINE - 15 - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        time_per_try = min(8, remaining_time / 3)
        result = _try_rank_optim(n, m, p, target_rank, time_limit=time_per_try, n_restarts=8)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            best_rank = target_rank
            # Don't break; try to go lower if time allows
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                v = round(float(val) * 2) / 2
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(float(v))
            result.append(new_row)
        return result
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    # Final verification
    if not _verify(np.array(U_out, dtype=float), np.array(V_out, dtype=float),
                   np.array(W_out, dtype=float), n, m, p):
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        U_out = to_list(U_sb)
        V_out = to_list(V_sb)
        W_out = to_list(W_sb)
    
    return {"U": U_out, "V": V_out, "W": W_out}
