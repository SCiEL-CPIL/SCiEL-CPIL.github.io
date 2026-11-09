import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matmul tensor for (n,m,p)."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(S, 1.0, atol=1e-9)

def _snap_to_half_integer(M):
    """Snap matrix entries to the nearest half-integer."""
    return np.round(M * 2) / 2

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of L = 0.5 * sum_ijk (S_ijk - 1)^2 where S_ijk = sum_r U_ijk V_jkr W_kir."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    E = S - 1.0
    # dL/dU_ijk = sum_k E_ijk * V_jkr * W_kir  (fix r index in einsum)
    grad_U_r = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
    # dL/dV_jkr = sum_i E_ijk * U_ijk * W_kir
    grad_V_r = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
    # dL/dW_kir = sum_j E_ijk * U_ijk * V_jkr
    grad_W_r = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
    return grad_U_r.reshape(n * m, rank), grad_V_r.reshape(m * p, rank), grad_W_r.reshape(p * n, rank)

def _try_rank(n, m, p, target_rank, max_time, num_restarts=10):
    """Try to find a decomposition of the given rank."""
    start_time = time.time()
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    U_shape = (n * m, target_rank)
    V_shape = (m * p, target_rank)
    W_shape = (p * n, target_rank)
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(restart * 100 + target_rank * 7)
        # Smaller init for stability
        U = np.random.randn(*U_shape) * 0.3
        V = np.random.randn(*V_shape) * 0.3
        W = np.random.randn(*W_shape) * 0.3
        
        lr = 0.02
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        prev_loss = float('inf')
        stuck_count = 0
        
        for it in range(2000):
            if time.time() - start_time > max_time:
                break
            
            grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p)
            
            # Momentum update
            momentum_U = 0.9 * momentum_U - lr * grad_U
            momentum_V = 0.9 * momentum_V - lr * grad_V
            momentum_W = 0.9 * momentum_W - lr * grad_W
            
            U += momentum_U
            V += momentum_V
            W += momentum_W
            
            # Compute loss
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            loss = 0.5 * np.sum((S - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_loss < 1e-12:
                U_snap = _snap_to_half_integer(best_U)
                V_snap = _snap_to_half_integer(best_V)
                W_snap = _snap_to_half_integer(best_W)
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap, target_rank
                # Try without snap - maybe exact solution isn't half-integer
                if _validate(best_U, best_V, best_W, n, m, p):
                    return best_U, best_V, best_W, target_rank
                break
            
            # Adaptive learning rate
            if loss > prev_loss * 1.01:
                lr *= 0.5
                stuck_count += 1
                if stuck_count > 20:
                    break
            else:
                stuck_count = 0
            prev_loss = loss
            
            # Decay learning rate slowly
            if it % 100 == 0 and it > 0:
                lr *= 0.95
                
    # Final polish on best solution if close
    if best_U is not None and best_loss < 1e-6:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        lr = 0.001
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        for it in range(500):
            if time.time() - start_time > max_time:
                break
            gU, gV, gW = _compute_gradients(U, V, W, n, m, p)
            mU = 0.9 * mU - lr * gU
            mV = 0.9 * mV - lr * gV
            mW = 0.9 * mW - lr * gW
            U += mU
            V += mV
            W += mW
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            loss = 0.5 * np.sum((S - 1.0)**2)
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            if best_loss < 1e-12:
                U_snap = _snap_to_half_integer(best_U)
                V_snap = _snap_to_half_integer(best_V)
                W_snap = _snap_to_half_integer(best_W)
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap, target_rank
                if _validate(best_U, best_V, best_W, n, m, p):
                    return best_U, best_V, best_W, target_rank
                break
    
    return None, None, None, best_loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = n * m * p  # 40
    best_U, best_V, best_W = None, None, None
    
    # Strategy 1: Solve sub-problem M_{2,2,5} and duplicate
    # Rank(M_{2,2,5}) lower bound is 10 (Strassen-like). If R_sub found, total = 2*R_sub
    n_sub, m_sub, p_sub = 2, 2, 5
    
    for R_sub in range(10, 16):
        if time.time() > deadline - 10:
            break
        time_left = deadline - time.time() - 10
        if time_left < 3:
            break
        
        # Give more restarts for lower ranks
        nr = 8 if R_sub <= 12 else 5
        U_sub, V_sub, W_sub, _ = _try_rank(n_sub, m_sub, p_sub, R_sub, 
                                           max_time=time_left * 0.6, num_restarts=nr)
        
        if U_sub is not None:
            R_total = 2 * R_sub
            U_full = np.zeros((n * m, R_total))
            V_full = np.zeros((m * p, R_total))
            W_full = np.zeros((p * n, R_total))
            
            # First block: j=0,1
            for i in range(n_sub):
                for js in range(m_sub):
                    U_full[i * m + js, :R_sub] = U_sub[i * m_sub + js, :]
            for js in range(m_sub):
                for k in range(p_sub):
                    V_full[js * p + k, :R_sub] = V_sub[js * p_sub + k, :]
            W_full[:, :R_sub] = W_sub
            
            # Second block: j=2,3
            for i in range(n_sub):
                for js in range(m_sub):
                    U_full[i * m + js + 2, R_sub:] = U_sub[i * m_sub + js, :]
            for js in range(m_sub):
                for k in range(p_sub):
                    V_full[(js + 2) * p + k, R_sub:] = V_sub[js * p_sub + k, :]
            W_full[:, R_sub:] = W_sub
            
            if _validate(U_full, V_full, W_full, n, m, p):
                if R_total < best_rank:
                    best_rank = R_total
                    best_U, best_V, best_W = U_full, V_full, W_full
                    if best_rank <= 20:
                        break
    
    # Strategy 2: Direct search on M_{2,4,5}
    if best_rank > 20:
        for R_direct in range(20, min(best_rank, 30)):
            if time.time() > deadline - 5:
                break
            time_left = deadline - time.time() - 5
            if time_left < 3:
                break
            
            U_d, V_d, W_d, _ = _try_rank(n, m, p, R_direct, 
                                         max_time=time_left * 0.5, num_restarts=4)
            if U_d is not None and R_direct < best_rank:
                best_rank = R_direct
                best_U, best_V, best_W = U_d, V_d, W_d
                if best_rank == 20:
                    break
    
    # Fallback to schoolbook
    if best_U is None:
        R = n * m * p
        U_sb = np.zeros((n * m, R))
        V_sb = np.zeros((m * p, R))
        W_sb = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = R

    # Snap to half-integers for output
    U_out = _snap_to_half_integer(best_U)
    V_out = _snap_to_half_integer(best_V)
    W_out = _snap_to_half_integer(best_W)
    
    # Verify snapped version; if invalid, use original
    if not _validate(U_out, V_out, W_out, n, m, p):
        U_out, V_out, W_out = best_U, best_V, best_W
    
    U_list = [[float(x) for x in row] for row in U_out.tolist()]
    V_list = [[float(x) for x in row] for row in V_out.tolist()]
    W_list = [[float(x) for x in row] for row in W_out.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
