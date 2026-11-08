import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matmul tensor for (n,m,p)."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to the nearest half-integer."""
    return np.round(M * 2) / 2

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of the loss function L = 0.5 * sum (S_ijk - 1)^2."""
    rank = U.shape[1]
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Precompute S_ijk? No, compute on the fly or store.
    # S_ijk = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # To be efficient, we can loop over i, j, k and update gradients for all r at once.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # S_ijk
                S_ijk = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                err = S_ijk - 1.0
                
                # dL/dU[u_idx, r] = err * V[v_idx, r] * W[w_idx, r]
                # dL/dV[v_idx, r] = err * U[u_idx, r] * W[w_idx, r]
                # dL/dW[w_idx, r] = err * U[u_idx, r] * V[v_idx, r]
                
                prod_VW = V[v_idx, :] * W[w_idx, :]
                prod_UW = U[u_idx, :] * W[w_idx, :]
                prod_UV = U[u_idx, :] * V[v_idx, :]
                
                grad_U[u_idx, :] += err * prod_VW
                grad_V[v_idx, :] += err * prod_UW
                grad_W[w_idx, :] += err * prod_UV
                
    return grad_U, grad_V, grad_W

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
        
        np.random.seed(restart * 100 + target_rank)
        U = np.random.randn(*U_shape) * 0.5
        V = np.random.randn(*V_shape) * 0.5
        W = np.random.randn(*W_shape) * 0.5
        
        lr = 0.01
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        
        for it in range(1000):
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
            
            # Check loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        S_ijk = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        loss += (S_ijk - 1.0)**2
                        
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_loss < 1e-12:
                # Found a solution
                # Try snapping to half-integers
                U_snap = _snap_to_half_integer(best_U)
                V_snap = _snap_to_half_integer(best_V)
                W_snap = _snap_to_half_integer(best_W)
                
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap, target_rank
                else:
                    # Check if original floats are valid (within tolerance)
                    # The prompt asks for integer/half-integer entries.
                    # If snapped is not valid, maybe the solution is not half-integer?
                    # But the prompt implies the optimal solution has them.
                    # If snapped is not valid, we might need to refine or accept float if allowed?
                    # "entries that exactly decompose... integer/half-integer entries"
                    # So we MUST return half-integers.
                    # If the continuous solution is close but snapping breaks it, 
                    # we might need to search locally in the half-integer grid.
                    pass
                    
            # Adaptive learning rate
            if it % 50 == 0:
                # If loss is not decreasing, reduce lr
                # Simple heuristic
                pass
                
    return None, None, None, best_loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy 1: Solve sub-problem M_{2,2,5} and duplicate
    # If Rank(M_{2,2,5}) = R_sub, then Rank(M_{2,4,5}) <= 2 * R_sub
    # Lower bound for M_{2,2,5} is 10.
    
    best_rank = n * m * p # 40
    best_U, best_V, best_W = None, None, None
    
    # Try to find decomposition for M_{2,2,5}
    n_sub, m_sub, p_sub = 2, 2, 5
    
    # We will try ranks from 10 upwards
    for R_sub in range(10, 21):
        if time.time() > deadline:
            break
        
        # Allocate time for this rank
        # If we are close to deadline, skip
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        U_sub, V_sub, W_sub, status = _try_rank(n_sub, m_sub, p_sub, R_sub, max_time=time_left * 0.5, num_restarts=5)
        
        if U_sub is not None:
            # Construct full M_{2,4,5}
            R_total = 2 * R_sub
            
            U_full = np.zeros((n * m, R_total))
            V_full = np.zeros((m * p, R_total))
            W_full = np.zeros((p * n, R_total))
            
            # First half: sub-problem 1 (j=0,1)
            # U_sub rows: i*m_sub + j_sub. Global U rows: i*m + j.
            # For j_sub in {0,1}, global j = j_sub.
            for i in range(n_sub):
                for j_sub in range(m_sub):
                    U_full[i * m + j_sub, :R_sub] = U_sub[i * m_sub + j_sub, :]
            
            # V_sub rows: j_sub*p_sub + k. Global V rows: j*p + k.
            # For j_sub in {0,1}, global j = j_sub.
            for j_sub in range(m_sub):
                for k in range(p_sub):
                    V_full[j_sub * p + k, :R_sub] = V_sub[j_sub * p_sub + k, :]
            
            # W_sub rows: k*p_sub? No, W index is k*n + i.
            # For M_{2,2,5}, W shape is (p*n, R) = (10, R).
            # Indices k in 0..4, i in 0..1. Row = k*2 + i.
            # For M_{2,4,5}, W shape is (p*n, R) = (10, R).
            # Indices k in 0..4, i in 0..1. Row = k*2 + i.
            # So W_full[:, :R_sub] = W_sub
            W_full[:, :R_sub] = W_sub
            
            # Second half: sub-problem 2 (j=2,3)
            # Map j_sub in {0,1} to global j in {2,3}
            for i in range(n_sub):
                for j_sub in range(m_sub):
                    global_j = j_sub + 2
                    U_full[i * m + global_j, R_sub:] = U_sub[i * m_sub + j_sub, :]
            
            for j_sub in range(m_sub):
                global_j = j_sub + 2
                for k in range(p_sub):
                    V_full[global_j * p + k, R_sub:] = V_sub[j_sub * p_sub + k, :]
            
            # W is same structure, but values can be different?
            # In the decomposition C = A1 B1 + A2 B2, the W terms for the second part
            # are independent. We can reuse the same W_sub or use a different one.
            # Since the problem is symmetric, we can just reuse W_sub.
            W_full[:, R_sub:] = W_sub
            
            # Verify
            if _validate(U_full, V_full, W_full, n, m, p):
                if R_total < best_rank:
                    best_rank = R_total
                    best_U = U_full
                    best_V = V_full
                    best_W = W_full
                    # If we hit the lower bound (20), we can stop
                    if best_rank == 20:
                        break
            else:
                # Validation failed, maybe snapping issue?
                # Try to refine? For now, skip.
                pass

    # Strategy 2: Direct search on M_{2,4,5} if sub-problem didn't yield 20
    if best_rank > 20:
        for R_direct in range(20, best_rank):
            if time.time() > deadline:
                break
            time_left = deadline - time.time()
            if time_left < 5:
                break
            
            U_d, V_d, W_d, status = _try_rank(n, m, p, R_direct, max_time=time_left * 0.5, num_restarts=3)
            if U_d is not None:
                if R_direct < best_rank:
                    best_rank = R_direct
                    best_U = U_d
                    best_V = V_d
                    best_W = W_d
                    if best_rank == 20:
                        break

    # Fallback to schoolbook if nothing better found
    if best_U is None:
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    # Convert to lists of half-integers
    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
