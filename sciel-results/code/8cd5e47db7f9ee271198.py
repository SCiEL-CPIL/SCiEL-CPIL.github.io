import time
import numpy as np
from scipy.optimize import minimize

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

def _loss_and_grad(x, n, m, p, rank):
    """
    Compute loss and gradient for L-BFGS-B.
    x is flattened [U, V, W].
    """
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = x[:U_size].reshape(n * m, rank)
    V = x[U_size:U_size+V_size].reshape(m * p, rank)
    W = x[U_size+V_size:].reshape(p * n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    E = S - 1.0
    loss = 0.5 * np.sum(E**2)
    
    # Gradients
    grad_U_r = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
    grad_V_r = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
    grad_W_r = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
    
    grad_flat = np.concatenate([
        grad_U_r.reshape(-1),
        grad_V_r.reshape(-1),
        grad_W_r.reshape(-1)
    ])
    
    return loss, grad_flat

def _try_rank_lbfgs(n, m, p, target_rank, max_time, num_restarts=20):
    """Try to find a decomposition of the given rank using L-BFGS-B."""
    start_time = time.time()
    
    U_size = n * m * target_rank
    V_size = m * p * target_rank
    W_size = p * n * target_rank
    
    best_result = None
    best_loss = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization
        np.random.seed(restart * 100 + target_rank)
        # Scale 0.5 is a good start for these problems
        U_flat = np.random.randn(U_size) * 0.5
        V_flat = np.random.randn(V_size) * 0.5
        W_flat = np.random.randn(W_size) * 0.5
        
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        # L-BFGS-B options
        res = minimize(
            _loss_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            args=(n, m, p, target_rank),
            options={
                'maxiter': 100,
                'maxfun': 2000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxls': 50
            }
        )
        
        # Check if this result is better
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res.x.copy()
            
            # Early exit if we found a very good solution
            if best_loss < 1e-10:
                break
                
    if best_result is None:
        return None, None, None, float('inf')
        
    # Extract best U, V, W
    U = best_result[:U_size].reshape(n * m, target_rank)
    V = best_result[U_size:U_size+V_size].reshape(m * p, target_rank)
    W = best_result[U_size+V_size:].reshape(p * n, target_rank)
    
    return U, V, W, best_loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_rank = n * m * p  # 40
    best_U, best_V, best_W = None, None, None
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 20
    # Try ranks from 20 upwards
    # We prioritize lower ranks as they are better scores.
    
    for target_rank in range(20, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        time_left = deadline - time.time() - 5
        if time_left < 2:
            break
            
        # Allocate time for this rank attempt
        # Give more time to lower ranks as they are harder to find if they exist
        if target_rank <= 22:
            max_time_for_rank = min(time_left * 0.5, 40)
            num_restarts = 20
        else:
            max_time_for_rank = min(time_left * 0.3, 20)
            num_restarts = 10
            
        U, V, W, loss = _try_rank_lbfgs(n, m, p, target_rank, max_time_for_rank, num_restarts)
        
        if U is not None and loss < 1e-8:
            # Try to snap to half-integers
            U_snap = _snap_to_half_integer(U)
            V_snap = _snap_to_half_integer(V)
            W_snap = _snap_to_half_integer(W)
            
            if _validate(U_snap, V_snap, W_snap, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                break # Found minimal rank, stop searching higher ranks
            
            # If snap fails, check if the float version is valid?
            # The prompt requires integer/half-integer entries.
            # If the float solution is exact but not half-integer, it's not a valid output per contract.
            # So we continue to next rank.
            
    # Fallback to schoolbook if no better rank found
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

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
