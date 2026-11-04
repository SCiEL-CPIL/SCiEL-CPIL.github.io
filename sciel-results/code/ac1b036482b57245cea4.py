import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check if entries are half-integers
    if not np.allclose(U_np * 2, np.round(U_np * 2), atol=1e-9):
        return False
    if not np.allclose(V_np * 2, np.round(V_np * 2), atol=1e-9):
        return False
    if not np.allclose(W_np * 2, np.round(W_np * 2), atol=1e-9):
        return False
        
    # Check tensor reconstruction
    T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
        
    return True

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    """Try to find a decomposition of target_rank using Gradient Descent."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # We will try multiple random initializations
    # If we find a solution, return it.
    
    for attempt in range(50):
        if time.time() > time_limit:
            break
            
        # Initialize
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        lr = 0.05
        for it in range(1000):
            if time.time() > time_limit:
                break
                
            # Compute T_recon
            T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
            E = T_recon - T_target
            
            # Gradients
            grad_U = np.einsum('uvw,vc,wc->uc', E, V, W)
            grad_V = np.einsum('uvw,uc,wc->vc', E, U, W)
            grad_W = np.einsum('uvw,uc,vc->wc', E, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check for solution periodically
            if it % 20 == 0:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    
                # If error is very small, try to polish
                err = np.linalg.norm(E)
                if err < 1e-4:
                    # Try a few local steps with smaller lr or jitter
                    for _ in range(5):
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            return U_r.tolist(), V_r.tolist(), W_r.tolist()
                        U += np.random.randn(nm, target_rank) * 0.01
                        V += np.random.randn(mp, target_rank) * 0.01
                        W += np.random.randn(pn, target_rank) * 0.01
                    break # Restart

    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a better decomposition
    # Start from rank 27 up to 35
    for r in range(27, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        result = _try_find_decomposition(n, m, p, r, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
