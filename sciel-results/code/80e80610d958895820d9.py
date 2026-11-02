"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 115  # 115s budget, leave 5s margin before 120s limit
    
    # Try to find a lower rank solution, specifically targeting rank 21
    # We know 2x3x4 has rank 21.
    target_ranks = [21, 22, 23]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        result = _try_rank_structured(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid lower rank, stop
            
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_structured(T, nm, mp, pn, rank, deadline):
    """
    Attempt to find a rank-`rank` decomposition using a structured search
    initialized from the schoolbook solution.
    """
    # Get all non-zero indices in T
    nz = np.argwhere(T != 0)
    # nz has shape (24, 3)
    
    np.random.seed(42)
    
    # Try multiple random initializations
    num_attempts = 10
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
            
        # Randomly select 'rank' indices to initialize
        if len(nz) > rank:
            selected = np.random.choice(len(nz), rank, replace=False)
        else:
            selected = np.arange(len(nz))
            
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for r in range(rank):
            idx = selected[r]
            a, b, d = nz[idx]
            U[a, r] = 1
            V[b, r] = 1
            W[d, r] = 1
            
        # Optimize U, V, W to minimize || sum U V W - T ||^2
        # Use gradient descent with adaptive learning rate
        U_opt, V_opt, W_opt, final_loss = _optimize_tensor(T, U, V, W, deadline)
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            
        if best_loss < 1e-10:
            break
    
    if best_U is None:
        return None
        
    # Final check and snapping
    final_tensor = np.einsum('ac,bc,dc->abd', best_U, best_V, best_W)
    if np.sum((final_tensor - T)**2) < 1e-8:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        if _verify(T, U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        if _verify(T, best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
    return None

def _optimize_tensor(T, U, V, W, deadline):
    """
    Optimize U, V, W to minimize || sum U V W - T ||^2 using gradient descent.
    Returns (U, V, W, final_loss).
    """
    lr = 0.01
    prev_loss = np.inf
    
    for it in range(500):
        if time.time() > deadline:
            tensor = np.einsum('ac,bc,dc->abd', U, V, W)
            residual = tensor - T
            return U, V, W, np.sum(residual**2)
            
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        residual = tensor - T
        loss = np.sum(residual**2)
        
        if loss < 1e-10:
            break
            
        # Compute gradients
        dU = np.einsum('abd,bc,dc->ac', residual, V, W)
        dV = np.einsum('abd,ac,dc->bc', residual, U, W)
        dW = np.einsum('abd,ac,bc->dc', residual, U, V)
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive LR
        if loss > prev_loss:
            lr *= 0.5
        elif it % 50 == 0:
            lr = min(lr * 1.1, 0.1)
            
        prev_loss = loss
        
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    residual = tensor - T
    return U, V, W, np.sum(residual**2)

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)

def _try_rank_fast(T, nm, mp, pn, rank, deadline):
    """Backward-compatible wrapper."""
    return _try_rank_structured(T, nm, mp, pn, rank, deadline)
# EVOLVE-BLOCK-END
