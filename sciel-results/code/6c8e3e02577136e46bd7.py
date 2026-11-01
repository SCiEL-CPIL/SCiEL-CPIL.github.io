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
    
    deadline = time.time() + 115  # 115s budget, leave 5s margin before 120s limit
    
    # Try ranks from schoolbook-1 down
    for rank in range(schoolbook_rank - 1, 6, -1):
        if time.time() > deadline:
            break
        result = _try_rank_improved(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            break  # Found lower rank, stop
            
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

def _try_rank_improved(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition of T using improved random search."""
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    # More trials with adaptive step size
    num_tries = 20
    
    for trial in range(num_tries):
        if time.time() > deadline:
            return None
        
        # Diverse random initialization
        scale = 0.1 + 0.05 * (trial % 5)
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        # Adaptive gradient descent with line search
        lr = 0.1
        prev_loss = np.inf
        for it in range(200):
            if time.time() > deadline:
                return None
            
            tensor = np.einsum('ac,bc,dc->abd', U, V, W)
            residual = tensor - T
            loss = np.sum(residual**2)
            
            # Adaptive learning rate: reduce if loss increases
            if loss > prev_loss:
                lr *= 0.5
                # Undo step
                U = U + lr * np.einsum('abd,bc,dc->ac', residual, V, W)
                V = V + lr * np.einsum('abd,ac,dc->bc', residual, U, W)
                W = W + lr * np.einsum('abd,ac,bc->dc', residual, U, V)
            else:
                # Increase lr slightly if making good progress
                if it % 10 == 0 and loss < 0.9 * prev_loss:
                    lr = min(lr * 1.1, 0.5)
            
            dU = np.einsum('abd,bc,dc->ac', residual, V, W)
            dV = np.einsum('abd,ac,dc->bc', residual, U, W)
            dW = np.einsum('abd,ac,bc->dc', residual, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            prev_loss = loss
            
            if loss < 1e-12:
                break
        
        # Check if converged
        final_tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        if np.sum((final_tensor - T)**2) < 1e-10:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify(T, U_snap, V_snap, W_snap):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            # Also try without snapping in case rounding broke it
            if _verify(T, U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)

def _try_rank_fast(T, nm, mp, pn, rank, deadline):
    """Backward-compatible wrapper."""
    return _try_rank_improved(T, nm, mp, pn, rank, deadline)
# EVOLVE-BLOCK-END
