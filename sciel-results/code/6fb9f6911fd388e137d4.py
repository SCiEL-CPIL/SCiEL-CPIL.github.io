"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
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
    
    # Try to find a lower rank solution
    # For 2x3x4, the known rank is 21 (from literature)
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
    Attempt to find a rank-`rank` decomposition using structured initialization
    and gradient-based optimization with snapping to half-integers.
    """
    np.random.seed(42)
    
    # Get all non-zero indices in T (the "support" of the tensor)
    nz = np.argwhere(T != 0)
    # nz has shape (24, 3) for 2x3x4
    
    # Strategy: Initialize by selecting 'rank' terms from the schoolbook set
    # and then optimize. For rank < 24, we need to merge some terms.
    
    # Multiple restarts with different initializations
    for restart in range(20):
        if time.time() > deadline:
            return None
            
        # Different initialization strategies
        if restart == 0:
            # Random selection from support
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
        else:
            # Perturbed initialization: start from schoolbook and merge
            # Randomly perturb the schoolbook solution
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Use first 'rank' schoolbook terms as base
            for r in range(min(rank, len(nz))):
                a, b, d = nz[r]
                U[a, r] = 1
                V[b, r] = 1
                W[d, r] = 1
            
            # Add small random perturbations to encourage merging
            U += np.random.randn(nm, rank) * 0.1
            V += np.random.randn(mp, rank) * 0.1
            W += np.random.randn(pn, rank) * 0.1
        
        # Optimize using gradient descent with adaptive learning rate
        result = _optimize(U, V, W, T, nm, mp, pn, deadline)
        if result is not None:
            return result
    
    return None

def _optimize(U, V, W, T, nm, mp, pn, deadline):
    """Optimize U, V, W to minimize ||sum U*V*W - T||^2 and snap to half-integers."""
    lr = 0.05
    prev_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    for it in range(300):
        if time.time() > deadline:
            return None
            
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        residual = tensor - T
        loss = np.sum(residual**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if loss < 1e-12:
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
        elif it % 30 == 0:
            lr = min(lr * 1.1, 0.2)
            
        prev_loss = loss
    
    # Final check and snapping to half-integers
    for U_try, V_try, W_try in [(best_U, best_V, best_W), (U, V, W)]:
        final_tensor = np.einsum('ac,bc,dc->abd', U_try, V_try, W_try)
        if np.sum((final_tensor - T)**2) < 1e-8:
            # Snap to half-integers
            U_snap = np.round(U_try * 2) / 2
            V_snap = np.round(V_try * 2) / 2
            W_snap = np.round(W_try * 2) / 2
            if _verify(T, U_snap, V_snap, W_snap):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            if _verify(T, U_try, V_try, W_try):
                return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    return None

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)

def _try_rank_fast(T, nm, mp, pn, rank, deadline):
    """Backward-compatible wrapper."""
    return _try_rank_structured(T, nm, mp, pn, rank, deadline)
# EVOLVE-BLOCK-END
