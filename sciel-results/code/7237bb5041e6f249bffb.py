"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from collections import deque

DIMENSIONS = [2, 5, 6]

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using tensor structure."""
    r = U.shape[1]
    # Check all 60 non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for rr in range(r):
                    val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                if abs(val - 1.0) > tol:
                    return False
    # Check sample of zero entries
    rng = np.random.RandomState(42)
    nm, mp, pn = n*m, m*p, p*n
    for _ in range(300):
        a = rng.randint(nm)
        b = rng.randint(mp)
        c = rng.randint(pn)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for rr in range(r):
            val += U[a, rr] * V[b, rr] * W[c, rr]
        if abs(val) > tol:
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Full exact verification."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for rr in range(r):
                    val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                if abs(val - 1.0) > 1e-10:
                    return False
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = 0.0
                for rr in range(r):
                    val += U[a, rr] * V[b, rr] * W[c, rr]
                if abs(val) > 1e-10:
                    return False
    return True

def _loss_and_grad(U, V, W, T, n, m, p):
    """Compute loss and gradients for tensor decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    # Compute reconstruction
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    R = T - recon
    loss = np.sum(R**2)
    # Gradients
    grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
    grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
    grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
    return loss, grad_U, grad_V, grad_W

def _gradient_descent(U, V, W, T, n, m, p, max_steps=80, lr=0.01, deadline=None, tol=1e-10):
    """Gradient descent with adaptive learning rate."""
    nm, mp, pn = n*m, m*p, p*n
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    momentum = 0.9
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        vU = momentum * vU + (1 - momentum) * gU
        vV = momentum * vV + (1 - momentum) * gV
        vW = momentum * vW + (1 - momentum) * gW
        
        U = U - lr * vU
        V = V - lr * vV
        W = W - lr * vW
        
        if loss > best_loss * 1.5:
            lr *= 0.7
            vU, vV, vW = 0, 0, 0
    
    return best_U, best_V, best_W, best_loss

def _als_polish(U, V, W, T, n, m, p, iterations=25, deadline=None, tol=1e-10):
    """ALS polishing with deadline guard."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for it in range(iterations):
        if deadline and time.time() > deadline:
            break
        
        # Update U
        for a in range(nm):
            A = (V * W[np.newaxis, :]).T.reshape(-1, r)
            T_a = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(A.T, T_a, rcond=None)
            except:
                pass
        
        # Update V
        for b in range(mp):
            A = (U * W[np.newaxis, :]).T.reshape(-1, r)
            T_b = T[:, b, :].flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(A.T, T_b, rcond=None)
            except:
                pass
        
        # Update W
        for c in range(pn):
            A = (U * V[np.newaxis, :]).T.reshape(-1, r)
            T_c = T[:, :, c].flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(A.T, T_c, rcond=None)
            except:
                pass
    
    return U, V, W

def _structured_init(n, m, p, rank, seed=42):
    """Structured initialization exploiting block structure."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    if rank >= m:
        terms_per_j = rank // m
        remainder = rank % m
        col = 0
        for j in range(m):
            n_terms = terms_per_j + (1 if j < remainder else 0)
            for t in range(n_terms):
                for i in range(n):
                    U[i*m+j, col] = rng.randn()
                for k in range(p):
                    V[j*p+k, col] = rng.randn()
                    W[k*n+i, col] = rng.randn()
                col += 1
    else:
        scale = 1.0 / np.sqrt(rank)
        U = rng.randn(nm, rank) * scale
        V = rng.randn(mp, rank) * scale
        W = rng.randn(pn, rank) * scale
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                U, V, W = _structured_init(n, m, p, rank, seed=42 + restart * 13)
            elif restart % 3 == 1:
                scale = 1.0 / np.sqrt(rank)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            else:
                scale = 1.0 / np.sqrt(rank * 0.5)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
                mask_U = np.random.rand(nm, rank) < 0.6
                mask_V = np.random.rand(mp, rank) < 0.6
                mask_W = np.random.rand(pn, rank) < 0.6
                U[mask_U] = 0
                V[mask_V] = 0
                W[mask_W] = 0
            
            # Phase 1: GD
            U, V, W, loss = _gradient_descent(U, V, W, T, n, m, p, max_steps=50, lr=0.015, deadline=deadline - 15)
            
            # Phase 2: ALS polish if close
            if loss < 2.0:
                U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=20, deadline=deadline - 8)
                
                # Verify
                if _verify_fast(U, V, W, n, m, p, tol=1e-7):
                    # Confirm with exact verification
                    if _verify_exact(U, V, W, n, m, p):
                        found = True
                        best_rank = rank
                        best_result = (U, V, W)
                        break
            
            # Phase 3: Second GD pass if still close
            if loss < 10.0 and not found:
                U2, V2, W2, loss2 = _gradient_descent(U, V, W, T, n, m, p, max_steps=30, lr=0.005, deadline=deadline - 8)
                if loss2 < 0.5:
                    U2, V2, W2 = _als_polish(U2, V2, W2, T, n, m, p, iterations=15, deadline=deadline - 5)
                    if _verify_fast(U2, V2, W2, n, m, p, tol=1e-7):
                        if _verify_exact(U2, V2, W2, n, m, p):
                            found = True
                            best_rank = rank
                            best_result = (U2, V2, W2)
                            break
        
        if found:
            break
    
    # Final polish on best result
    U_final, V_final, W_final = best_result
    if time.time() < deadline - 15:
        U_final, V_final, W_final = _als_polish(U_final, V_final, W_final, T, n, m, p, iterations=30, deadline=deadline - 5)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
