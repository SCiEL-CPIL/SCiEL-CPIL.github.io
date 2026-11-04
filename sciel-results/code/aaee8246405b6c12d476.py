"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _snap_to_rationals(arr, max_denom=16):
    """Snap values to nearby rationals with small denominators."""
    snapped = arr.copy()
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        if abs(val) < 1e-10:
            snapped[idx] = 0.0
            continue
        best_frac = val
        best_err = abs(val - best_frac)
        for d in range(1, max_denom + 1):
            num = round(val * d)
            if abs(val - num / d) < best_err:
                best_err = abs(val - num / d)
                best_frac = num / d
        if best_err < 1e-6:
            snapped[idx] = best_frac
    return snapped

def _als_solve(U, V, W, T, n, m, p, max_iters=20, tol=1e-12):
    """Fast ALS polish using Kronecker products."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for it in range(max_iters):
        # Solve for U: T_mat = (V ⊗ W)^T @ U.T
        VW = np.kron(W.T, V.T).reshape(pn * mp, r)
        T_mat = T.reshape(nm, mp * pn).T
        try:
            U_new = np.linalg.lstsq(VW, T_mat, rcond=None)[0].T
        except:
            break
        
        # Solve for V: T_mat = (U ⊗ W)^T @ V.T
        UW = np.kron(W.T, U_new.T).reshape(pn * nm, r)
        T_mat_v = T.reshape(mp, nm * pn).T
        try:
            V_new = np.linalg.lstsq(UW, T_mat_v, rcond=None)[0].T
        except:
            break
        
        # Solve for W: T_mat = (U ⊗ V)^T @ W.T
        UV = np.kron(V_new.T, U_new.T).reshape(mp * nm, r)
        T_mat_w = T.reshape(pn, nm * mp).T
        try:
            W_new = np.linalg.lstsq(UV, T_mat_w, rcond=None)[0].T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        if _verify_fast(U, V, W, T, tol=1e-10):
            break
    
    return U, V, W

def _gradient_descent(U, V, W, T, n, m, p, rank, max_steps=200, lr_init=0.05, deadline=None):
    """Adam gradient descent with early stopping."""
    nm, mp, pn = n*m, m*p, p*n
    m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    beta1, beta2 = 0.9, 0.999
    lr = lr_init
    t = 0
    best_loss = np.inf
    best_U_c, best_V_c, best_W_c = U.copy(), V.copy(), W.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U_c, best_V_c, best_W_c = U.copy(), V.copy(), W.copy()
            if loss < 1e-14:
                break
        
        gU = -np.einsum('abc,br,cr->ar', R, V, W)
        gV = -np.einsum('abc,ar,cr->br', R, U, W)
        gW = -np.einsum('abc,ar,br->cr', R, U, V)
        
        t += 1
        for i, (g, param) in enumerate(zip([gU, gV, gW], [U, V, W])):
            m_[i] = beta1 * m_[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)
            m_hat = m_[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        if step % 50 == 0 and step > 0:
            lr *= 0.8
        
        if step % 30 == 0 and loss < 1e-10:
            if _verify_fast(U, V, W, T, tol=1e-10):
                return U, V, W, True, best_loss
    
    return best_U_c, best_V_c, best_W_c, False, best_loss

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 30 up to 59
    for rank in range(30, best_rank):
        if time.time() > deadline - 40:
            break
        
        max_restarts = max(3, 15 - rank // 3)
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + restart * 7919 + rank * 13)
            
            # Structured init: mix of random and structured
            if restart % 3 == 0:
                # Pure random
                U = np.random.randn(nm, rank) * 0.1
                V = np.random.randn(mp, rank) * 0.1
                W = np.random.randn(pn, rank) * 0.1
            elif restart % 3 == 1:
                # Semi-structured: initialize based on tensor structure
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                for r_idx in range(rank):
                    idx = min(r_idx * len(triples) // rank, len(triples) - 1)
                    i, j, k = triples[idx]
                    U[i*m+j, r_idx] = 1.0
                    V[j*p+k, r_idx] = 1.0
                    W[k*n+i, r_idx] = 1.0
                U += np.random.randn(nm, rank) * 0.05
                V += np.random.randn(mp, rank) * 0.05
                W += np.random.randn(pn, rank) * 0.05
            else:
                # Sparse random
                U = np.random.randn(nm, rank) * 0.1
                V = np.random.randn(mp, rank) * 0.1
                W = np.random.randn(pn, rank) * 0.1
                mask_u = np.random.rand(nm, rank) < 0.3
                mask_v = np.random.rand(mp, rank) < 0.3
                mask_w = np.random.rand(pn, rank) < 0.3
                U *= mask_u
                V *= mask_v
                W *= mask_w
            
            # Gradient descent
            sub_deadline = min(deadline - 15, time.time() + 30)
            U, V, W, gd_found, final_loss = _gradient_descent(
                U, V, W, T, n, m, p, rank, max_steps=300, lr_init=0.05, deadline=sub_deadline
            )
            
            if gd_found:
                found = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            
            # Try ALS polish if GD got close
            if final_loss < 1e-6:
                U, V, W = _als_solve(U, V, W, T, n, m, p, max_iters=10)
                if _verify_fast(U, V, W, T, tol=1e-10):
                    found = True
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            # Try rational snapping if very close
            if final_loss < 1e-8:
                U_snap = _snap_to_rationals(U, max_denom=8)
                V_snap = _snap_to_rationals(V, max_denom=8)
                W_snap = _snap_to_rationals(W, max_denom=8)
                if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-10):
                    found = True
                    best_rank = rank
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    break
        
        if found:
            break
    
    # Final polish on best result
    if best_rank < rank_sb:
        best_U, best_V, best_W = _als_solve(best_U, best_V, best_W, T, n, m, p, max_iters=5)
        # Try rational snapping
        U_snap = _snap_to_rationals(best_U, max_denom=16)
        V_snap = _snap_to_rationals(best_V, max_denom=16)
        W_snap = _snap_to_rationals(best_W, max_denom=16)
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-10):
            best_U, best_V, best_W = U_snap, V_snap, W_snap
    
    U_final, V_final, W_final = best_U, best_V, best_W
    return {
        "U": [[float(x) for x in row] for row in U_final],
        "V": [[float(x) for x in row] for row in V_final],
        "W": [[float(x) for x in row] for row in W_final]
    }
