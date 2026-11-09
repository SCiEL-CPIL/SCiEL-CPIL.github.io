"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

N, M, P = 3, 4, 6
DEADLINE = 145.0  # safety margin before 150s kill

def verify_fast(U, V, W, rank):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)

def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def round_to_half(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(np.array(U) * 2) / 2
    V_r = np.round(np.array(V) * 2) / 2
    W_r = np.round(np.array(W) * 2) / 2
    return U_r, V_r, W_r

def try_remove_rank(seed=42):
    """
    Try to reduce the schoolbook rank by 1 by removing one term
    and optimizing the rest.
    """
    n, m, p = N, M, P
    rank_full = n * m * p
    rank_target = rank_full - 1
    
    # Get schoolbook matrices
    sb = schoolbook()
    U_full = np.array(sb["U"])
    V_full = np.array(sb["V"])
    W_full = np.array(sb["W"])
    
    # We will try to remove one specific column (term)
    # Which one? Randomly or systematically.
    # Let's try removing a term that might be "redundant".
    # In 2x2x2, removing any one term and optimizing the rest with Strassen-like values works.
    # Here, we remove term 'idx' and optimize the remaining 'rank_target' terms.
    
    start = time.time()
    
    for idx_to_remove in range(rank_full):
        if time.time() - start > 10: # Time limit per removal attempt
            break
            
        # Create initial guess for the reduced rank
        # We take the full matrices and remove the column idx_to_remove
        mask = [i for i in range(rank_full) if i != idx_to_remove]
        
        U0 = U_full[:, mask]
        V0 = V_full[:, mask]
        W0 = W_full[:, mask]
        
        # Flatten for optimization
        def pack(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unpack(x):
            size_U = n * m * rank_target
            size_V = m * p * rank_target
            U = x[:size_U].reshape(n, m, rank_target)
            V = x[size_U:size_U + size_V].reshape(m, p, rank_target)
            W = x[size_U + size_V:].reshape(p, n, rank_target)
            return U, V, W

        def objective_and_grad(x):
            U, V, W = unpack(x)
            # Note: unpack returns 3D arrays directly for einsum
            # But we need to be careful with reshaping if we flatten.
            # Let's stick to the previous unpack which returns 2D and reshape inside.
            # Wait, unpack above returns 3D. Let's adjust objective to use 3D.
            
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            res = T - 1.0
            val = np.sum(res * res)
            
            grad_U = np.einsum('ijk,jkc,kic->ijc', res, V, W)
            grad_V = np.einsum('ijk,ijc,kic->jkc', res, U, W)
            grad_W = np.einsum('ijk,ijc,jkc->kic', res, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return val, grad

        x0 = pack(U0, V0, W0)
        
        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e-6:
                U_opt, V_opt, W_opt = unpack(res.x)
                # Reshape to 2D for rounding and verification
                U_2d = U_opt.reshape(n * m, rank_target)
                V_2d = V_opt.reshape(m * p, rank_target)
                W_2d = W_opt.reshape(p * n, rank_target)
                
                U_r, V_r, W_r = round_to_half(U_2d, V_2d, W_2d)
                
                if verify_fast(U_r, V_r, W_r, rank_target):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
        except Exception:
            continue
            
    return None

def construct():
    start_time = time.time()
    best = schoolbook()
    
    # Try to find rank 71
    result = try_remove_rank()
    if result is not None:
        best = result
        # Check time, if we have plenty, try for 70?
        # For now, 71 is a solid improvement over 72.
        
    # Verify best
    rank = len(best["U"][0])
    if not verify_fast(best["U"], best["V"], best["W"], rank):
        best = schoolbook()
        
    return best
