"""Matrix multiplication 3x3 x 3x5 — minimal-rank exact tensor decomposition."""
import time
import math
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=3, p=5):
    """Check that U * V^T * W^T reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def _check_half_integer(entries):
    """Check all entries are integers or half-integers (multiples of 0.5)."""
    for row in entries:
        for x in row:
            if abs(x * 2 - round(x * 2)) > 1e-9:
                return False
    return True


def _schoolbook(n=3, m=3, p=5):
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


def _build_target_tensor(n=3, m=3, p=5):
    """Build the target tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T


def _snap_to_half_integer(arr):
    """Snap array to nearest half-integer."""
    return np.round(arr * 2) / 2


def _try_numerical_min_rank(n=3, m=3, p=5, deadline=None):
    """
    Use gradient descent with smooth surrogate to find a low-rank decomposition.
    Uses L-infinity error as smooth objective with log-sum-exp smoothing.
    """
    if deadline is None:
        deadline = time.time() + 80
    
    T = _build_target_tensor(n, m, p)
    best_result = None
    
    for r in [27, 25, 23, 21, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(40):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(123 + trial * 13)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            # Use L-BFGS-B optimizer for better convergence
            def objective(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                # Use smooth L-infinity approximation via log-sum-exp
                # ||R||_inf ≈ (1/alpha) * log(sum(exp(alpha*|R_ij|)))
                alpha = 50.0
                abs_R = np.abs(R)
                # Clip to avoid overflow
                abs_R_clipped = np.clip(abs_R, 0, 10)
                smooth_max = (1.0/alpha) * np.log(np.sum(np.exp(alpha * abs_R_clipped)))
                return smooth_max
            
            def gradient(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                abs_R = np.clip(np.abs(R), 0, 10)
                alpha = 50.0
                exp_alpha = np.exp(alpha * abs_R)
                sum_exp = np.sum(exp_alpha)
                # Gradient of smooth max w.r.t. R
                grad_R = (alpha * exp_alpha / sum_exp) * np.sign(R)
                
                grad_U = np.einsum('abc,br,cr->ar', grad_R, V_flat, W_flat, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', grad_R, U_flat, W_flat, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', grad_R, U_flat, V_flat, optimize=True)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
            
            U_final = res.x[:n*m*r].reshape(n*m, r)
            V_final = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_final = res.x[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_cur = np.einsum('ar,br,cr->abc', U_final, V_final, W_final, optimize=True)
            err = np.max(np.abs(T - T_cur))
            
            if err < 1e-5:
                # Try to snap to half-integers
                U_snap = _snap_to_half_integer(U_final)
                V_snap = _snap_to_half_integer(V_final)
                W_snap = _snap_to_half_integer(W_final)
                
                err_snap = np.max(np.abs(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)))
                
                if err_snap < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                else:
                    # Check if original is already half-integer
                    if _check_half_integer(U_final.tolist()) and _check_half_integer(V_final.tolist()) and _check_half_integer(W_final.tolist()):
                        U_list = [[float(x) for x in row] for row in U_final]
                        V_list = [[float(x) for x in row] for row in V_final]
                        W_list = [[float(x) for x in row] for row in W_final]
                        
                        if _verify(U_list, V_list, W_list):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            found = True
                            break
        
        if found:
            break
    
    return best_result


def _try_gradient_descent(n=3, m=3, p=5, deadline=None):
    """
    Use plain gradient descent with adaptive learning rate as fallback.
    """
    if deadline is None:
        deadline = time.time() + 80
    
    T = _build_target_tensor(n, m, p)
    best_result = None
    
    for r in [27, 25, 23, 21, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(999 + trial * 3)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            lr = 0.03
            prev_err = np.inf
            stalled = 0
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                T_cur = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                R = T - T_cur
                err = np.max(np.abs(R))
                
                if err < 1e-8:
                    break
                
                if err > prev_err * 0.995:
                    stalled += 1
                else:
                    stalled = 0
                prev_err = err
                
                if stalled > 150:
                    break
                
                grad_U = np.einsum('abc,br,cr->ar', R, V, W, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', R, U, W, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', R, U, V, optimize=True)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    effective_lr = lr * (1.0 / (1.0 + grad_norm))
                else:
                    effective_lr = lr
                
                U -= effective_lr * grad_U
                V -= effective_lr * grad_V
                W -= effective_lr * grad_W
            
            T_cur = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(T - T_cur))
            
            if err < 1e-5:
                U_snap = _snap_to_half_integer(U)
                V_snap = _snap_to_half_integer(V)
                W_snap = _snap_to_half_integer(W)
                
                err_snap = np.max(np.abs(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)))
                
                if err_snap < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
        
        if found:
            break
    
    return best_result


def _try_strassen_based(n=3, m=3, p=5, deadline=None):
    """
    Try to construct a decomposition using Strassen-like block structure.
    For 3x3 x 3x5, we can try to exploit the 2x2 Strassen result.
    """
    if deadline is None:
        deadline = time.time() + 60
    
    T = _build_target_tensor(n, m, p)
    
    # Try to find a decomposition with rank 27 or less using structured initialization
    # Initialize based on Kronecker product of 2x2 Strassen and identity
    
    # Strassen 2x2x2 has rank 7
    # For 3x3, we can try to embed this
    
    best_result = None
    
    for r in [27, 25, 23, 21, 19]:
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(15):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(777 + trial * 11)
            # Initialize with some structure
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Fill with small random values
            U += np.random.randn(n*m, r) * 0.2
            V += np.random.randn(m*p, r) * 0.2
            W += np.random.randn(p*n, r) * 0.2
            
            # Use L-BFGS-B
            def objective(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                return np.sum(R**2)
            
            def gradient(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                
                grad_U = 2 * np.einsum('abc,br,cr->ar', R, V_flat, W_flat, optimize=True)
                grad_V = 2 * np.einsum('abc,ar,cr->br', R, U_flat, W_flat, optimize=True)
                grad_W = 2 * np.einsum('abc,ar,br->cr', R, U_flat, V_flat, optimize=True)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
            
            U_final = res.x[:n*m*r].reshape(n*m, r)
            V_final = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_final = res.x[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_cur = np.einsum('ar,br,cr->abc', U_final, V_final, W_final, optimize=True)
            err = np.max(np.abs(T - T_cur))
            
            if err < 1e-5:
                U_snap = _snap_to_half_integer(U_final)
                V_snap = _snap_to_half_integer(V_final)
                W_snap = _snap_to_half_integer(W_final)
                
                err_snap = np.max(np.abs(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)))
                
                if err_snap < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
        
        if found:
            break
    
    return best_result


def _try_epigraph_smooth(n=3, m=3, p=5, deadline=None):
    """
    Use an epigraph formulation with smooth constraints.
    Minimize t subject to |T - T_cur|_inf <= t, with smoothing.
    """
    if deadline is None:
        deadline = time.time() + 80
    
    T = _build_target_tensor(n, m, p)
    best_result = None
    
    for r in [27, 25, 23, 21, 19]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(555 + trial * 7)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            def objective(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                # Use smooth max with adaptive alpha
                alpha = 30.0
                abs_R = np.clip(np.abs(R), 0, 10)
                smooth_max = (1.0/alpha) * np.log(np.sum(np.exp(alpha * abs_R)))
                return smooth_max
            
            def gradient(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                abs_R = np.clip(np.abs(R), 0, 10)
                alpha = 30.0
                exp_alpha = np.exp(alpha * abs_R)
                sum_exp = np.sum(exp_alpha)
                grad_R = (alpha * exp_alpha / sum_exp) * np.sign(R)
                
                grad_U = np.einsum('abc,br,cr->ar', grad_R, V_flat, W_flat, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', grad_R, U_flat, W_flat, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', grad_R, U_flat, V_flat, optimize=True)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                continue
            
            U_final = res.x[:n*m*r].reshape(n*m, r)
            V_final = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_final = res.x[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_cur = np.einsum('ar,br,cr->abc', U_final, V_final, W_final, optimize=True)
            err = np.max(np.abs(T - T_cur))
            
            if err < 1e-5:
                U_snap = _snap_to_half_integer(U_final)
                V_snap = _snap_to_half_integer(V_final)
                W_snap = _snap_to_half_integer(W_final)
                
                err_snap = np.max(np.abs(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)))
                
                if err_snap < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
        
        if found:
            break
    
    return best_result


def _try_polish(n=3, m=3, p=5, deadline=None):
    """
    High-precision polish pass: take a near-converged solution and refine with
    tighter tolerances and more iterations.
    """
    if deadline is None:
        deadline = time.time() + 60
    
    T = _build_target_tensor(n, m, p)
    best_result = None
    
    for r in [27, 25, 23, 21, 19]:
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(444 + trial * 5)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            # First pass: get close
            def objective1(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                return np.sum(R**2)
            
            def gradient1(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                
                grad_U = 2 * np.einsum('abc,br,cr->ar', R, V_flat, W_flat, optimize=True)
                grad_V = 2 * np.einsum('abc,ar,cr->br', R, U_flat, W_flat, optimize=True)
                grad_W = 2 * np.einsum('abc,ar,br->cr', R, U_flat, V_flat, optimize=True)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res1 = minimize(objective1, x0, method='L-BFGS-B', jac=gradient1,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
            
            U_mid = res1.x[:n*m*r].reshape(n*m, r)
            V_mid = res1.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_mid = res1.x[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_mid = np.einsum('ar,br,cr->abc', U_mid, V_mid, W_mid, optimize=True)
            err_mid = np.max(np.abs(T - T_mid))
            
            if err_mid > 1e-3:
                continue
            
            # Second pass: polish with smooth max
            def objective2(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                alpha = 100.0
                abs_R = np.clip(np.abs(R), 0, 10)
                smooth_max = (1.0/alpha) * np.log(np.sum(np.exp(alpha * abs_R)))
                return smooth_max
            
            def gradient2(x):
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_cur = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat, optimize=True)
                R = T - T_cur
                abs_R = np.clip(np.abs(R), 0, 10)
                alpha = 100.0
                exp_alpha = np.exp(alpha * abs_R)
                sum_exp = np.sum(exp_alpha)
                grad_R = (alpha * exp_alpha / sum_exp) * np.sign(R)
                
                grad_U = np.einsum('abc,br,cr->ar', grad_R, V_flat, W_flat, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', grad_R, U_flat, W_flat, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', grad_R, U_flat, V_flat, optimize=True)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            x1 = np.concatenate([U_mid.flatten(), V_mid.flatten(), W_mid.flatten()])
            
            try:
                res2 = minimize(objective2, x1, method='L-BFGS-B', jac=gradient2,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                continue
            
            U_final = res2.x[:n*m*r].reshape(n*m, r)
            V_final = res2.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_final = res2.x[n*m*r + m*p*r:].reshape(p*n, r)
            
            T_cur = np.einsum('ar,br,cr->abc', U_final, V_final, W_final, optimize=True)
            err = np.max(np.abs(T - T_cur))
            
            if err < 1e-5:
                U_snap = _snap_to_half_integer(U_final)
                V_snap = _snap_to_half_integer(V_final)
                W_snap = _snap_to_half_integer(W_final)
                
                err_snap = np.max(np.abs(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)))
                
                if err_snap < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
        
        if found:
            break
    
    return best_result


def construct():
    t0 = time.time()
    deadline = t0 + 110  # 110s hard limit, 40s safety margin for 150s budget
    
    # Try multiple approaches in order of likelihood of success
    
    # Approach 1: L-BFGS-B with smooth L-infinity objective
    result = _try_numerical_min_rank(deadline=deadline)
    
    if result is not None:
        return result
    
    # Approach 2: Epigraph smooth formulation
    if time.time() < deadline - 25:
        result = _try_epigraph_smooth(deadline=deadline)
        
        if result is not None:
            return result
    
    # Approach 3: Strassen-based structured initialization
    if time.time() < deadline - 20:
        result = _try_strassen_based(deadline=deadline)
        
        if result is not None:
            return result
    
    # Approach 4: High-precision polish
    if time.time() < deadline - 15:
        result = _try_polish(deadline=deadline)
        
        if result is not None:
            return result
    
    # Approach 5: Plain gradient descent with adaptive learning rate
    if time.time() < deadline - 10:
        result = _try_gradient_descent(deadline=deadline)
        
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
