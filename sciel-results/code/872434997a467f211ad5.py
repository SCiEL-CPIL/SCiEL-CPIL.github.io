"""Tensor decomposition search for 2x4 x 4x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the full multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if the decomposition exactly reproduces the tensor."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T_recon - T))
    return max_err < tol, max_err

def _snap_to_half_integer(x, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    half = 2.0 * x
    rounded = np.round(half)
    if np.all(np.abs(half - rounded) < tol):
        return rounded / 2.0
    return None

def _verify_exact(U, V, W, n, m, p):
    """Verify the decomposition exactly using fractions."""
    T_frac = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_frac[(a, b, c)] = Fraction(1, 1)
    
    rank = U.shape[1]
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                expected = T_frac.get((a, b, c), Fraction(0, 1))
                actual = Fraction(0, 1)
                for r in range(rank):
                    u_val = U[a, r]
                    v_val = V[b, r]
                    w_val = W[c, r]
                    u_f = Fraction(u_val).limit_denominator(1000)
                    v_f = Fraction(v_val).limit_denominator(1000)
                    w_f = Fraction(w_val).limit_denominator(1000)
                    actual += u_f * v_f * w_f
                if actual != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _gradient_step(U, V, W, T, lr=0.01):
    """Perform one gradient descent step on all three matrices simultaneously.
    
    The objective is sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
    Gradients:
      d/dU[a,r] = 2 * sum_{b,c} (est[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
      d/dV[b,r] = 2 * sum_{a,c} (est[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
      d/dW[c,r] = 2 * sum_{a,b} (est[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
    """
    # Compute estimates
    est = np.einsum('ar,br,cr->abc', U, V, W)
    diff = est - T  # shape (nm, mp, pn)
    
    # Gradients
    # grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    U -= lr * grad_U
    V -= lr * grad_V
    W -= lr * grad_W
    return U, V, W

def _try_rank_r_gradient(r, n, m, p, T, max_iterations=20000, seed=0, deadline=None):
    """Try to find a rank-r decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf
    
    # Try multiple random initializations
    for init in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        # Adaptive learning rate
        lr = 0.01
        for it in range(max_iterations):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Gradient step
            U, V, W = _gradient_step(U, V, W, T, lr)
            
            # Adaptive learning rate: reduce if error increased
            T_recon_new = np.einsum('ar,br,cr->abc', U, V, W)
            err_new = np.sum((T_recon_new - T) ** 2)
            if err_new > err * 1.5:
                lr *= 0.5
                # Revert
                U, V, W = U + lr * 2.0 * np.einsum('abc,br,cr->ar', T_recon_new - T, V, W), \
                          V + lr * 2.0 * np.einsum('abc,ar,cr->br', T_recon_new - T, U, W), \
                          W + lr * 2.0 * np.einsum('abc,ar,br->cr', T_recon_new - T, U, V)
            elif err < 1e-6:
                lr = min(lr * 1.2, 0.05)
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        # Try to snap to half-integers
        U_snap = _snap_to_half_integer(best_U)
        V_snap = _snap_to_half_integer(best_V)
        W_snap = _snap_to_half_integer(best_W)
        if U_snap is not None and V_snap is not None and W_snap is not None:
            return {"U": U_snap, "V": V_snap, "W": W_snap}, best_err
        else:
            return {"U": best_U, "V": best_V, "W": best_W}, best_err
    
    return None, best_err

def _try_rank_r_local_search(r, n, m, p, T, max_iterations=5000, seed=0, deadline=None):
    """Try to find a rank-r decomposition using local search on half-integers."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Use a wider set of values to allow for larger coefficients
    # The values can be integers or half-integers
    vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf
    
    # Try multiple random initializations
    # Increase restarts for better coverage
    for init in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.choice(vals, size=(nm, r))
        V = rng.choice(vals, size=(mp, r))
        W = rng.choice(vals, size=(pn, r))
        
        # Local search with simulated annealing-like acceptance
        temp = 1.0
        for it in range(max_iterations):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
            
            # Pick a random element to perturb
            if r > 0:
                mat_choice = rng.randint(3)
                if mat_choice == 0:
                    row = rng.randint(nm)
                    col = rng.randint(r)
                    old = U[row, col]
                    U[row, col] = rng.choice(vals)
                    mat_name = 'U'
                elif mat_choice == 1:
                    row = rng.randint(mp)
                    col = rng.randint(r)
                    old = V[row, col]
                    V[row, col] = rng.choice(vals)
                    mat_name = 'V'
                else:
                    row = rng.randint(pn)
                    col = rng.randint(r)
                    old = W[row, col]
                    W[row, col] = rng.choice(vals)
                    mat_name = 'W'
                
                # Compute new error
                T_recon_new = np.einsum('ar,br,cr->abc', U, V, W)
                err_new = np.sum((T_recon_new - T) ** 2)
                
                delta = err_new - err
                # Accept if better, or with probability exp(-delta/temp)
                if delta < 0 or rng.rand() < np.exp(-delta / temp):
                    # Accept
                    temp *= 0.99
                else:
                    # Revert
                    if mat_name == 'U':
                        U[row, col] = old
                    elif mat_name == 'V':
                        V[row, col] = old
                    else:
                        W[row, col] = old
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-10:
        # Try to snap to half-integers
        U_snap = _snap_to_half_integer(best_U)
        V_snap = _snap_to_half_integer(best_V)
        W_snap = _snap_to_half_integer(best_W)
        if U_snap is not None and V_snap is not None and W_snap is not None:
            return {"U": U_snap, "V": V_snap, "W": W_snap}, best_err
        else:
            return {"U": best_U, "V": best_V, "W": best_W}, best_err
    
    return None, best_err

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try ranks from 39 down to some lower bound
    # We prioritize local search as it is more effective for discrete/half-integer solutions
    for r in range(39, 25, -1):
        if time.time() > deadline:
            break
        
        # Try local search first with more iterations
        result, err = _try_rank_r_local_search(r, n, m, p, T, max_iterations=5000, seed=42, deadline=deadline)
        if result is not None:
            # Verify
            U_np = np.array(result["U"], dtype=np.float64)
            V_np = np.array(result["V"], dtype=np.float64)
            W_np = np.array(result["W"], dtype=np.float64)
            ok, max_err = _check_decomposition(U_np, V_np, W_np, T)
            if ok:
                return result
        
        # Then try gradient descent if time remains
        if time.time() > deadline:
            break
        result, err = _try_rank_r_gradient(r, n, m, p, T, max_iterations=5000, seed=42, deadline=deadline)
        if result is not None:
            # Verify
            U_np = np.array(result["U"], dtype=np.float64)
            V_np = np.array(result["V"], dtype=np.float64)
            W_np = np.array(result["W"], dtype=np.float64)
            ok, max_err = _check_decomposition(U_np, V_np, W_np, T)
            if ok:
                return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
