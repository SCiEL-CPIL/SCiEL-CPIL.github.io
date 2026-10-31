"""Program for the uncertainty inequality task using Remez-like optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
import math


def _hermite_poly(n, x):
    """Compute the nth physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        return (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
                - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    else:
        return hermite(n)(x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 on A(f)*A(F) for the given Hermite coefficients.
    
    Key insight: p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12 is even.
    g(x) = p(x)*exp(-x^2/2)
    A_f = sup |x*g(x)|
    G(xi) = sqrt(2*pi)*p(xi)*exp(-xi^2/2)  (since (-i)^n = 1 for even n)
    A_F = sup |xi*G(xi)| = sqrt(2*pi) * sup |xi*p(xi)*exp(-xi^2/2)| = sqrt(2*pi)*A_f
    So C4 = A_f * A_F = sqrt(2*pi) * A_f^2
    """
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    x = np.linspace(0, 8, 20001)
    gauss = np.exp(-x**2 / 2)
    p = (c0 * _hermite_poly(0, x) + c4 * _hermite_poly(4, x) + 
         c8 * _hermite_poly(8, x) + c12 * _hermite_poly(12, x))
    xg = np.abs(x * p * gauss)
    A_f = np.max(xg)
    
    return math.sqrt(2 * math.pi) * A_f * A_f


def _compute_Af(c0, c4, c8, x_grid):
    """Compute A_f = sup|x*p(x)*e^{-x^2/2}| on the given grid."""
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    gauss = np.exp(-x_grid**2 / 2)
    p = (c0 * _hermite_poly(0, x_grid) + c4 * _hermite_poly(4, x_grid) + 
         c8 * _hermite_poly(8, x_grid) + c12 * _hermite_poly(12, x_grid))
    xg = np.abs(x_grid * p * gauss)
    return np.max(xg), xg


def _remez_optimize(c_init, max_iter=200, tol=1e-14):
    """
    Remez exchange algorithm for minimizing A_f = sup|x*p(x)*e^{-x^2/2}|
    where p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12, c12 determined by p(0)=0.
    """
    c = np.array(c_init, dtype=float)
    x_grid = np.linspace(0.01, 6.0, 5001)
    
    best_Af, _ = _compute_Af(*c, x_grid)
    best_c = c.copy()
    best_bound = math.sqrt(2 * math.pi) * best_Af**2
    
    Af, xg = _compute_Af(*c, x_grid)
    
    # Find initial 4 extremal points
    local_max_indices = []
    for i in range(1, len(xg) - 1):
        if xg[i] > xg[i-1] and xg[i] >= xg[i+1]:
            local_max_indices.append(i)
    
    if len(local_max_indices) >= 4:
        local_max_indices.sort(key=lambda i: xg[i], reverse=True)
        crit_indices = local_max_indices[:4]
    else:
        crit_indices = np.argsort(xg)[-4:]
    
    crit_indices = sorted(crit_indices)
    
    for iteration in range(max_iter):
        c0, c4, c8 = c
        c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
        
        xc = x_grid[crit_indices]
        gauss_c = np.exp(-xc**2 / 2)
        H0_c = np.ones_like(xc)
        H4_c = 16 * xc**4 - 48 * xc**2 + 12
        H8_c = 256 * xc**8 - 3584 * xc**6 + 13440 * xc**4 - 13440 * xc**2 + 1680
        H12_c = (4096 * xc**12 - 122880 * xc**10 + 1290240 * xc**8 
                 - 6451200 * xc**6 + 13107200 * xc**4 - 9031680 * xc**2 + 1032192)
        
        p_c = c0 * H0_c + c4 * H4_c + c8 * H8_c + c12 * H12_c
        xg_c = xc * p_c * gauss_c
        signed_vals = xg_c
        
        q = xc * gauss_c
        inv_1032192 = 1.0 / 1032192.0
        A0 = H0_c - H12_c * inv_1032192
        A4 = H4_c - 12 * H12_c * inv_1032192
        A8 = H8_c - 1680 * H12_c * inv_1032192
        
        s = np.ones(len(crit_indices))
        if len(crit_indices) >= 2:
            first_sign = np.sign(signed_vals[0])
            if first_sign == 0:
                first_sign = 1
            for i in range(len(s)):
                s[i] = first_sign * ((-1)**i)
        
        M = np.zeros((4, 4))
        M[:, 0] = q * A0
        M[:, 1] = q * A4
        M[:, 2] = q * A8
        M[:, 3] = -s
        
        try:
            U, S, Vt = np.linalg.svd(M)
            sol = Vt[-1]
            c_new = np.array([sol[0], sol[1], sol[2]])
            t_new = sol[3]
            if t_new < 0:
                c_new = -c_new
                t_new = -t_new
        except:
            c_new = c.copy()
        
        Af_new, xg_new = _compute_Af(*c_new, x_grid)
        
        if Af_new < best_Af * (1 - 1e-10):
            best_Af = Af_new
            best_c = c_new.copy()
            best_bound = math.sqrt(2 * math.pi) * best_Af**2
            c = c_new
        
        Af_curr, xg_curr = _compute_Af(*c, x_grid)
        
        local_max_indices = []
        for i in range(1, len(xg_curr) - 1):
            if xg_curr[i] > xg_curr[i-1] and xg_curr[i] >= xg_curr[i+1]:
                local_max_indices.append(i)
        
        max_idx = np.argmax(xg_curr)
        for offset in range(-5, 6):
            idx = max_idx + offset
            if 0 <= idx < len(xg_curr):
                if idx not in local_max_indices:
                    local_max_indices.append(idx)
        
        if len(local_max_indices) > 0:
            local_max_indices = list(set(local_max_indices))
            local_max_indices.sort(key=lambda i: xg_curr[i], reverse=True)
            crit_indices = local_max_indices[:4]
            crit_indices = sorted(crit_indices)
        
        if iteration > 10:
            prev_Af, _ = _compute_Af(*best_c, x_grid)
            if abs(prev_Af - best_Af) < tol * best_Af:
                break
    
    x_fine = np.linspace(0.001, 7.0, 50001)
    Af_final, _ = _compute_Af(*best_c, x_fine)
    final_bound = math.sqrt(2 * math.pi) * Af_final**2
    
    return best_c, final_bound


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_bound = _compute_bound(*best_c)
    
    start_points = [
        best_c,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.2, 0.05, 0.0005]),
        np.array([-0.9, 0.15, 0.002]),
        np.array([-1.1, 0.08, 0.001]),
        np.array([-1.05, 0.09, 0.0012]),
        np.array([-1.15, 0.07, 0.0008]),
    ]
    
    for sp in start_points:
        try:
            c_opt, bound_opt = _remez_optimize(sp, max_iter=150, tol=1e-14)
            if bound_opt < best_bound:
                best_bound = bound_opt
                best_c = c_opt
        except Exception:
            pass
    
    # Gradient descent refinement
    x_grid = np.linspace(0.01, 6.0, 10001)
    for _ in range(50):
        c0, c4, c8 = best_c
        Af_curr, _ = _compute_Af(*best_c, x_grid)
        bound_curr = math.sqrt(2 * math.pi) * Af_curr**2
        
        eps = 1e-7
        grad = np.zeros(3)
        for i in range(3):
            c_plus = best_c.copy()
            c_minus = best_c.copy()
            c_plus[i] += eps
            c_minus[i] -= eps
            Af_plus, _ = _compute_Af(*c_plus, x_grid)
            Af_minus, _ = _compute_Af(*c_minus, x_grid)
            bound_plus = math.sqrt(2 * math.pi) * Af_plus**2
            bound_minus = math.sqrt(2 * math.pi) * Af_minus**2
            grad[i] = (bound_plus - bound_minus) / (2 * eps)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-12:
            break
        
        step_size = 0.01 / norm_grad
        c_new = best_c - step_size * grad
        
        Af_new, _ = _compute_Af(*c_new, x_grid)
        bound_new = math.sqrt(2 * math.pi) * Af_new**2
        
        if bound_new < bound_curr:
            best_c = c_new
            best_bound = bound_new
        else:
            step_size = 0.001 / norm_grad
            c_new = best_c + step_size * grad
            Af_new, _ = _compute_Af(*c_new, x_grid)
            bound_new = math.sqrt(2 * math.pi) * Af_new**2
            if bound_new < bound_curr:
                best_c = c_new
                best_bound = bound_new
            else:
                break
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
