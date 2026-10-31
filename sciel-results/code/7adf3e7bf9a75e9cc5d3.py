"""Program for the uncertainty inequality task using Remez exchange on the dual min-max problem."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def _hermite_poly(n, x):
    """Compute the nth physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        return (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
                - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    else:
        raise ValueError("Only H0, H4, H8, H12 needed")


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 = A(f)*A(F) for the given Hermite coefficients.
    c12 is determined by the constraint g(0) = 0.
    """
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    # Use a fine grid
    x = np.linspace(-8, 8, 8001)
    gauss = np.exp(-x**2 / 2)
    
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H12 = (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
           - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * gauss
    
    A_f = np.max(np.abs(x * g))
    
    # By symmetry (even powers only), G(xi) has the same form with sqrt(2*pi) factor
    sqrt_2pi = math.sqrt(2 * math.pi)
    G = sqrt_2pi * P * gauss  # Same P since even powers, xi = x
    A_F = np.max(np.abs(x * G))
    
    return A_f * A_F


def _find_critical_points(x, g, n_max=4):
    """
    Find critical points where |x*g(x)| achieves local maxima near the global max.
    Returns sorted list of x values.
    """
    xg = np.abs(x * g)
    max_val = np.max(xg)
    
    # Find all local maxima
    local_max_idx = []
    for i in range(1, len(xg) - 1):
        if xg[i] > xg[i-1] and xg[i] >= xg[i+1]:
            local_max_idx.append(i)
    
    if not local_max_idx:
        # Fallback: use points near global max
        thresh = 0.95 * max_val
        near_max = np.where(xg >= thresh)[0]
        if len(near_max) > 0:
            local_max_idx = [near_max[0], near_max[-1]]
    
    # Keep only significant local maxima (within 5% of global max)
    significant = [i for i in local_max_idx if xg[i] >= 0.95 * max_val]
    
    if len(significant) < 2:
        # Use top n_max points
        top_idx = np.argsort(xg)[-n_max:]
        significant = sorted(top_idx.tolist())
    else:
        # Keep up to n_max most significant
        significant.sort(key=lambda i: -xg[i])
        significant = significant[:n_max]
        significant.sort()  # Sort by x value
    
    return [x[i] for i in significant]


def _remez_exchange(c_init, max_iter=200, tol=1e-10):
    """
    Remez exchange algorithm for the dual min-max problem.
    
    We minimize C4 = A_f * A_F subject to g(0) = 0.
    At the optimum, the "error" should equiosculate at critical points.
    
    Strategy:
    1. Find critical points of |x*g(x)| (position space)
    2. By symmetry, same points are critical for |xi*G(xi)| (frequency space)
    3. Set up linear system to make x*g(x) alternate signs at these points
    4. Solve for c0, c4, c8
    """
    c = np.array(c_init, dtype=float)
    best_bound = _compute_bound(*c)
    best_c = c.copy()
    
    # Fixed grid
    x = np.linspace(-6, 6, 6001)
    gauss = np.exp(-x**2 / 2)
    
    # Precompute Hermite polynomials
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H12 = (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
           - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    
    sqrt_2pi = math.sqrt(2 * math.pi)
    
    for iteration in range(max_iter):
        c0, c4, c8 = c
        c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
        
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = P * gauss
        
        xg_signed = x * g  # Signed version for equiosculation
        
        # Find critical points (local maxima of |xg|)
        xg_abs = np.abs(xg_signed)
        crit_points = _find_critical_points(x, g, n_max=4)
        
        if len(crit_points) < 3:
            # Not enough critical points, use fallback
            crit_points = [x[np.argmax(xg_abs)], x[np.argmin(xg_abs)]]
        
        # Build the equiosculation system
        # We want x_i * g(x_i) = (-1)^i * t for i = 0, 1, ..., m-1
        # where m = len(crit_points)
        # 
        # x_i * g(x_i) = x_i * (c0*H0(x_i) + c4*H4(x_i) + c8*H8(x_i) + c12*H12(x_i)) * gauss(x_i)
        # = x_i * gauss(x_i) * (c0*H0(x_i) + c4*H4(x_i) + c8*H8(x_i) + c12*H12(x_i))
        #
        # Substitute c12:
        # c12 = -(c0 + 12*c4 + 1680*c8) / 1032192
        #
        # So: x_i * g(x_i) = x_i * gauss(x_i) * [c0*(H0(x_i) - H12(x_i)/1032192) 
        #                         + c4*(H4(x_i) - 12*H12(x_i)/1032192)
        #                         + c8*(H8(x_i) - 1680*H12(x_i)/1032192)]
        
        m = len(crit_points)
        
        # Build coefficient matrix
        # For each critical point x_i:
        # a_i0 * c0 + a_i1 * c4 + a_i2 * c8 = (-1)^i * t
        # where a_ij = x_i * gauss(x_i) * (H_j(x_i) - (coeff_j/1032192)*H12(x_i))
        
        # Coefficients for c12 substitution
        c12_coeff_0 = 1.0 / 1032192.0
        c12_coeff_4 = 12.0 / 1032192.0
        c12_coeff_8 = 1680.0 / 1032192.0
        
        A = np.zeros((m, 3))
        b = np.zeros(m)
        
        for i, xi in enumerate(crit_points):
            # Find index in x grid
            idx = np.argmin(np.abs(x - xi))
            
            H0_i = H0[idx]
            H4_i = H4[idx]
            H8_i = H8[idx]
            H12_i = H12[idx]
            gauss_i = gauss[idx]
            
            # Effective coefficients after substituting c12
            eff_0 = H0_i - c12_coeff_0 * H12_i
            eff_4 = H4_i - c12_coeff_4 * H12_i
            eff_8 = H8_i - c12_coeff_8 * H12_i
            
            factor = x[idx] * gauss_i
            
            A[i, 0] = factor * eff_0
            A[i, 1] = factor * eff_4
            A[i, 2] = factor * eff_8
            
            # Alternating signs
            b[i] = (-1)**i
        
        # Solve A * [c0, c4, c8]^T = t * b
        # This is a homogeneous system in (c0, c4, c8, t)
        # Augment: [A | -b] * [c0, c4, c8, t]^T = 0
        
        A_aug = np.column_stack([A, -b])
        
        # Use SVD to find the null space (right singular vector for smallest singular value)
        U, s, Vt = np.linalg.svd(A_aug)
        
        # The solution is the last row of Vt (corresponding to smallest singular value)
        c_new_t = Vt[-1]
        c_new = c_new_t[:3]
        t_val = c_new_t[3]
        
        # Normalize: the sign of t doesn't matter, but we want consistent scaling
        # The bound is determined by the magnitude
        new_bound = _compute_bound(*c_new)
        
        if new_bound < best_bound * (1 - 1e-12):
            best_bound = new_bound
            best_c = c_new.copy()
            c = c_new
        else:
            # If the Remez step didn't improve, try a local search
            improved = False
            for step in [0.01, 0.001, 0.0001]:
                for d in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                          np.array([1,1,0]), np.array([1,0,1]), np.array([0,1,1]),
                          np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,-1]),
                          np.array([-1,-1,0]), np.array([-1,0,-1]), np.array([0,-1,-1])]:
                    trial = c + step * d
                    trial_bound = _compute_bound(*trial)
                    if trial_bound < best_bound * (1 - 1e-8):
                        c = trial
                        best_bound = trial_bound
                        best_c = c.copy()
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        # Check convergence
        if iteration > 5:
            prev_best = _compute_bound(*best_c)
            if abs(prev_best - best_bound) < tol * best_bound:
                break
    
    return best_c, best_bound


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    # Start from the known good solution
    best_c = np.array([-1.13, 0.04, 0.000308642])
    best_bound = _compute_bound(*best_c)
    
    # Run Remez optimization from multiple starting points
    start_points = [
        best_c,
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.03, 0.0003]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00028]),
    ]
    
    for sp in start_points:
        try:
            c_opt, bound_opt = _remez_exchange(sp, max_iter=100, tol=1e-10)
            if bound_opt < best_bound:
                best_bound = bound_opt
                best_c = c_opt
        except Exception:
            pass
    
    # Final refinement
    try:
        c_final, bound_final = _remez_exchange(best_c, max_iter=50, tol=1e-12)
        if bound_final < best_bound:
            best_bound = bound_final
            best_c = c_final
    except Exception:
        pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
