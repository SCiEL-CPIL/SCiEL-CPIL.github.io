"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

# Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 138600x^4 - 332640x^2 + 10395

H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def _get_Q_coeffs(c0, c1, c2):
    """Return coefficients of Q(y) = g(x)/x^2 where y=x^2, as [a5, a4, a3, a2, a1, a0]"""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g is even, g(0) = 0
    # Q(y) = g(sqrt(y)) / y = sum of x^{2k} terms / x^2
    # g2 = -6*c1 - 420*c2 - 332640*c3
    # g4 = c1 + 210*c2 + 138600*c3
    # g6 = -28*c2 - 25200*c3
    # g8 = c2 + 2100*c3
    # g10 = -84*c3
    # g12 = c3
    # Q(y) = g2 + g4*y + g6*y^2 + g8*y^3 + g10*y^4 + g12*y^5
    
    a5 = c3
    a4 = -84.0 * c3
    a3 = 2100.0 * c3 + c2
    a2 = -25200.0 * c3 - 28.0 * c2
    a1 = 138600.0 * c3 + 210.0 * c2 + c1
    a0 = -332640.0 * c3 - 420.0 * c2 - 6.0 * c1
    
    return np.array([a5, a4, a3, a2, a1, a0])

def _find_largest_sign_change_root(coeffs):
    """Find the largest positive root of Q(y) where the sign changes."""
    # Use np.roots for polynomial roots
    try:
        roots = np.roots(coeffs)
    except:
        return None
    
    candidates = []
    for root in roots:
        if np.isreal(root):
            r = np.real(root)
            if r > 1e-10:
                # Check if it's a sign-changing root by evaluating nearby
                eps = max(1e-6, abs(r) * 1e-5)
                q_left = np.polyval(coeffs, r - eps)
                q_right = np.polyval(coeffs, r + eps)
                if q_left * q_right < 0:
                    candidates.append(r)
    
    if not candidates:
        return None
    return max(candidates)

def compute_value(c0, c1, c2):
    coeffs = _get_Q_coeffs(c0, c1, c2)
    R = _find_largest_sign_change_root(coeffs)
    if R is None or R <= 0:
        return 1e10
    # value = r^2 / (2*pi) where r^2 = R
    return R / (2.0 * np.pi)

def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = compute_value(seed[0], seed[1], seed[2])
    
    best = seed.copy()
    best_val = seed_val
    
    # Multi-start local optimization from different points
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = [seed.copy()]
    
    # Add points from a grid around the seed
    for dc0 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
        for dc1 in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            for dc2 in [-0.001, -0.0005, 0.0, 0.0005, 0.001]:
                pt = seed + np.array([dc0, dc1, dc2])
                starts.append(pt)
    
    # Add some random perturbations
    for _ in range(20):
        pt = seed + rng.normal(0, 0.1, 3)
        starts.append(pt)
    
    # Also try points with different scales
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        pt = seed * scale + rng.normal(0, 0.05 * scale, 3)
        starts.append(pt)
    
    # Try some structured points
    starts.append(np.array([-1.0, 0.0, 0.0]))
    starts.append(np.array([0.0, 1.0, 0.0]))
    starts.append(np.array([0.0, 0.0, 1.0]))
    starts.append(np.array([-1.0, 0.5, 0.1]))
    starts.append(np.array([-0.5, -0.5, 0.01]))
    
    for start in starts:
        if not np.all(np.isfinite(start)):
            continue
        
        def objective(coeffs):
            return compute_value(coeffs[0], coeffs[1], coeffs[2])
        
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
            
            if np.all(np.isfinite(res.x)):
                val = res.fun
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best = res.x.copy()
        except:
            continue
    
    # Try differential evolution for global search
    try:
        def objective_de(coeffs):
            return compute_value(coeffs[0], coeffs[1], coeffs[2])
        
        # Use reasonable bounds
        bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.1, 0.1)]
        
        res_de = differential_evolution(objective_de, bounds, seed=42, maxiter=100, 
                                        popsize=20, tol=1e-6, polish=True)
        
        if np.all(np.isfinite(res_de.x)):
            val = res_de.fun
            if np.isfinite(val) and val < best_val:
                best_val = val
                best = res_de.x.copy()
    except:
        pass
    
    if best_val < seed_val:
        return best
    return seed
# EVOLVE-BLOCK-END
