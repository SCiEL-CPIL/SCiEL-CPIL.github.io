import numpy as np
import time
from scipy.optimize import minimize

# Precompute grid
x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]
gauss = np.exp(-x**2 / 2)

# Probabilists' Hermite polynomials He_0, He_4, He_8, He_12 evaluated on x
def he0(x): return np.ones_like(x)
def he4(x): return x**4 - 6*x**2 + 3
def he8(x): return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105
def he12(x): return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

H0 = he0(x)
H4 = he4(x)
H8 = he8(x)
H12 = he12(x)

# Values at 0
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def c3_from(c0, c1, c2):
    return -(c0 + H4_0*c1 + H8_0*c2) / H12_0

def _compute_g_and_hat_g(c0, c1, c2):
    """Compute g(x) and |hat_g(xi)| on the grid."""
    c3 = c3_from(c0, c1, c2)
    g = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    # For k=0,4,8,12: (i)^k = 1, so P(i*xi) = P(xi)
    hat_g = np.sqrt(2*np.pi) * (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    return g, hat_g

def objective(coeffs):
    c0, c1, c2 = coeffs
    g, hat_g = _compute_g_and_hat_g(c0, c1, c2)
    l1 = np.sum(np.abs(g)) * dx
    min_abs_hat = np.min(np.abs(hat_g))
    if min_abs_hat < 1e-10:
        return 1e10
    return l1 / min_abs_hat

def objective_smooth(coeffs, beta=50.0):
    """Smooth surrogate using soft-min for |hat_g| via log-sum-exp."""
    c0, c1, c2 = coeffs
    g, hat_g = _compute_g_and_hat_g(c0, c1, c2)
    l1 = np.sum(np.abs(g)) * dx
    
    abs_hat = np.abs(hat_g)
    # Soft-min: -beta * log(sum(exp(-beta * abs_hat))) approximates min(abs_hat)
    # Use shifted version for numerical stability
    m = np.min(abs_hat)
    if m < 1e-15:
        return 1e10
    # soft_min = m - (1/beta) * log(sum(exp(-beta*(abs_hat - m))))
    shifted = -beta * (abs_hat - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = m - log_sum / beta
    
    if soft_min < 1e-10:
        return 1e10
    return l1 / soft_min

def construct():
    t0 = time.time()
    deadline = t0 + 195  # safety margin before 210s limit
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective(best)
    
    # Phase 1: Local refinement from seed using smooth objective
    try:
        res = minimize(objective_smooth, seed, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() < deadline - 60 and res.fun < best_val:
            candidate = res.x.copy()
            val = objective(candidate)
            if val < best_val:
                best = candidate
                best_val = val
    except Exception:
        pass
    
    # Phase 2: Multi-start with diverse initial points
    if time.time() < deadline - 80:
        rng = np.random.default_rng(42)
        # Try points in a wider range
        n_starts = 40
        for i in range(n_starts):
            if time.time() > deadline - 40:
                break
            # Vary the scale of random perturbations
            scale = 0.5 + 1.5 * (i / n_starts)
            start = seed + rng.uniform(-scale, scale, size=3)
            try:
                # Use smooth objective for speed
                res = minimize(objective_smooth, start, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
                if time.time() > deadline - 40:
                    break
                candidate = res.x.copy()
                val = objective(candidate)
                if val < best_val:
                    best = candidate
                    best_val = val
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead on the exact (nonsmooth) objective from best point
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol':1e-9, 'fatol':1e-11})
            if time.time() < deadline - 15 and res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol':1e-12, 'fatol':1e-14})
            if res.fun < best_val:
                best = res.x.copy()
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 5: One more smooth refinement + exact evaluation
    if time.time() < deadline - 5:
        try:
            res = minimize(objective_smooth, best, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                candidate = res.x.copy()
                val = objective(candidate)
                if val < best_val:
                    best = candidate
                    best_val = val
        except Exception:
            pass
    
    return np.array(best)
