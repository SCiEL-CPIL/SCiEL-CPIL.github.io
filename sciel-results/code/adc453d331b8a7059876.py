import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad
import time

# Pre-compute Hermite values and weights for fast objective evaluation
# Use a dense grid with trapezoidal rule for speed and robustness
_X_MIN = -8.0
_X_MAX = 8.0
_N_PTS = 10000
_x_grid = np.linspace(_X_MIN, _X_MAX, _N_PTS)
_dx = _x_grid[1] - _x_grid[0]

# Pre-compute Hermite polynomials on the grid
_H0 = np.ones_like(_x_grid)
_H4 = eval_hermite(4, _x_grid)
_H8 = eval_hermite(8, _x_grid)
_H12 = eval_hermite(12, _x_grid)

# Pre-compute exponentials
_exp_2pi = np.exp(-2 * np.pi * _x_grid**2)
_exp_4pi = np.exp(-4 * np.pi * _x_grid**2)

# Weights for integration
_w_2 = _exp_2pi * _dx
_w_4 = _exp_4pi * _dx

def _compute_objective_fast(coeffs):
    a0, a4, a8 = coeffs
    
    # Calculate a12 such that P(0) = 0
    # P(0) = a0*1 + a4*(-12) + a8*(1680) + a12*(-332640) = 0
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 332640.0
    
    # Compute P(x) and P_minus(x)
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
    
    P2 = P * P
    P4 = P2 * P2
    
    Pm2 = P_minus * P_minus
    Pm4 = Pm2 * Pm2
    
    norm2_sq = np.dot(P2, _w_2)
    norm4_sq = np.dot(P4, _w_4)
    
    norm2_sq_hat = np.dot(Pm2, _w_2)
    norm4_sq_hat = np.dot(Pm4, _w_4)
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
        
    obj = (norm4_sq / (norm2_sq * norm2_sq)) * (norm4_sq_hat / (norm2_sq_hat * norm2_sq_hat))
    
    return obj

def _compute_objective_exact(coeffs):
    """Compute objective using adaptive quadrature for exact integrals."""
    a0, a4, a8 = coeffs
    
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 332640.0
    
    def P(x):
        return a0*1.0 + a4*eval_hermite(4, x) + a8*eval_hermite(8, x) + a12*eval_hermite(12, x)
    
    def P_minus(x):
        return a0*1.0 + a4*eval_hermite(4, x) + a8*eval_hermite(8, x) - a12*eval_hermite(12, x)
    
    def integrand2(x):
        return P(x)**2 * np.exp(-2*np.pi*x**2)
    
    def integrand4(x):
        return P(x)**4 * np.exp(-4*np.pi*x**2)
    
    def integrand2_hat(x):
        return P_minus(x)**2 * np.exp(-2*np.pi*x**2)
    
    def integrand4_hat(x):
        return P_minus(x)**4 * np.exp(-4*np.pi*x**2)
    
    norm2_sq, _ = quad(integrand2, -np.inf, np.inf, limit=200)
    norm4_sq, _ = quad(integrand4, -np.inf, np.inf, limit=200)
    norm2_sq_hat, _ = quad(integrand2_hat, -np.inf, np.inf, limit=200)
    norm4_sq_hat, _ = quad(integrand4_hat, -np.inf, np.inf, limit=200)
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
        
    obj = (norm4_sq / (norm2_sq * norm2_sq)) * (norm4_sq_hat / (norm2_sq_hat * norm2_sq_hat))
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    seed_score = _compute_objective_fast(x0)
    
    best_result = x0.copy()
    best_score = seed_score
    
    # Generate diverse starting points
    np.random.seed(42)
    
    starts = [x0.copy()]
    
    # Small perturbations
    for i in range(15):
        starts.append(x0 + np.random.randn(3) * np.array([0.1, 0.01, 0.0001]))
        
    # Larger perturbations
    for i in range(15):
        starts.append(x0 + np.random.randn(3) * np.array([1.0, 0.1, 0.001]))
        
    # Even larger perturbations
    for i in range(10):
        starts.append(x0 + np.random.randn(3) * np.array([5.0, 0.5, 0.01]))
        
    # Specific heuristics
    starts.append(np.array([-1.0, 0.1, 0.0]))
    starts.append(np.array([-0.5, 0.05, 0.0]))
    starts.append(np.array([-2.0, 0.2, 0.0]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-1.13, 0.04, 0.0003]))
    starts.append(np.array([-1.5, 0.06, 0.0005]))
    starts.append(np.array([-0.8, 0.03, 0.0002]))
    
    # Pass 1: Nelder-Mead global search (derivative-free, robust to local minima)
    for s in starts:
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(_compute_objective_fast, s, method='Nelder-Mead',
                           options={'maxiter': 5000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    # Pass 2: L-BFGS-B local refinement from best Nelder-Mead result
    if time.time() < deadline - 40:
        try:
            res = minimize(_compute_objective_fast, best_result, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    # Pass 3: More Nelder-Mead starts around the current best
    if time.time() < deadline - 30:
        for i in range(20):
            if time.time() > deadline - 25:
                break
            s = best_result + np.random.randn(3) * np.array([0.05, 0.005, 0.00005])
            try:
                res = minimize(_compute_objective_fast, s, method='Nelder-Mead',
                               options={'maxiter': 3000, 'fatol': 1e-13, 'xatol': 1e-13, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
                
    # Pass 4: Another L-BFGS-B polish
    if time.time() < deadline - 20:
        try:
            res = minimize(_compute_objective_fast, best_result, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    # Pass 5: High-precision polish using exact quadrature
    if time.time() < deadline - 15:
        try:
            res_exact = minimize(_compute_objective_exact, best_result, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
            if res_exact.fun < best_score:
                best_score = res_exact.fun
                best_result = res_exact.x.copy()
        except Exception:
            pass
        
        # Final tight polish with the fast objective
        if time.time() < deadline - 5:
            try:
                res = minimize(_compute_objective_fast, best_result, method='L-BFGS-B',
                               options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
                
    # Ensure we return a valid result
    if best_result is None or not np.all(np.isfinite(best_result)):
        return x0
        
    return best_result
