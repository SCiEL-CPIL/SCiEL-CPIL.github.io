"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import time

N = 50

# EVOLVE-BLOCK-START
def _objective(x):
    """
    Objective to minimize: ratio = (L2_norm^2) / (L1_norm * sup_norm(autoconv))
    We minimize the negative of this ratio to maximize it.
    x: vector of nonnegative heights.
    """
    x = np.maximum(x, 0)  # Enforce nonnegativity in the objective
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 1e10
    l2_sq = np.sum(x**2)
    # Autoconvolution: linear convolution of x with itself
    ac = convolve(x, x, mode='full')
    sup_ac = np.max(ac)
    if sup_ac < 1e-12:
        return 1e10
    ratio = l2_sq / (l1 * sup_ac)
    return -ratio

def _gradient(x):
    """
    Approximate gradient for the objective.
    Since the objective involves max and abs-like behavior (nonnegativity),
    we use a numerical gradient or a simplified analytic one.
    For SLSQP, we can use numerical gradient if analytic is complex.
    """
    # Use numerical gradient for robustness
    eps = 1e-8
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        # Ensure nonnegativity in the perturbed versions for consistent evaluation
        # Actually, the objective function clamps to 0, so we just evaluate as is
        grad[i] = (_objective(x_plus) - _objective(x_minus)) / (2 * eps)
    return grad

def _constraints(x):
    # No explicit constraints needed if we handle nonnegativity in objective/gradient
    # But SLSQP can handle bounds
    pass

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_x = np.ones(N)
    best_score = _objective(best_x)
    
    # Parametric initialization: Gaussian-like peaks
    # Try a few different initial structures
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Single Gaussian in the center
    x_gauss = np.zeros(N)
    centers = np.linspace(0, N-1, N)
    for mu in [N//2, N//3, 2*N//3]:
        for sigma in [2, 5, 10, N//4]:
            x_gauss = np.exp(-0.5 * ((centers - mu) / sigma)**2)
            inits.append(x_gauss.copy())
    
    # 3. Double Gaussian
    for mu1 in [N//4, N//3]:
        for mu2 in [3*N//4, 2*N//3]:
            for sigma in [3, 5, 8]:
                x_double = np.exp(-0.5 * ((centers - mu1) / sigma)**2) + \
                           np.exp(-0.5 * ((centers - mu2) / sigma)**2)
                inits.append(x_double.copy())
    
    # 4. Cosine bump
    x_cos = np.cos(np.pi * (centers - N/2 + 0.5) / N)
    x_cos = np.maximum(x_cos, 0)
    inits.append(x_cos.copy())
    
    # 5. Triangle
    x_tri = np.maximum(1 - np.abs(centers - N/2) / (N/2), 0)
    inits.append(x_tri.copy())
    
    # Optimize each initialization
    for init in inits:
        if time.time() - start_time > deadline:
            break
        
        # Normalize to avoid scaling issues
        init = init / (np.sum(init) + 1e-12) * N  # Keep sum around N
        
        try:
            res = minimize(
                _objective,
                init,
                method='SLSQP',
                jac=_gradient,
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if res.success and res.fun < best_score:
                best_x = np.maximum(res.x, 0)
                best_score = res.fun
        except Exception:
            pass
    
    # Coordinate descent refinement
    if time.time() - start_time < deadline:
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            improved = False
            for i in range(N):
                # Try adjusting x[i] slightly
                for delta in [-0.1, 0.1, -0.01, 0.01, -0.001, 0.001]:
                    x_test = best_x.copy()
                    x_test[i] = max(0, best_x[i] + delta)
                    score_test = _objective(x_test)
                    if score_test < best_score:
                        best_x = x_test
                        best_score = score_test
                        improved = True
                        break
            if not improved:
                break
    
    # Final high-precision polish
    if time.time() - start_time < deadline:
        try:
            res = minimize(
                _objective,
                best_x,
                method='SLSQP',
                jac=_gradient,
                bounds=[(0, None)] * N,
                options={'maxiter': 2000, 'ftol': 1e-14}
            )
            if res.success and res.fun < best_score:
                best_x = np.maximum(res.x, 0)
                best_score = res.fun
        except Exception:
            pass
    
    return list(best_x)
# EVOLVE-BLOCK-END
