"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin below 120s limit
    
    def score(h):
        """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf).
        
        For a step function f(x) = h_i on interval i of width d:
          ||f||_1 = d * sum(h_i)
          ||f||_2^2 = d * sum(h_i^2)
          ||f*f||_inf = d * max_k sum_i h_i * h_{i+k}  (linear autocorrelation)
        
        So ratio = sum(h^2) / (d * sum(h) * max_autocorr)
        """
        h = np.asarray(h, dtype=float)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        # Linear autocorrelation
        ac = np.correlate(h, h, mode='full')
        max_ac = np.max(ac)
        if max_ac < 1e-30:
            return 0.0
        ratio = sum_sq / (d * sum_h * max_ac)
        return ratio
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Since the problem is scale-invariant, fix ||h||_2 = 1 and
    # maximize ||h||_1 / ||autoconv||_inf, equivalently minimize
    # log(||autoconv||_inf) - log(||h||_1) subject to ||h||_2 = 1, h >= 0.
    # Parameterize h = exp(y) / ||exp(y)||_2.
    
    def soft_max(x, beta=100.0):
        """Smooth approximation to max(x)"""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp((x - x_max) * beta))) / beta
    
    def objective_constrained(y, beta=50.0):
        """Minimize log(||autoconv||_inf / ||h||_1) with ||h||_2 = 1, h > 0."""
        h = np.exp(y)
        h_norm = np.sqrt(np.sum(h**2))
        h = h / h_norm
        
        sum_h = np.sum(h)
        if sum_h < 1e-15:
            return 1e10
        
        ac = np.correlate(h, h, mode='full')
        max_ac_soft = soft_max(ac, beta)
        
        val = np.log(max_ac_soft) - np.log(sum_h)
        return val
    
    def score_from_y(y):
        """Compute the actual score from the y-parameterization."""
        h = np.exp(y)
        h_norm = np.sqrt(np.sum(h**2))
        h = h / h_norm
        return score(h)
    
    # Build diverse initializations
    np.random.seed(42)
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros(N))
    
    # 2. Spike-like: one large value at various positions
    for peak_pos in [0, N//4, N//2, 3*N//4, N-1]:
        y = np.full(N, -5.0)
        y[peak_pos] = 0.0
        initializations.append(y.copy())
        
    # 3. Two spikes
    for p1, p2 in [(0, N-1), (N//4, 3*N//4), (0, N//2), (N//4, N//2), (N//3, 2*N//3)]:
        y = np.full(N, -5.0)
        y[p1] = 0.0
        y[p2] = 0.0
        initializations.append(y.copy())
    
    # 4. Random smooth
    for _ in range(15):
        y = np.random.randn(N) * 1.0
        initializations.append(y.copy())
    
    # 5. Cosine-based
    t = np.arange(N)
    for freq in [1, 2, 3, 4, 5]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            y = np.cos(2*np.pi*freq*t/N + phase) * 2.0
            initializations.append(y.copy())
    
    # 6. Gaussian-like bumps
    for center in [N//4, N//2, 3*N//4]:
        for width in [3, 5, 10, 20]:
            y = -((t - center)**2) / (2 * width**2)
            initializations.append(y.copy())
    
    # 7. Structured: alternating patterns
    for period in [2, 4, 5, 10]:
        y = np.zeros(N)
        for i in range(N):
            y[i] = 2.0 if (i % period) < period // 2 else -3.0
        initializations.append(y.copy())
    
    # 8. Ramp patterns
    for direction in [1, -1]:
        y = np.linspace(-3, 3, N) * direction
        initializations.append(y.copy())
    
    # Run L-BFGS-B from multiple initializations
    for y0 in initializations:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                objective_constrained,
                y0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            y_opt = res.x
            s = score_from_y(y_opt)
            if s > best_score:
                best_score = s
                best_h = np.exp(y_opt)
                best_h = best_h / np.sqrt(np.sum(best_h**2))
        except Exception:
            continue
    
    # Nelder-Mead polish from best and a few other points
    nm_initials = [best_h.copy() / np.sqrt(np.sum(best_h**2))]
    
    # Add diverse points from initializations
    for y0 in initializations[:10]:
        h = np.exp(y0)
        h = h / np.sqrt(np.sum(h**2))
        nm_initials.append(h)
    
    # Nelder-Mead on the ratio directly
    for h0 in nm_initials:
        if time.time() > deadline:
            break
        
        def nm_obj(h):
            return -score(h)
        
        try:
            res = minimize(
                nm_obj,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_h = res.x.copy()
        except Exception:
            continue
    
    # Final coordinate descent polish
    if time.time() < deadline - 15:
        h = best_h.copy()
        for _ in range(5000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try multiplicative perturbations
                for mult in [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 0.01, 0.0]:
                    h[i] = max(0.0, old * mult)
                    s = score(h)
                    if s > best_score:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break
    
    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)

# EVOLVE-BLOCK-END
