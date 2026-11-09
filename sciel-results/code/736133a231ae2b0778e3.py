"""Second autocorrelation inequality: maximize ratio via Fourier phase optimization."""
import numpy as np
import time
import sys

N = 50
DT = 1.0 / N  # interval spacing on [-1/4, 1/4]

# Precompute the FFT/IFFT setup
# We work in Fourier domain. The signal x has N nonnegative values.
# We parameterize via: x[n] = |IFFT(F)[n]| where F has fixed L2 norm.
# Actually, let's think more carefully.

# The ratio to maximize is:
#   R = ||x||_2^2 / (||x||_1 * ||x*x||_∞)
# where x*x is the autoconvolution.

# With equal spacing DT on [-1/4, 1/4], the discrete autoconvolution is:
#   (x*x)[k] = sum_j x[j] * x[k-j]  (circular or linear?)
# For a signal on [-1/4, 1/4] with N=50 points, DT = 1/50.
# The autoconvolution of a signal supported on [-1/4,1/4] has support on [-1/2, 1/2].
# The supremum norm is the max absolute value of the autoconvolution.

# Let's use linear convolution (non-circular) since the support is finite.
# Actually, for the L∞ norm of the autoconvolution, we need to be careful.

# Let me reconsider. The task says "autoconvolution" which typically means:
#   (x * x)(t) = ∫ x(s) x(t-s) ds
# For discrete signals on equal intervals, this is a scaled convolution.

# Let's use numpy's convolve for linear convolution.
# The L2 norm: ||x||_2 = sqrt(sum(x^2)) (or with DT factor?)
# The L1 norm: ||x||_1 = sum(|x|) (or with DT factor?)

# Since all x >= 0, L1 = sum(x).
# The ratio R = ||x||_2^2 / (||x||_1 * ||x*x||_∞)

# For the discrete case, let's define:
# ||x||_2^2 = sum(x[n]^2)
# ||x||_1 = sum(x[n])
# (x*x)[k] = sum_j x[j] x[k-j] for linear convolution
# ||x*x||_∞ = max_k |(x*x)[k]|

# To maximize R, we want to minimize the denominator relative to numerator.
# Since x >= 0, ||x||_1 is fixed by scaling. The ratio is scale-invariant!
# R = ||x||_2^2 / (||x||_1 * ||x*x||_∞)
# If we scale x by c, numerator scales by c^2, denominator by c^2. So R is scale-invariant.
# We can fix ||x||_2 = 1 and minimize ||x||_1 * ||x*x||_∞.

# Strategy: Use Fourier phase optimization.
# Let X = FFT(x). Then x = IFFT(X).
# We parameterize X = A * exp(i*phi) where A is fixed (magnitude) and phi is the phase.
# But we need x >= 0, which is a constraint.

# Alternative: Direct gradient-based optimization on x with nonnegativity constraint.
# Use projected gradient descent or similar.

# Let's try a direct approach: optimize x >= 0 to maximize R.
# R(x) = sum(x^2) / (sum(x) * max(conv(x,x)))
# Since R is scale-invariant, fix sum(x^2) = 1 (unit L2 norm).
# Then maximize 1 / (sum(x) * max(conv(x,x)))
# i.e., minimize f(x) = sum(x) * max(conv(x,x)) subject to sum(x^2)=1, x>=0.

# This is non-smooth due to max. We can use subgradient or smooth approximation.

# Let's use a simpler approach: try various structured signals and local optimization.

def _autoconv(x):
    """Linear autoconvolution of x with itself."""
    return np.convolve(x, x, mode='full')

def _ratio(x):
    """Compute the ratio R = ||x||_2^2 / (||x||_1 * ||x*x||_inf)."""
    x = np.asarray(x, dtype=np.float64)
    l2_sq = np.sum(x ** 2)
    if l2_sq < 1e-30:
        return 0.0
    l1 = np.sum(np.abs(x))
    if l1 < 1e-30:
        return 0.0
    ac = _autoconv(x)
    ac_max = np.max(np.abs(ac))
    if ac_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _neg_ratio(x):
    return -_ratio(x)

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _ratio(best_x)
    
    # Try several initial configurations
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Triangular
    t = np.linspace(-1, 1, N)
    inits.append(np.maximum(0, 1 - np.abs(t)))
    # Gaussian-like
    t = np.linspace(-1, 1, N)
    inits.append(np.exp(-t**2))
    # Sinc-like (positive part)
    t = np.linspace(-1, 1, N)
    sinc = np.sinc(t)
    inits.append(np.maximum(0, sinc))
    # Two peaks
    x2 = np.zeros(N)
    x2[N//4] = 1.0
    x2[3*N//4] = 1.0
    inits.append(x2)
    # Ramp up then down
    x3 = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    inits.append(x3)
    # Cosine bump
    t = np.linspace(0, np.pi, N)
    inits.append(0.5 * (1 + np.cos(t)))
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        x_rand = rng.random(N)
        inits.append(x_rand)
    
    # For each init, do local optimization using projected gradient descent
    # Objective: minimize f(x) = log(sum(x) * max(conv(x,x))) subject to sum(x^2)=1, x>=0
    # Or directly maximize R(x).
    
    # Let's use scipy.optimize.minimize with bounds and constraint.
    from scipy.optimize import minimize
    
    for x0 in inits:
        if time.time() > deadline - 5:
            break
        
        x0 = np.asarray(x0, dtype=np.float64)
        if np.sum(x0**2) < 1e-30:
            continue
        # Normalize to unit L2
        x0 = x0 / np.sqrt(np.sum(x0**2))
        
        # Ensure nonnegative
        x0 = np.maximum(x0, 0)
        if np.sum(x0**2) < 1e-30:
            continue
        x0 = x0 / np.sqrt(np.sum(x0**2))
        
        def objective(x):
            return _neg_ratio(x)
        
        # Use SLSQP with nonnegativity constraints
        bounds = [(0, None)] * N
        # Add L2 norm constraint
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                        'jac': lambda x: 2 * x}]
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            if result.success or result.fun < best_score:
                x_opt = np.maximum(result.x, 0)
                if np.sum(x_opt**2) > 1e-30:
                    x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                    score = _ratio(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
        except Exception:
            pass
    
    # Additional local refinement with Nelder-Mead on a subset of dimensions
    # Try a few more random restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            x_rand = rng.random(N)
            x_rand = np.maximum(x_rand, 0)
            if np.sum(x_rand**2) < 1e-30:
                continue
            x_rand = x_rand / np.sqrt(np.sum(x_rand**2))
            
            try:
                result = minimize(_neg_ratio, x_rand, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                x_opt = np.maximum(result.x, 0)
                if np.sum(x_opt**2) > 1e-30:
                    x_opt = x_opt / np.sqrt(np.sum(x_opt**2))
                    score = _ratio(x_opt)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
            except Exception:
                pass
    
    # Final polish: local search on the best solution
    if time.time() < deadline - 2:
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            x_pert = best_x + 0.01 * rng.standard_normal(N)
            x_pert = np.maximum(x_pert, 0)
            if np.sum(x_pert**2) < 1e-30:
                continue
            x_pert = x_pert / np.sqrt(np.sum(x_pert**2))
            score = _ratio(x_pert)
            if score > best_score:
                best_score = score
                best_x = x_pert
    
    return list(best_x)

# Keep the interface exactly as required
if __name__ == '__main__':
    result = construct()
    print(f"Score: {_ratio(result)}")
