"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute interval width
    dx = (0.25 - (-0.25)) / N  # 0.5 / 600 = 1/1200

    def autoconvolve_max(x):
        """Compute the normalized maximum of the autoconvolution of x.
        
        x is a nonnegative vector of step heights on N equal intervals.
        The autoconvolution (continuous) approximated by discrete convolution:
        (x * x) at lag k is sum_i x[i] * x[i+k] (circular or linear?)
        
        For a function on [-1/4, 1/4] with support in that interval, the 
        autoconvolution at shift t is integral f(s) f(s-t) ds.
        
        For step functions with N intervals of width dx, the discrete autoconvolution
        at lag k (shift = k*dx) is dx * sum_i x[i] * x[i+k] for valid i, i+k.
        
        The normalized maximum: max over all shifts of the autoconvolution value.
        We want to minimize this.
        
        Note: The problem says "normalized maximum of its autoconvolution".
        Likely means: max_t |autoconv(t)| / (some normalization).
        
        Given the context of C1 (first autocorrelation inequality), the relevant 
        quantity is probably max_t autoconv(t) where autoconv(t) = integral f(s)f(s-t)ds.
        
        For nonnegative f, the maximum is at t=0: autoconv(0) = integral f(s)^2 ds.
        But that seems too simple. Let me reconsider.
        
        Actually, "first autocorrelation inequality" likely refers to:
        There exists C1 such that for any nonnegative f with certain properties,
        max_t autoconv(t) <= C1 * (something).
        
        The task says: "minimize the normalized maximum of its autoconvolution; 
        this gives an upper bound on C1."
        
        So we're choosing f to minimize max_t autoconv(t) / norm(f)^2 or similar.
        
        Let's assume: score = max_k (sum_i x[i]*x[i+k]) / (sum_i x[i]^2) 
        or similar. But without exact spec, let's use a reasonable formulation.
        
        Actually, re-reading: "Choose a nonnegative step function ... to minimize 
        the normalized maximum of its autoconvolution; this gives an upper bound on C1."
        
        A common formulation: For f with ||f||_1 = 1 (or some normalization),
        minimize max_t (f*f)(t). The "normalized" might mean normalized by ||f||_2^2 or ||f||_1^2.
        
        Let's go with: score = max_k conv[k] / (sum(x)^2) 
        where conv[k] = sum_i x[i]*x[i+k] (discrete autoconvolution).
        
        Actually, for the continuous case: autoconv(t) = integral f(s) f(s-t) ds.
        If we normalize so that integral f = 1, then autoconv(0) = integral f^2.
        The max of autoconv is at t=0 for nonnegative f.
        
        So minimizing max autoconv = minimizing integral f^2 given integral f = 1.
        The minimum is achieved by f = constant = 1/support = 4 (on [-1/4,1/4]).
        Then integral f^2 = 4^2 * 0.5 = 8.
        
        But that's a trivial answer. The problem must be more nuanced.
        
        Let me re-read: "minimize the normalized maximum of its autoconvolution".
        Perhaps the normalization is by the L1 norm cubed or something else.
        
        Given the ambiguity, I'll implement a reasonable interpretation and use 
        coordinate descent / scipy optimization to find a non-trivial local minimum.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        # Compute discrete autoconvolution (linear, not circular)
        # conv[k] for k in range(-n+1, n) 
        # For efficiency, compute only non-negative lags and note symmetry
        # autoconv[k] = sum_{i=0}^{n-1-k} x[i] * x[i+k] for k >= 0
        
        conv = np.zeros(n)
        for k in range(n):
            conv[k] = np.dot(x[:n-k], x[k:])
        
        max_conv = np.max(conv)
        
        # Normalization: let's use (sum(x))^2 as a natural scale
        total = np.sum(x)
        if total <= 0:
            return 1e10  # avoid division by zero, penalize
        
        return max_conv / (total * total)

    def objective(x):
        return autoconvolve_max(x)

    # Initial guess: uniform
    best_x = np.ones(N)
    best_val = objective(best_x)
    
    # Try a few different initializations
    initializations = [
        np.ones(N),
        np.exp(-np.linspace(0, 3, N)),  # decaying
        np.exp(-np.linspace(3, 0, N)),  # growing
        0.5 + 0.5 * np.cos(np.pi * np.linspace(0, 1, N)),  # cosine bump
        np.ones(N) * 0.5 + 0.5 * np.sin(np.pi * np.linspace(0, 1, N))**2,  # sin^2
    ]
    
    for x0 in initializations:
        if time.time() > deadline:
            break
        x0 = np.abs(x0)  # ensure nonnegative
        if np.sum(x0) < 1e-10:
            continue
        
        # Use minimize_scalar on a 1D parameterization? 
        # Better: use coordinate descent with line search
        
        x_current = x0.copy()
        x_current = np.abs(x_current)
        
        # Coordinate descent with scipy line search per coordinate
        n_iter = 0
        max_iter = 5000
        
        while n_iter < max_iter and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                    
                old_val = x_current[i]
                
                # Define 1D function: minimize over x_i >= 0
                def f_1d(xi):
                    x_test = x_current.copy()
                    x_test[i] = max(0, xi)
                    return objective(x_test)
                
                # Use scipy.optimize.minimize_scalar on bounded interval
                # Scale: x values are O(1), search in [0, 3*max]
                x_max = np.max(x_current) if np.max(x_current) > 0 else 1.0
                bounds = (0, 3 * x_max)
                
                try:
                    res = minimize_scalar(f_1d, bounds=bounds, method='bounded',
                                         options={'xatol': 1e-6})
                    new_val = max(0, res.x)
                    new_obj = f_1d(new_val)
                    
                    if new_obj < best_val - 1e-12:
                        x_current[i] = new_val
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                break
            n_iter += 1
        
        val = objective(x_current)
        if val < best_val:
            best_val = val
            best_x = x_current.copy()
    
    # Ensure nonnegative
    best_x = np.abs(best_x)
    
    return list(best_x)

# EVOLVE-BLOCK-END
