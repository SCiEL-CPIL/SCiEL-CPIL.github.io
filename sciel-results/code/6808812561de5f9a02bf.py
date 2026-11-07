import numpy as np
import time
import random

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # i ranges 0 to m (inclusive), so m+1 terms
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Strategy: Low-dimensional parameterization via symmetric basis functions.
    # The objective is invariant to reversal of h. 
    # We parameterize h using a few symmetric modes (cosine-like) and amplitudes.
    # This reduces the search space from 50D to K D, enabling exhaustive or dense global search.
    
    # Precompute basis functions
    # We use a set of symmetric basis vectors that span the space of symmetric functions.
    # Basis 0: Uniform (DC)
    # Basis 1: Cos(pi * x)
    # Basis 2: Cos(2 * pi * x)
    # ... up to some order.
    # Also include non-symmetric modes? The optimal might not be perfectly symmetric, 
    # but autocorrelation depends on the power spectrum. 
    # Actually, let's just use a linear combination of the first K eigenvectors of the identity? No.
    # Let's use a set of "bump" functions and "wave" functions.
    
    # A better approach for "different strategy": 
    # Instead of optimizing h directly, optimize the *autocorrelation* g directly?
    # g must be positive definite (autocorrelation of a real signal).
    # This is a constrained optimization on the cone of positive definite sequences.
    # But constructing h from g requires spectral factorization, which is complex.
    
    # Let's stick to h, but use a **Genetic Algorithm** with a specific **Symmetric Crossover**.
    # The current code uses random coordinate descent. 
    # The inspiration 2 mentioned GA. 
    # But let's try something even more structural: **Simulated Annealing on the Log-Height Vector**.
    # Heights must be non-negative. Optimizing log(h) avoids boundary issues at 0.
    # However, h can be 0. So we mix: h_i = exp(x_i) if x_i > -10 else 0.
    
    # Actually, the most "materially different" approach that often works for these 
    # ratio-of-norms problems is **Direct Search on the Unit Sphere of the Height Vector**.
    # Since the objective is scale-invariant? 
    # Let's check scale invariance.
    # g = h * h. If h -> c*h, g -> c^2 * g.
    # l2 -> c^2 * l2.
    # norm_1 -> c^2 * norm_1.
    # norm_inf -> c^2 * norm_inf.
    # value = (c^2 * l2) / (c^2 * norm_1 * c^2 * norm_inf) ??
    # Wait.
    # norm_1 = sum(|g|)/(m+1). If g -> c^2 g, norm_1 -> c^2 norm_1.
    # norm_inf = max(|g|). If g -> c^2 g, norm_inf -> c^2 norm_inf.
    # l2 is quadratic in y, which is linear in g. So l2 -> c^2 l2.
    # value = (c^2 l2) / (c^2 norm_1 * c^2 norm_inf) = l2 / (c^2 norm_1 norm_inf).
    # It is NOT scale invariant. It scales as 1/c^2.
    # So we want small c? 
    # If c -> 0, h -> 0, g -> 0. 0/0?
    # If h is scaled down, value goes to infinity?
    # Let's check the seed. h = [1]*50.
    # If I use h = [0.1]*50, value should be 100x larger?
    # Let's verify with a quick mental check or small code.
    # l2 ~ h^2. norm_1 ~ h^2. norm_inf ~ h^2.
    # value ~ h^2 / (h^2 * h^2) = 1/h^2.
    # So making h smaller increases the value!
    # Is there a lower bound on h? "nonnegative step heights".
    # If h -> 0, value -> infinity?
    # Let's re-read the task carefully.
    # "Choose 50 nonnegative step heights h"
    # "value = l2 / (norm_1 * norm_inf)"
    # If I set h = [1e-10] * 50, the value will be huge.
    # Why is the seed score 1.0?
    # Maybe the evaluator normalizes h?
    # "The evaluator computes exactly this value... higher is better."
    # "With g = np.convolve(h, h)..."
    # "norm_1 = sum(abs(g)) / (m + 1)"
    # "norm_inf = max(abs(g))"
    # "value = l2 / (norm_1 * norm_inf)"
    
    # If the problem allows arbitrary scaling, the answer is trivial (make h tiny).
    # There must be a constraint I'm missing or the "seed" implies a normalization.
    # "The seed is 50 equal heights." -> h=[1]*50.
    # "search score is value/seed".
    # If I can make value arbitrarily large, the score would be arbitrarily large.
    # In competitive programming/evolution tasks, usually there is a normalization.
    # Let's look at the definition of norm_1 and norm_inf again.
    # Maybe h is constrained to have a specific norm?
    # "Choose 50 nonnegative step heights h on equal intervals of [-1/4, 1/4]."
    # It does not say "unit norm" or "sum to 1".
    # However, if the value scales as 1/scale^2, then any program that outputs h=0.1*seed would score 100x better.
    # Since the current best is 1.24, it implies that either:
    # 1. The evaluator normalizes h before computing (e.g., h = h / max(h) or h = h / sum(h)).
    # 2. My scaling analysis is wrong.
    
    # Let's re-evaluate scaling.
    # y is built from g. g is convolution of h.
    # If h -> c h, then g -> c^2 g.
    # y -> c^2 y.
    # l2 = sum (w/3) (y_i^2 + y_i y_{i+1} + y_{i+1}^2).
    # Each term is quadratic in y, so l2 -> c^4 l2.
    # norm_1 = sum |g| / (m+1). g is quadratic in h? No, g is h*h.
    # If h -> c h, g -> c^2 g.
    # So norm_1 -> c^2 norm_1.
    # norm_inf -> c^2 norm_inf.
    # Denominator: norm_1 * norm_inf -> c^2 * c^2 = c^4.
    # Numerator: l2 -> c^4.
    # Value = c^4 / c^4 = 1.
    # **IT IS SCALE INVARIANT.**
    
    # Okay, my previous mental math error: l2 is quadratic in y, y is linear in g, g is quadratic in h.
    # So l2 is quartic in h.
    # norm_1 is linear in g, so quadratic in h.
    # norm_inf is linear in g, so quadratic in h.
    # Ratio is invariant.
    
    # So we cannot just scale. We must change the *shape* of h.
    
    # Strategy: **Global Optimization via CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
    # CMA-ES is a state-of-the-art black-box optimizer for continuous, non-convex problems.
    # It is significantly more effective than random restarts + coordinate descent for smooth, high-dimensional landscapes.
    # We will implement a lightweight CMA-ES or use scipy's differential_evolution if available?
    # SciPy is available.
    # Let's use scipy.optimize.differential_evolution.
    # It is a global search method, materially different from local coordinate descent.
    
    # However, differential_evolution can be slow for 50 dimensions.
    # Let's try to reduce dimensionality using Symmetry.
    # The objective is invariant to reversal of h.
    # So h[i] == h[N-1-i] is a good constraint.
    # We can optimize the first 25 elements and mirror them.
    # This reduces dimension to 25.
    
    # Plan:
    # 1. Define a function that takes 25 parameters, constructs a symmetric h of length 50, and evaluates it.
    # 2. Use scipy.optimize.differential_evolution to find the global maximum.
    # 3. Bounds: [0, 1] for each parameter (scale doesn't matter, so we can fix max height to 1 or just let it search in [0, 1]).
    #    Actually, since it's scale invariant, we can constrain the max height to be 1 to help the optimizer.
    #    Or just search in [0, 1].
    
    def symmetric_h(params):
        # params is length 25
        # h is length 50
        # h[0:25] = params
        # h[25:50] = params[::-1]
        # Ensure non-negative
        h = np.concatenate([params, params[::-1]])
        return h
    
    def objective(params):
        h = symmetric_h(params)
        # To help the optimizer, we can fix the scale.
        # Since value is scale invariant, we can normalize h to have max=1.
        # This prevents the optimizer from drifting to 0 or infinity, which might cause numerical issues or slow convergence in DE.
        max_h = np.max(h)
        if max_h < 1e-10:
            return 0.0
        h_norm = h / max_h
        val = evaluate(h_norm)
        # We want to maximize, so return negative for minimization
        return -val

    # Initial best
    best_h = [1.0] * N
    best_val = evaluate(best_h)
    
    # Run Differential Evolution
    # 25 dimensions
    bounds = [(0.0, 1.0) for _ in range(N // 2)]
    
    # We have ~115 seconds.
    # DE can be expensive. Let's limit the number of iterations or use a smaller population.
    # popsize=15 (default is 15*dim, which is 375, too big).
    # Let's try popsize=10, maxiter=50.
    # This might be slow.
    # Alternative: Use a faster global method like Basin Hopping?
    # Or just a well-tuned Random Search with Symmetry?
    
    # Let's try a hybrid:
    # 1. Fast Random Search on 25D symmetric space.
    # 2. Local refinement using Nelder-Mead on the 25D symmetric space.
    
    def local_refine_symmetric(params, time_limit):
        """Nelder-Mead on symmetric params."""
        from scipy.optimize import minimize
        # Use Nelder-Mead, it's fast for low dimensions
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        # Check time
        if time.time() > time_limit:
            return params, -best_val # Return original if timeout? No, return best found.
        # The objective returns -val. So res.fun is -val.
        val = -res.fun
        h = symmetric_h(res.x)
        max_h = np.max(h)
        if max_h > 1e-10:
            h = h / max_h
        else:
            h = np.ones(N)
        return list(res.x), val

    # Time allocation
    t0 = time.time()
    
    # 1. Random Sampling of Symmetric Profiles
    num_samples = 1000
    # We have ~110 seconds. 1000 samples * 50us each = 50ms. Very fast.
    # We can do many more.
    
    best_params = np.ones(N // 2)
    best_sym_val = best_val
    
    # Generate random symmetric profiles
    # Mix of uniform, spiky, smooth
    for i in range(5000):
        if time.time() > deadline - 10.0:
            break
            
        if i % 100 == 0:
            # Smooth
            params = np.random.uniform(0.5, 1.0, N // 2)
            # Add some structure
            x = np.linspace(0, 1, N // 2)
            params = params * (0.5 + 0.5 * np.cos(np.pi * x * np.random.uniform(1, 5)))
        elif i % 100 == 50:
            # Spiky
            params = np.zeros(N // 2)
            idx = np.random.choice(N // 2, size=5, replace=False)
            params[idx] = np.random.uniform(0.5, 1.0)
        else:
            # Pure Random
            params = np.random.uniform(0, 1, N // 2)
            
        h = symmetric_h(params)
        max_h = np.max(h)
        if max_h < 1e-10: continue
        h = h / max_h
        val = evaluate(h)
        
        if val > best_sym_val:
            best_sym_val = val
            best_params = params.copy()
            best_h = list(h)
            
    # 2. Local Refinement with Nelder-Mead from the best point found
    if time.time() < deadline - 5.0:
        refined_params, refined_val = local_refine_symmetric(best_params, deadline - 2.0)
        if refined_val > best_sym_val:
            best_sym_val = refined_val
            h = symmetric_h(refined_params)
            max_h = np.max(h)
            if max_h > 1e-10:
                best_h = list(h / max_h)
            else:
                best_h = [1.0] * N

    # 3. Try a few more random restarts if time permits
    if time.time() < deadline - 10.0:
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            # Perturb best
            params = best_params + np.random.normal(0, 0.2, N // 2)
            params = np.clip(params, 0, 1)
            h = symmetric_h(params)
            max_h = np.max(h)
            if max_h < 1e-10: continue
            h = h / max_h
            val = evaluate(h)
            
            if val > best_sym_val:
                # Refine this new best
                refined_params, refined_val = local_refine_symmetric(params, deadline - 5.0)
                if refined_val > best_sym_val:
                    best_sym_val = refined_val
                    h = symmetric_h(refined_params)
                    max_h = np.max(h)
                    if max_h > 1e-10:
                        best_h = list(h / max_h)
                    else:
                        best_h = [1.0] * N

    return best_h

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
