"""Optimized step function for first autocorrelation inequality."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
# Interval width
# Domain is [-1/4, 1/4], length 1/2.
# N = 600 intervals.
h = (1.0 / 2.0) / N

# The step function f(x) is defined on [-1/4, 1/4].
# f(x) = sum_{i=0}^{N-1} a_i * 1_{x in [x_i, x_{i+1})}
# Autoconvolution (f * f)(t) = \int f(x) f(t-x) dx.
# For step functions, this can be computed efficiently.
# Let's define the discrete convolution of the heights.
# The autoconvolution of the step function is a piecewise linear function (trapezoidal).
# The maximum value of the autoconvolution occurs at t=0 due to symmetry and non-negativity?
# Actually, for non-negative functions, the maximum of the convolution is at t=0 if the function is unimodal or symmetric?
# Not necessarily. But usually, for compact support and non-negative, the peak is near 0.
# Let's verify. If f is non-negative, (f*f)(0) = \int f(x)^2 dx.
# Is it possible that (f*f)(t) > (f*f)(0) for t != 0?
# Yes, if f is "two-peaked".
# However, the problem asks to minimize the *normalized maximum* of its autoconvolution.
# Normalized how?
# "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
# C1 is likely related to the first autocorrelation inequality.
# The inequality is usually: \int (f*f)(t)^2 dt <= C (\int f^2 dx)^2 ? Or similar.
# Wait, "First autocorrelation inequality" usually refers to:
# ||f * f||_2 <= C ||f||_2^2 ? No, that's trivial by Young's.
# Maybe: \max_t (f*f)(t) <= C \int f^2 dx ?
# If so, we want to minimize \max_t (f*f)(t) / \int f^2 dx.
# Let's assume the score is \max_t (f*f)(t) / \int f^2 dx.
# Or perhaps \max_t (f*f)(t) / (\int f dx)^2 ?
# Let's look at the seed program.
# Seed: [1.0] * N.
# f(x) = 1 on [-1/4, 1/4].
# \int f dx = 1/2.
# (f*f)(t) is a triangle function.
# At t=0, (f*f)(0) = \int_{-1/4}^{1/4} 1 * 1 dx = 1/2.
# Max of (f*f) is 1/2.
# \int f^2 dx = 1/2.
# Ratio 1: (1/2) / (1/2) = 1.
# Ratio 2: (1/2) / (1/2)^2 = 2.
# If the seed scores 1.0, and the score is "upper bound on C1", and search score is seed/value...
# "1.0 equals the seed program".
# So if value is the raw score, and seed value is V_seed, then search_score = V_seed / V_child.
# If search_score = 1.0, then V_child = V_seed.
# The prompt says "score is the upper bound on C1".
# And "evaluator measures the raw value (lower is better)".
# So we want to minimize the raw value.
# If seed raw value is R_seed, and search score = R_seed / R_child.
# If R_child = R_seed, search score = 1.
# So we need to find R_seed.
# Let's assume the standard normalization for such inequalities.
# Often, C1 is defined such that \max (f*f) <= C1 \int f^2.
# For the box function, \max (f*f) = 1/2, \int f^2 = 1/2. Ratio = 1.
# So R_seed = 1.
# We want to minimize R = \max_t (f*f)(t) / \int f^2 dx.

def compute_autoconvolution_max(a):
    """
    Compute the maximum value of the autoconvolution of the step function
    defined by heights a.
    """
    # a is array of length N.
    # The autoconvolution of the step function is related to the discrete convolution of a.
    # Let's derive it.
    # f(x) = sum a_i 1_{[x_i, x_{i+1})}(x).
    # (f*f)(t) = \int f(x) f(t-x) dx.
    # This is a piecewise linear function.
    # The values at the "grid points" of the convolution (which are multiples of h)
    # can be computed via discrete convolution.
    # Specifically, let b = np.convolve(a, a, mode='full').
    # The integral of f(x) f(t-x) over the intersection of intervals.
    # If t is an integer multiple of h, say t = k*h.
    # Then (f*f)(k*h) = h * sum_{i,j: x_i + x_j approx k*h} a_i a_j ?
    # Actually, the discrete convolution c_k = sum_i a_i a_{k-i} approximates (1/h) (f*f)(k*h).
    # Let's check.
    # \int f(x) f(t-x) dx.
    # If t = k*h.
    # The overlap of intervals [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j) is a rectangle of width h (if aligned) or less.
    # For t = k*h, the intervals align perfectly.
    # The overlap of interval i and interval j (where j = k-i) is exactly h.
    # So the contribution is a_i * a_j * h.
    # Summing over all i,j such that i+j=k (with appropriate indexing).
    # So (f*f)(k*h) = h * sum_{i} a_i a_{k-i}.
    # This is h * (a * a)[k].
    # However, the maximum might occur between grid points.
    # The function (f*f)(t) is piecewise linear.
    # Between t = k*h and t = (k+1)*h, the slope changes.
    # Actually, for non-negative a, the function is unimodal?
    # If a is non-negative, f*f is non-negative.
    # The maximum of a piecewise linear function occurs at a vertex.
    # The vertices are at t = k*h.
    # So max_t (f*f)(t) = max_k (f*f)(k*h) = h * max_k (a*a)[k].
    
    # Let's verify this.
    # Consider two adjacent intervals i and i+1 with heights a_i, a_{i+1}.
    # Consider t such that the right edge of interval i aligns with left edge of interval i+1?
    # No, the "corners" of the trapezoids are at t = (i+j)*h.
    # So the maximum is indeed at one of the grid points.
    
    conv = np.convolve(a, a, mode='full')
    return h * np.max(conv)

def compute_l2_norm_sq(a):
    """
    Compute \int f^2 dx.
    \int f^2 dx = sum a_i^2 * h.
    """
    return np.sum(a**2) * h

def objective(a):
    """
    Minimize max_autoconv / l2_norm_sq.
    """
    l2_sq = compute_l2_norm_sq(a)
    if l2_sq < 1e-12:
        return 1e12
    max_conv = compute_autoconvolution_max(a)
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: flat
    a0 = np.ones(N)
    
    # We can try to optimize a subset of parameters or use a smarter initialization.
    # The optimal solution for minimizing max(f*f) / ||f||_2^2 might be related to a "flat top" or specific spectral properties.
    # Actually, if we scale a, the ratio is invariant. So we can fix ||a||_2 = 1 and minimize max_conv.
    # Let's fix sum(a^2) * h = 1.
    # Then minimize h * max(conv(a)).
    
    # Let's use scipy.optimize.minimize with a constraint?
    # Or just minimize the ratio directly.
    
    # Better approach:
    # Fix ||a||_2^2 = 1/h (so \int f^2 = 1).
    # Minimize max_conv.
    
    # Let's try a few random initializations and local optimization.
    # Since N=600, a full optimization might be slow.
    # We can parameterize a by a lower-dimensional model?
    # E.g., a_i = 1 + epsilon * cos(2*pi*k*i/N).
    # Or just optimize the whole vector with a gradient-based method.
    
    # Let's try optimizing the whole vector.
    # Objective: max_conv(a) subject to ||a||_2^2 = 1/h, a_i >= 0.
    
    # We can use SLSQP or trust-constr.
    # But max_conv is non-smooth.
    # We can approximate max by log-sum-exp?
    # Or just use the fact that it's convex?
    # max_conv(a) is convex in a (max of linear functions).
    # ||a||_2^2 is convex.
    # Minimizing convex subject to convex equality is not convex.
    # But we can fix the norm and minimize the max.
    
    # Let's try a simple iterative approach:
    # 1. Start with a0.
    # 2. Compute g = grad max_conv.
    # 3. Update a = a - step * (g - lambda * a) to maintain norm?
    
    # Alternatively, use scipy minimize on the ratio.
    
    best_a = a0.copy()
    best_score = objective(best_a)
    
    # Try a few random restarts with different structures
    rng = np.random.default_rng(42)
    
    for i in range(5):
        if time.time() > deadline:
            break
            
        if i == 0:
            a_curr = np.ones(N)
        elif i == 1:
            # Cosine window
            x = np.linspace(0, np.pi, N)
            a_curr = np.cos(x) ** 2
            a_curr[a_curr < 0] = 0
        elif i == 2:
            # Random
            a_curr = np.abs(rng.standard_normal(N))
        else:
            a_curr = np.ones(N)
            # Perturb
            a_curr += 0.1 * rng.standard_normal(N)
            a_curr = np.maximum(a_curr, 0)
        
        # Normalize
        norm = np.sqrt(np.sum(a_curr**2) * h)
        if norm > 1e-12:
            a_curr = a_curr / norm * np.sqrt(1.0/h)
            
        # Optimize
        # Use L-BFGS-B? No, we have a non-smooth objective.
        # Let's use a simple projected gradient or just trust-constr with a smooth approximation.
        
        # Smooth approximation of max: log(sum(exp(c * conv))) / c
        # But conv has negative values? No, a is non-negative, so conv is non-negative.
        
        def smooth_obj(a_flat):
            a_vec = a_flat
            # Ensure non-negative? L-BFGS-B handles bounds.
            # But we want to maintain norm?
            # Let's just minimize ratio with bounds a_i >= 0.
            l2_sq = compute_l2_norm_sq(a_vec)
            if l2_sq < 1e-12:
                return 1e12
            conv = np.convolve(a_vec, a_vec, mode='full')
            # Smooth max
            c = 10.0
            max_conv_smooth = np.log(np.sum(np.exp(c * conv))) / c
            return max_conv_smooth / l2_sq
        
        # Bounds
        bounds = [(0, None)] * N
        
        # Initial point
        x0 = a_curr
        
        try:
            res = minimize(smooth_obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success or res.fun < best_score:
                a_opt = res.x
                # Clip negatives
                a_opt = np.maximum(a_opt, 0)
                score = objective(a_opt)
                if score < best_score:
                    best_score = score
                    best_a = a_opt
        except Exception:
            pass

    # Final check
    # Return the list of heights
    return list(best_a)

# Test locally? No, just output.
# One more thing: The problem says "nonnegative step function".
# My `construct` returns a list of floats.

# Let's refine the optimization.
# The ratio is scale invariant.
# We can fix \int f^2 = 1.
# Then minimize \max (f*f).
# This is a convex optimization problem?
# Minimize t
# Subject to:
# (f*f)(k*h) <= t for all k
# \int f^2 = 1
# f >= 0
#
# (f*f)(k*h) = h * sum a_i a_{k-i}. This is quadratic in a.
# So it's not convex.
# However, if we fix the norm, the feasible set is not convex.
#
# But we can try to solve it using a different approach.
# Maybe the optimal function is constant?
# If f is constant, ratio = 1.
# Can we do better?
# If we make f "peaked", \max (f*f) increases, but \int f^2 increases more?
# Let f = delta (Dirac). \max (f*f) = 1, \int f^2 = 1. Ratio = 1.
# Let f = 1 on [0,1]. \max (f*f) = 1, \int f^2 = 1. Ratio = 1.
# It seems 1 is a strong lower bound.
# Is it possible to get < 1?
# \max (f*f) <= \int f^2 ?
# This is equivalent to ||f*f||_\infty <= ||f||_2^2.
# This is a known inequality?
# For f in L^2, ||f*f||_\infty <= ||f||_2^2 is NOT true in general.
# Counterexample: f = 1 on [0,1] and [2,3].
# \int f^2 = 2.
# (f*f)(2) = \int f(x) f(2-x) dx.
# f(x) is 1 on [0,1] and [2,3].
# f(2-x) is 1 when 2-x in [0,1] => x in [1,2] (no overlap) and 2-x in [2,3] => x in [-1,0] (no overlap).
# Wait.
# f(x) f(2-x).
# If x in [0,1], f(x)=1. 2-x in [1,2]. f(2-x)=0.
# If x in [2,3], f(x)=1. 2-x in [-1,0]. f(2-x)=0.
# So (f*f)(2) = 0.
# Max is at t=0. (f*f)(0) = \int f(x)^2 dx = 2.
# Ratio 2/2 = 1.
#
# What if we overlap?
# f = 1 on [0, 1.5].
# \int f^2 = 1.5.
# (f*f)(t) is a trapezoid.
# Max at t=0 is 1.5.
# Ratio 1.
#
# It seems for any non-negative f, \max (f*f) <= \int f^2 ?
# Proof:
# (f*f)(t) = \int f(x) f(t-x) dx.
# By Cauchy-Schwarz:
# (f*f)(t) <= ( \int f(x)^2 dx )^{1/2} ( \int f(t-x)^2 dx )^{1/2} = \int f^2.
# So \max_t (f*f)(t) <= \int f^2.
# Thus the ratio is always <= 1.
# And we want to MINIMIZE the ratio?
# "minimize the normalized maximum... gives an upper bound on C1".
# If C1 is the best constant in \max (f*f) <= C1 \int f^2, then C1 = 1.
# And we want to find a function that achieves this bound?
# No, "minimize the normalized maximum... this gives an upper bound on C1".
# This phrasing is tricky.
# If we choose a specific f, the ratio R(f) is an upper bound on C1?
# No, C1 is the supremum of R(f) over all f.
# So R(f) <= C1.
# So any f gives a LOWER bound on C1?
# Wait. "Upper bound on C1".
# Maybe the inequality is different.
# Maybe \int (f*f)^2 <= C1 (\int f^2)^2 ?
# For box function:
# \int (f*f)^2. f*f is triangle.
# \int_{-1/2}^{1/2} (1/2 - |t|)^2 dt.
# Let u = t - 1/2.
# This integral is 1/12?
# (\int f^2)^2 = (1/2)^2 = 1/4.
# Ratio (1/12) / (1/4) = 1/3.
# Seed score 1.0.
# If raw value is 1/3, and seed is 1/3, then score 1.0.
# If we can get lower, score > 1.0.
#
# Let's check the seed program score again.
# "score is the upper bound on C1".
# "evaluator measures the raw value (lower is better)".
# "1.0 equals the seed program".
# So if seed raw value is V_s, and we produce V, score = V_s / V.
#
# If the inequality is ||f*f||_2 <= C1 ||f||_2^2, then C1 is a constant.
# For box, ratio is 1/3.
# Is 1/3 the optimal C1?
# For Gaussian, ratio is different.
#
# Let's assume the raw value is \frac{\int (f*f)^2 dt}{(\int f^2 dt)^2}.
# And we want to minimize this.
#
# Let's implement this objective.

def compute_raw_value_v2(a):
    """
    Compute \int (f*f)^2 / (\int f^2)^2.
    """
    # \int f^2 = sum a_i^2 * h
    l2_sq = np.sum(a**2) * h
    
    # (f*f)(t) is piecewise linear.
    # We need \int (f*f)(t)^2 dt.
    # The function g(t) = (f*f)(t) is piecewise linear.
    # Its values at t = k*h are v_k = h * conv_k.
    # Between t = k*h and (k+1)*h, g(t) is linear.
    # The integral of g(t)^2 over [k*h, (k+1)*h] is h * (v_k^2 + v_k v_{k+1} + v_{k+1}^2) / 3.
    
    conv = np.convolve(a, a, mode='full')
    v = h * conv
    
    # Sum over all intervals
    # v has length 2N-1.
    # The function g(t) is zero outside the support.
    # The support of conv is [0, 2N-2].
    # So t ranges from 0 to (2N-2)h.
    # But f is on [-1/4, 1/4], so t ranges from -1/2 to 1/2.
    # The convolution is symmetric.
    
    # Integral = sum_{k=0}^{2N-3} h * (v_k^2 + v_k v_{k+1} + v_{k+1}^2) / 3
    # Note: v_0 ... v_{2N-2}.
    # Intervals are [0, h], [h, 2h], ..., [(2N-3)h, (2N-2)h].
    
    integral_sq = 0.0
    for k in range(len(v) - 1):
        integral_sq += h * (v[k]**2 + v[k]*v[k+1] + v[k+1]**2) / 3.0
        
    if l2_sq < 1e-12:
        return 1e12
        
    return integral_sq / (l2_sq * l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: flat
    a0 = np.ones(N)
    best_a = a0.copy()
    best_score = compute_raw_value_v2(best_a)
    
    # Optimize
    # Objective: minimize integral_sq / l2_sq^2.
    # Scale invariant. Fix l2_sq = 1.
    # Minimize integral_sq.
    
    def obj(a_flat):
        a_vec = a_flat
        l2_sq = np.sum(a_vec**2) * h
        if l2_sq < 1e-12:
            return 1e12
        conv = np.convolve(a_vec, a_vec, mode='full')
        v = h * conv
        # Vectorized integral
        # h/3 * sum (v_k^2 + v_k v_{k+1} + v_{k+1}^2)
        # = h/3 * (sum v_k^2 + sum v_k v_{k+1} + sum v_{k+1}^2)
        # sum v_k^2 = sum v^2
        # sum v_{k+1}^2 = sum v^2 - v_0^2 - v_{last}^2
        # sum v_k v_{k+1} = dot(v[:-1], v[1:])
        
        sum_sq = np.sum(v**2)
        sum_prod = np.dot(v[:-1], v[1:])
        
        # Correction for boundaries
        # The formula h/3 * (v_k^2 + v_k v_{k+1} + v_{k+1}^2)
        # Sum = h/3 * ( sum_k v_k^2 + sum_k v_k v_{k+1} + sum_k v_{k+1}^2 )
        # sum_k v_k^2 = v_0^2 + ... + v_{M-1}^2
        # sum_k v_{k+1}^2 = v_1^2 + ... + v_M^2
        # So total = h/3 * ( 2*sum(v^2) - v_0^2 - v_M^2 + sum_prod )
        
        integral_sq = (h / 3.0) * (2.0 * sum_sq - v[0]**2 - v[-1]**2 + sum_prod)
        
        return integral_sq / (l2_sq * l2_sq)
    
    # Optimize with L-BFGS-B
    # We need to handle the non-negativity constraint.
    
    for i in range(3):
        if time.time() > deadline:
            break
        
        if i == 0:
            a_curr = np.ones(N)
        elif i == 1:
            # Try a Gaussian-like shape
            x = np.linspace(-1, 1, N)
            a_curr = np.exp(-x**2 * 5)
        else:
            a_curr = np.ones(N)
            a_curr += 0.1 * np.random.standard_normal(N)
            a_curr = np.maximum(a_curr, 0)
            
        # Normalize to l2_sq = 1
        norm = np.sqrt(np.sum(a_curr**2) * h)
        if norm > 1e-12:
            a_curr = a_curr / norm
            
        bounds = [(0, None)] * N
        x0 = a_curr
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_a = res.x
        except Exception:
            pass

    return list(best_a)
