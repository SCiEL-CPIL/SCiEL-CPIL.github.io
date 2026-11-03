"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 600
# Interval width
dx = (1.0/4 - (-1.0/4)) / N  # 0.5 / 600 = 1/1200

def compute_score(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    heights: array of length N, nonnegative.
    
    The autoconvolution of f with itself is (f * f)(t) = integral f(s)f(t-s) ds.
    For a step function on equal intervals, we can compute the discrete convolution.
    
    Let's think about the normalization. The task says "normalized maximum of its autoconvolution".
    Typically, for such problems, the normalization might involve the L1 norm or L2 norm.
    
    Given the context of "First autocorrelation inequality" and "upper bound on C1", this likely relates to the 
    inequality involving the autocorrelation. Let's assume the standard normalization:
    score = max(f*f) / (integral f)^2 ? Or max(f*f) / (integral f^2)?
    
    Actually, let's look at the seed: [1.0]*N.
    If f=1 on [-1/4, 1/4], integral f = 1/2.
    f*f is the triangle function with peak at 0.
    (f*f)(0) = integral_{-1/4}^{1/4} 1*1 ds = 1/2.
    If normalized by (integral f)^2 = (1/2)^2 = 1/4, score = (1/2)/(1/4) = 2.
    If normalized by integral f^2 = 1/2, score = (1/2)/(1/2) = 1.
    
    The problem asks for an "upper bound on C1". In many contexts, C1 is a constant in an inequality like:
    max |f*f| <= C * ||f||_1^2 ?
    If so, score = max(f*f) / ||f||_1^2.
    
    Let's assume score = max_conv / (sum(heights)*dx)^2.
    """
    # Compute discrete convolution
    # f is defined on [-1/4, 1/4]. The support of f*f is [-1/2, 1/2].
    # The grid for f has N points. The grid for f*f will have 2N-1 points if we align centers.
    # However, since we are using step functions, we can treat heights as values on intervals.
    
    # Let's use np.convolve.
    # Note: np.convolve does 'full' convolution by default.
    conv = np.convolve(heights, heights, mode='full')
    max_conv = np.max(conv)
    
    # Integral of f
    integral_f = np.sum(heights) * dx
    
    # Normalization
    if integral_f == 0:
        return float('inf')
        
    score = max_conv / (integral_f * integral_f)
    return score

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We want to minimize the score.
    # The problem is high-dimensional (600 variables).
    # Direct optimization is intractable.
    
    # Strategy: Use a structural change.
    # Instead of optimizing 600 variables, we can exploit symmetry or specific structures.
    # The seed is a uniform step. 
    # A common structure in these problems is to use a function that is 1 on a subset and 0 elsewhere, 
    # or a smooth bump.
    
    # Let's try to optimize a lower-dimensional parameterized family of functions.
    # Family 1: Symmetric step functions with a "hole" in the middle or edges.
    # Family 2: Gaussian-like bumps (approximated by steps).
    
    # Let's try a parameterized family: f(x) = 1 if |x| <= r, else 0.
    # This is a symmetric interval.
    # The number of intervals filled is k.
    # Due to symmetry, k should be even or odd depending on centering.
    # The intervals are centered at -1/4 + (i+0.5)*dx for i=0..N-1.
    # Center of the domain is 0.
    # Indices near N/2 are near 0.
    
    # Let's parameterize by the number of non-zero intervals centered at 0.
    # Let m be the number of intervals.
    # If m is even, we take m/2 on the left and m/2 on the right.
    # If m is odd, we take (m-1)/2 on each side and 1 in the middle? 
    # Actually, the grid is fixed. Let's just pick a contiguous block of indices centered at the middle.
    
    # Let's define a function that generates heights for a given "width" parameter.
    # Width w in [0, 0.5].
    # Number of intervals = int(w / dx).
    
    def make_box(w):
        # w is the half-width of the box? No, let's say w is the total width.
        # Support is [-w/2, w/2].
        # Indices where center is in [-w/2, w/2].
        centers = -0.25 + (np.arange(N) + 0.5) * dx
        mask = np.abs(centers) <= w / 2.0
        return mask.astype(float)
    
    # Let's optimize over the width of a box function.
    # This is a 1D optimization problem.
    # Is the box function optimal? 
    # For L1-normalized autoconvolution max, the box function (indicator of an interval) 
    # gives a triangular convolution. 
    # max(f*f) = w (for box of width w, height 1).
    # ||f||_1 = w.
    # Score = w / w^2 = 1/w.
    # To minimize 1/w, we want w to be as large as possible.
    # Max w is 0.5 (the whole domain).
    # Score = 1/0.5 = 2.
    # This matches our earlier calculation for the seed.
    
    # So the box function is not good for minimizing if we can vary the shape.
    # What if we use a Gaussian?
    # f(x) = exp(-x^2/sigma^2).
    # f*f is also Gaussian-like.
    # max(f*f) is proportional to 1/sigma.
    # ||f||_1 is proportional to sigma.
    # Score ~ (1/sigma) / sigma^2 = 1/sigma^3?
    # Let's check.
    # f(x) = exp(-x^2/a).
    # integral f = sqrt(pi*a).
    # (f*f)(0) = integral exp(-s^2/a) exp(-(0-s)^2/a) ds = integral exp(-2s^2/a) ds = sqrt(pi*a/2).
    # Score = sqrt(pi*a/2) / (pi*a) = 1 / (sqrt(2*pi*a)).
    # To minimize score, we want to maximize a.
    # But the domain is limited to [-0.25, 0.25].
    # If a is very large, f is nearly constant 1. Score -> 2.
    # If a is small, f is a spike. 
    # Let's check the limit a -> 0.
    # f becomes a delta function. 
    # If we approximate a delta with height H and width w, H*w = 1 (L1 norm 1).
    # Then f*f has peak H^2 * w.
    # Score = (H^2 * w) / 1 = H^2 * w = (1/w)^2 * w = 1/w.
    # As w -> 0, Score -> infinity.
    
    # So it seems the minimum is at the boundary w=0.5, Score=2?
    # Wait, the seed score is 2.0?
    # Let's re-read the task.
    # "minimize the normalized maximum of its autoconvolution"
    # "upper bound on C1"
    # "seed/value, so 1.0 equals the seed program"
    # Seed program returns [1.0]*N.
    # If seed score is 2.0, and we get 2.0, ratio is 1.0.
    # If we can get a score lower than 2.0, the ratio will be > 1.0.
    
    # Is it possible to get a score lower than 2.0?
    # Let's test a different shape.
    # What if f is not symmetric?
    # What if f has a "dip" in the middle?
    
    # Let's consider the function f(x) = 1 for |x| in [a, b] and 0 elsewhere.
    # This is a ring.
    # Let's try to optimize a general shape using a few parameters.
    
    # Since we have 600 variables, we can try to use a low-rank parameterization.
    # Or, we can try to optimize the shape directly using gradient descent on the discrete variables, 
    # but the objective is non-smooth (max).
    
    # Let's try a different family: 
    # f(x) = (1 - |x|/b)^p for |x|<b, 0 otherwise.
    # This is a generalized triangle.
    
    # Actually, let's look at the "First autocorrelation inequality".
    # This might refer to the inequality:
    # max |f*f| <= C * ||f||_1^2
    # The best constant C is what we are trying to bound.
    # It is known that for L1 functions, the constant is not 2.
    # Wait, if f is an indicator of a set E, f*f is the measure of E intersect (E-x).
    # The maximum is the maximum chord length.
    # For a set of measure 1/2 in a domain of length 1/2? No, domain is [-1/4, 1/4].
    
    # Let's try to minimize the score by making f as "spread out" as possible, but we are limited to the domain.
    # The most spread out is the uniform function, which gives 2.
    
    # Is there a function that gives a lower score?
    # Consider f(x) = 1/2 + 0.5*cos(2*pi*x/L).
    # This is non-negative.
    # Let's calculate the score for a cosine bump.
    
    # Let's write a helper to compute score for a given array.
    
    # Let's try to optimize a small number of parameters for a smooth function.
    # Parameters: amplitude, width, center (fixed at 0), skew.
    
    # Let's try: f(x) = A * (1 - (x/b)^2)^k for |x|<b.
    
    # Let's use scipy.optimize.minimize_scalar on the width b and shape k.
    
    def make_bump(b, k):
        # b in (0, 0.25]
        # k > 0
        centers = -0.25 + (np.arange(N) + 0.5) * dx
        x = centers
        mask = np.abs(x) < b
        heights = np.zeros(N)
        heights[mask] = (1 - (x[mask]/b)**2)**k
        # Normalize? No, the score is scale-invariant if we normalize by ||f||_1^2.
        # But the problem asks for nonnegative step heights.
        # The score is max(f*f) / (integral f)^2.
        # This is invariant under scaling f -> c*f.
        # So we can fix the scale, e.g., max height = 1.
        return heights
    
    # Let's optimize over b and k.
    # b in [0.01, 0.25]
    # k in [0.1, 10]
    
    # We can use a grid search or nested minimize_scalar.
    
    best_score = float('inf')
    best_heights = np.ones(N)
    
    # Try a few values of k
    for k in [0.5, 1.0, 2.0, 5.0]:
        def obj(b):
            h = make_bump(b, k)
            s = compute_score(h)
            if s < best_score:
                # Update best? Careful with closures.
                pass
            return s
        
        res = minimize_scalar(obj, bounds=(0.01, 0.25), method='bounded')
        if res.success:
            h = make_bump(res.x, k)
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_heights = h
    
    # Also try the box function (k=inf effectively, or just step)
    # We already know box gives 2.0.
    
    # What if we try a function that is 1 on two intervals?
    # f(x) = 1 for |x| in [a, b].
    
    def make_ring(a, b):
        centers = -0.25 + (np.arange(N) + 0.5) * dx
        x = np.abs(centers)
        mask = (x >= a) & (x <= b)
        return mask.astype(float)
    
    # Optimize a and b.
    # a in [0, 0.2], b in [a, 0.25]
    
    # This is 2D. We can use nested loops or minimize.
    # Given time constraints, let's just try a few points.
    
    for a in [0.0, 0.05, 0.1]:
        for b in [a + 0.05, a + 0.1, 0.25]:
            if a >= 0.25 or b > 0.25:
                continue
            h = make_ring(a, b)
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_heights = h
    
    # Check time
    if time.time() > deadline:
        return list(best_heights)
    
    # If best_score is still inf, return seed.
    if best_score == float('inf'):
        return list(np.ones(N))
    
    # Return the best heights found.
    # The task asks for a sequence of nonnegative step heights.
    return list(best_heights)
# EVOLVE-BLOCK-END
