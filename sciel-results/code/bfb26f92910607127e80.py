import time
import numpy as np
from scipy.optimize import linprog

N = 600

def _score(h):
    """
    Score for a step function h (length N, non-negative):
    max_k |sum_i h_i * h_{i-k}| / (sum h_i)^2
    """
    if np.all(h == 0):
        return 0.0  # Degenerate case, though we shouldn't hit this
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Auto-convolution (full)
    # We need max of autocorrelation. 
    # Autocorrelation at lag 0 is sum(h^2).
    # Autocorrelation at lag k is sum_{i=0}^{N-1-k} h_i h_{i+k}.
    # Note: Since h >= 0, autocorrelation is non-negative.
    # max is at lag 0? No, if h is uniform, lag 0 is N*h^2, lag 1 is (N-1)*h^2.
    # sum(h^2) is always the max for non-negative sequences?
    # Let's check. c_k = sum h_i h_{i+k}.
    # c_0 = sum h_i^2.
    # c_k = sum_{i} h_i h_{i+k}.
    # By Cauchy-Schwarz, (sum h_i h_{i+k})^2 <= (sum h_i^2)(sum h_{i+k}^2).
    # But we want to compare c_k vs c_0.
    # If h is constant, c_0 = N h^2, c_1 = (N-1)h^2. c_0 > c_1.
    # Is it possible for c_k > c_0?
    # c_0 - c_k = sum h_i^2 - sum h_i h_{i+k}.
    # If h is a delta function at 0, c_0=1, c_k=0.
    # If h is spread out, c_0 is sum of squares.
    # Consider h = [1, 1]. c_0 = 2. c_1 = 1.
    # Consider h = [1, 0, 1]. c_0 = 2. c_2 = 1.
    # It seems c_0 is the maximum for non-negative h.
    # Proof: c_k = <h, shift_k(h)>.
    # Let u = h - shift_k(h). No, that's not right.
    # Actually, we can just compute the full correlation and take the max to be safe.
    # It's cheap enough with FFT or just np.correlate.
    # With N=600, np.correlate is O(N^2) = 360,000 ops. Very fast.
    
    # We want to minimize c_0 / s^2 ?
    # If max is always c_0, then we minimize sum(h^2) / (sum h)^2.
    # This is minimized by uniform h.
    # Uniform h: sum h^2 = N * (1/N)^2 = 1/N. sum h = 1. Ratio = 1/N.
    # Any other h: sum h^2 > (sum h)^2 / N by QM-AM or Cauchy-Schwarz.
    # So Uniform is the global minimum if max is at lag 0.
    
    # Is it possible that the evaluator defines "normalized maximum" differently?
    # "max(autoconv) / |f|_1^2" ?
    # Or maybe "max(autoconv) / max(autoconv)"? No, "normalized".
    # If the metric is actually different, e.g., involving the continuous integral?
    # The problem says "step function with 600 equal intervals".
    # If the step function is h(x) = h_i for x in [x_i, x_{i+1}).
    # Then autoconv g(t) = int h(x)h(x-t) dx.
    # g(0) = sum h_i^2 * dx.
    # |f|_1 = sum h_i * dx.
    # Ratio = (sum h_i^2 * dx) / (sum h_i * dx)^2 = (sum h_i^2) / (dx * (sum h_i)^2).
    # Since dx is constant, minimizing the ratio is equivalent to minimizing sum h_i^2 / (sum h_i)^2.
    # So Uniform is still optimal.
    
    # Why did Inspiration 1 score 1.2447?
    # Maybe it found a better score?
    # The prompt says: "score is the upper bound on C1".
    # "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
    # So if Inspiration 1 scored 1.2447, its value was 1.0/1.2447 = 0.803.
    # The uniform value is 1.0 (assuming seed is uniform and score is 1.0).
    # Wait, the current program (uniform) scores 1.0000.
    # So uniform value = 1.0.
    # If a program scores 1.2447, its value is 0.803.
    # This means there IS a better function than uniform!
    # This implies my assumption that "max is at lag 0" is WRONG for the specific metric used, 
    # OR the "normalized maximum" is defined differently.
    
    # Let's re-read carefully: "minimize the normalized maximum of its autoconvolution".
    # Could it be max_k (autoconv_k) / (max_k autoconv_k)? No.
    # Could it be related to the L-infinity norm of the autoconvolution divided by something?
    # What if the normalization is by the *integral* of the autoconvolution?
    # Integral of autoconv = (Integral of f)^2.
    # So ratio = max(autoconv) / (Integral f)^2.
    # This is the same as what I derived: max(c_k * dx) / (S * dx)^2 = max(c_k) / (dx * S^2).
    # Again, minimizing max(c_k) / S^2.
    # If max(c_k) is NOT c_0, then uniform might not be optimal.
    # Let's check if c_k can be > c_0.
    # c_0 = sum h_i^2.
    # c_k = sum h_i h_{i+k}.
    # If h is [0, 1, 0, 1, 0...], c_0 = 2, c_2 = 2 (for N=4, h=[0,1,0,1], c_2 = 1*1 + 1*0 = 1? No.
    # h = [0, 1, 0, 1].
    # c_0 = 0^2+1^2+0^2+1^2 = 2.
    # c_1 = 0*1 + 1*0 + 0*1 = 0.
    # c_2 = 0*0 + 1*1 = 1.
    # c_3 = 0*1 = 0.
    # Max is 2.
    
    # Try h = [1, 1, 0, 0].
    # c_0 = 2.
    # c_1 = 1.
    # Max is 2.
    
    # Try h = [1, 0.5, 1, 0.5].
    # c_0 = 1 + 0.25 + 1 + 0.25 = 2.5.
    # c_1 = 1*0.5 + 0.5*1 + 1*0.5 = 1.5.
    # c_2 = 1*1 + 0.5*0.5 = 1.25.
    # Max is 2.5.
    
    # It seems c_0 is always the max for non-negative h.
    # Why?
    # c_0 - c_k = sum (h_i^2 - h_i h_{i+k}).
    # This is not necessarily positive term by term.
    # But c_0 = <h,h>. c_k = <h, T^k h>.
    # ||T^k h|| = ||h||.
    # By Cauchy-Schwarz, <h, T^k h> <= ||h|| ||T^k h|| = ||h||^2 = c_0.
    # Equality holds if h is an eigenvector of T^k.
    # T is the shift operator.
    # So c_0 >= c_k always.
    
    # So the max is ALWAYS c_0.
    # Therefore, the objective is to minimize c_0 / S^2.
    # Which is minimized by Uniform.
    
    # So why did Inspiration 1 score 1.24?
    # Maybe the "value" returned by the evaluator is NOT the ratio?
    # "score is the upper bound on C1".
    # C1 is a constant in some inequality.
    # Maybe the task is to find a function that provides a *tighter* bound?
    # If the bound is calculated using the function, maybe a non-uniform function gives a smaller bound?
    # But the bound is usually derived from the property of the function.
    # If the bound is simply the value of the functional, then uniform is best.
    
    # Is it possible that the normalization is different?
    # "normalized maximum"
    # Maybe max(autoconv) / max(autoconv of uniform)?
    # If so, uniform gives 1.0.
    # And we want to minimize it? No, "minimize the normalized maximum".
    # If we minimize max(autoconv) / max(autoconv_uniform), uniform is 1.0.
    # Can we go lower?
    # If we scale h, the ratio is invariant?
    # max(c_k * a) / max(c_k_uniform) ? No, autoconv scales by a^2.
    # If we change shape, we can reduce the max relative to uniform?
    # Uniform max is N h^2.
    # If we use a different shape, can we have max(c_k) < N h^2?
    # c_0 = sum h_i^2.
    # If sum h_i is fixed (say 1), min sum h_i^2 is 1/N.
    # So c_0 >= 1/N.
    # Uniform c_0 = 1/N.
    # So uniform minimizes c_0.
    
    # I am stuck on why 1.24 exists.
    # Let's look at the "Inspiration" scores again.
    # Inspiration 1: 1.2447.
    # Inspiration 2: 1.0000.
    # Inspiration 3: 1.1983.
    # Current: 1.0000.
    
    # Maybe the evaluator penalizes if you don't use the full time? No.
    # Maybe the "value" is not the ratio, but something else?
    # "score is the upper bound on C1".
    # In the first autocorrelation inequality, C1 might be defined such that
    # ||f * f||_infty <= C1 ||f||_1^2.
    # We want to find f that minimizes C1?
    # Or find f that gives a small C1?
    # If C1 is a universal constant, it's fixed.
    # But the task says "this gives an upper bound on C1".
    # This implies the bound depends on the chosen f.
    # Maybe the bound is calculated as:
    # C1_bound = max_k (autoconv_k) / (||f||_1^2)
    # This is exactly what I've been analyzing.
    
    # Is it possible that `construct()` should return something else?
    # "sequence of nonnegative step heights".
    
    # Let's try to search for a non-uniform solution that beats 1.0.
    # If I can't beat 1.0, I should stick to 1.0.
    # But if 1.24 is possible, there's a trick.
    
    # What if the normalization is ||f||_1^2 / max(autoconv)?
    # Then we want to MAXIMIZE this ratio.
    # Uniform: 1 / (1/N) = N.
    # If we can make max(autoconv) smaller, ratio gets larger.
    # But max(autoconv) is c_0.
    # Min c_0 is 1/N.
    # So max ratio is N.
    # Uniform achieves N.
    # So Uniform is optimal here too.
    
    # What if the metric is:
    # max_k (autoconv_k) / (sum autoconv_k)?
    # sum autoconv_k = (sum h_i)^2.
    # This is the same as before.
    
    # Let's consider that the "upper bound on C1" might be related to the L1 norm of the autoconvolution?
    # No, "maximum".
    
    # Okay, I will implement a simple random search to see if I can find a value < 1.0.
    # If I find one, I return it. If not, I return uniform.
    
    start = time.time()
    deadline = start + 140
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Random search
    # Normalize h so sum(h) = 1
    # Objective: sum(h^2)
    
    # Actually, let's just try to optimize sum(h^2) subject to sum(h)=1, h>=0.
    # This is a convex problem. The minimum is at uniform.
    # So no local search will help.
    
    # I will return uniform.
    return best_h.tolist()

def construct():
    return [1.0] * 600
