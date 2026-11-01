"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    
    The autoconvolution of h is c = h * h (discrete convolution).
    The upper bound on C3 is typically related to the ratio of the L1 norm of the
    autoconvolution to the square of the L1 norm of h, or similar functional.
    
    For a signed step function, a common bound involves:
    C3 <= max_k |c_k| / (sum |h_i|)^2 * (some normalization)
    
    However, given the task mentions "upper bound on C3" and the seed returns np.ones(N),
    let's consider the standard definition in additive combinatorics context:
    For a function f, the autoconvolution is f*f(x) = sum_i f(i) f(x-i).
    The "C3" bound often refers to bounds involving the L3 norm or similar.
    
    A common metric for such problems is:
    score = max_k (h*h)[k] / (sum(h)^2)
    
    But since h is signed, sum(h) could be small or zero. Let's look at the absolute value placement.
    "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
    
    Let's assume the metric is:
    upper_bound = max_k |h*h[k]| / (sum_i |h_i|)^2
    
    For the seed program h = np.ones(N):
    h*h is a triangular convolution.
    sum_i |h_i| = N = 400.
    max_k |h*h[k]| = N (at the peak, where all 400 terms overlap).
    So upper_bound = 400 / (400^2) = 1/400 = 0.0025.
    
    If we can make this smaller, we improve the score (since lower is better for the raw value, 
    but the search score is seed/value, so if value decreases, score increases).
    
    Wait, the task says: "score is the upper bound on C3" and "evaluator measures the raw value (lower is better)".
    And "search score is seed/value".
    So if value is the upper bound, and we want to minimize it, then a smaller value means a higher search score.
    
    Let's verify with the seed:
    Seed returns np.ones(N).
    Value = 1/400 = 0.0025.
    Search score = seed_value / value = 0.0025 / 0.0025 = 1.0.
    
    So we need to minimize:
    f(h) = max_k |convolve(h, h)[k]| / (sum(|h|))^2
    
    Note: convolve(h, h) has length 2N-1.
    
    This is a non-convex problem due to the max and the absolute values.
    However, we can use gradient-based optimization by smoothing the max and abs, or by using subgradients.
    Since scipy's minimize doesn't handle non-differentiable functions well directly, we might need to:
    1. Use a smooth approximation of max and abs.
    2. Or use a different approach: notice that for a fixed denominator, minimizing the max of the convolution peaks is key.
    
    Alternatively, we can parameterize h to be bounded, e.g., h_i in [-1, 1], and use L-BFGS-B.
    
    Let's define:
    g(h) = log(sum(|h|)^2) - log(max_k |conv(h, h)_k|)
    and maximize g(h), or minimize -g(h).
    
    But log(max) is non-smooth. We can use the soft-max:
    log(sum(exp(x_k / T))) * T ≈ max(x_k) for large T.
    Similarly, |x| ≈ sqrt(x^2 + eps).
    
    Let's try:
    def objective(h):
        h = np.tanh(h)  # Keep h in [-1, 1]
        conv = np.convolve(h, h, mode='full')
        abs_conv = np.sqrt(conv**2 + 1e-10)
        # Soft max
        T = 10.0
        log_max = T * np.log(np.sum(np.exp(abs_conv / T)))
        sum_abs_h = np.sum(np.sqrt(h**2 + 1e-10))
        value = log_max / (sum_abs_h**2)
        return value
    
    Then we minimize this value.
    
    Initial guess: h = np.ones(N)
    """
    def objective(x):
        h = np.tanh(x)  # Bounded in (-1, 1)
        conv = np.convolve(h, h, mode='full')
        # Use smooth absolute value and smooth max
        eps = 1e-8
        abs_conv = np.sqrt(conv**2 + eps)
        # Soft max with temperature
        T = 5.0
        # To avoid overflow, subtract max
        max_abs = np.max(abs_conv)
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = log_max / (sum_abs_h**2)
        return value
    
    # Initial guess: all ones in the tanh parameter space
    # tanh(x) = 1 => x = inf, so use a large number
    x0 = np.full(N, 5.0)  # tanh(5) ≈ 0.9999
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    
    h_opt = np.tanh(res.x)
    return h_opt


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
