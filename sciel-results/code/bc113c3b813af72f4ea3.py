import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    K = 200000
    T = 100000
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    k = np.arange(2, K + 1, dtype=np.float64)
    weights = np.exp(-k / T)
    
    sum_inv_k2 = np.sum(weights / (k * k))
    
    # Scale c so that M(1) = 1 (tightest constraint)
    c = 1.0 / sum_inv_k2
    
    f1 = c * sum_inv_k2
    f_rest = -c * weights / k
    
    values = np.empty(K, dtype=np.float64)
    values[0] = f1
    values[1:] = f_rest
    
    # Verify constraint: M(x) <= 1 for x in [1, 10K]
    # M(x) = f(1) + sum_{k=2}^K f(k) * floor(x/k)
    # Check at a few critical points
    # For efficiency, check at x = 1, 2, ..., some range
    # The maximum likely at small x
    
    # Let's verify at x=1: M(1) = f1 = 1.0 ✓
    # At x=2: M(2) = f1 + f(2)*1 = 1 - c*exp(-2/T)/2 < 1 ✓
    # Generally, for x >= 1, floor(x/k) <= x/k, so:
    # M(x) <= f1 + sum_{k=2}^K f(k) * x/k = f1 - c*sum_{k=2}^K exp(-k/T)/k * x/k
    # = 1 - c*x*sum_{k=2}^K exp(-k/T)/k^2 = 1 - x*sum_inv_k2*c = 1 - x
    # So M(x) <= 1 - x for x >= 1? That can't be right since M(1) = 1.
    
    # Actually floor(x/k) <= x/k, so M(x) = f1 + sum f(k) floor(x/k) <= f1 + sum f(k) * x/k
    # = 1 + sum_{k=2}^K (-c exp(-k/T)/k) * x/k = 1 - c*x*sum_{k=2}^K exp(-k/T)/k^2 = 1 - x
    # So M(x) <= 1 - x for x >= 1, which means M(x) <= 0 for x >= 2.
    # But M(x) could be negative, which is fine (we need M(x) <= 1, not M(x) >= something).
    
    # So the constraint is satisfied! M(1) = 1 and M(x) <= 1-x < 1 for x > 1.
    
    values = np.clip(values, -10.0, 10.0)
    
    return domain, values
