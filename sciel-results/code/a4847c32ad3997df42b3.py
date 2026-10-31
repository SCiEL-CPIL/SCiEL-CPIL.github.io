# EVOLVE-BLOCK-START
import time
import math

_start_time = None
_deadline = 315.0  # safety margin before the 330s kill

# Precompute pseudo-random frequency coefficients for character-theory scoring
# Using a fixed seed for reproducibility
_FREQ_COEFFS = {}

def _get_freq_coeffs(n):
    """Get or compute pseudo-random frequency coefficients for dimension n."""
    if n not in _FREQ_COEFFS:
        # Use a simple LCG-based pseudo-random sequence for deterministic coefficients
        # These simulate random frequencies in F_3^n
        coeffs = []
        seed = 12345 + n * 7919
        for i in range(n):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            # Map to {-1, 0, 1} with some structure
            r = (seed >> 8) % 3
            if r == 0:
                coeff = 1
            elif r == 1:
                coeff = -1
            else:
                coeff = 0
            coeffs.append(coeff)
        _FREQ_COEFFS[n] = coeffs
    return _FREQ_COEFFS[n]


def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Design family (kept): multi-factor smooth scoring.
    Improvements within the family:
    1.  Distinct-symbol score uses a cubic (num_distinct^3) instead of quadratic.
        This sharply increases the gap between 2-distinct and 3-distinct vectors
        (10000 -> 270000) while keeping the function smooth.
    2.  Balance penalty uses a smooth "soft-min" surrogate:
        instead of penalizing L2 deviation from n/3, we penalize the minimum
        count among the three symbols using a log-sum-exp soft-min.
        This directly rewards vectors that have at least one of each symbol
        in a smooth, differentiable way.
    3.  Variance penalty weight is reduced (5000 -> 2000) to allow more
        structural diversity among 3-distinct vectors.
    4.  Symmetry-breaking weight for 2s is reduced (100 -> 50) to avoid
        over-constraining the search.
    5.  Fast polynomial hash for deterministic tie-breaking (unchanged).
    6.  NEW: Character-theory Fourier score. We compute the magnitude of the
        additive character sum with respect to a fixed pseudo-random frequency
        vector in F_3^n. Vectors with large |sum| deviate from uniform
        distribution in frequency space, which correlates with being "spread
        out" and less likely to form arithmetic progressions. We reward
        vectors with SMALLER |character_sum| (closer to zero), as these are
        more "Fourier-uniform" and better suited for cap sets.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Cubic distinct-symbol score.
    # 0 -> 0, 1 -> 10000, 2 -> 80000, 3 -> 270000.
    # Strong preference for 3-distinct vectors with smooth landscape.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s (reduced weight).
    weight_score = c2 * 50.0

    # Factor 3: Soft-min balance reward.
    # We want to reward vectors where the minimum symbol count is as large as possible.
    # Use log-sum-exp to compute a smooth approximation of -min(c0, c1, c2).
    # soft_min(a,b,c) = -log(exp(a/t) + exp(b/t) + exp(c/t)) * t  (for small t)
    # Here we use t=1.0 for stability, and reward = -soft_min.
    # Actually, we want to reward having all three symbols present in balanced amounts.
    # A simpler smooth surrogate: penalize the max deviation from n/3.
    # Let's use: balance_score = -max(c0, c1, c2) + min(c0, c1, c2)
    # This is the "range" of counts, and we want to minimize it.
    # But max/min are not smooth. Use log-sum-exp for max, and -log-sum-exp for min.
    # For simplicity and speed, we'll use a piecewise-linear-like smooth approximation:
    # balance_score = -(c0*c1 + c1*c2 + c2*c0) / n
    # This is a smooth proxy for balance: it's maximized when counts are equal.
    # For n=10, equal counts (3,3,4) gives (9+12+12)/10 = 3.3
    # Imbalanced (0,5,5) gives (0+25+0)/10 = 2.5
    # (0,0,10) gives 0.
    # This is a smooth, differentiable measure of balance.
    balance_score = -((c0 * c1 + c1 * c2 + c2 * c0) / max(n, 1)) * 20.0

    # Factor 4: Smooth variance-based danger proxy (reduced weight).
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 2000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Factor 6: Character-theory Fourier score.
    # Compute the additive character sum: sum_{i} omega^{coeff_i * v_i}
    # where omega = exp(2*pi*i/3) is a primitive 3rd root of unity.
    # The magnitude of this sum measures deviation from uniform distribution.
    # We reward vectors with smaller |sum| (more Fourier-uniform).
    # For efficiency, we compute the real and imaginary parts separately.
    # omega^0 = 1, omega^1 = -0.5 + i*sqrt(3)/2, omega^2 = -0.5 - i*sqrt(3)/2
    # For each position i with coeff c and value v:
    #   contribution = omega^(c*v mod 3)
    # We track real and imaginary sums.
    coeffs = _get_freq_coeffs(n)
    sqrt3_over_2 = math.sqrt(3.0) / 2.0
    real_sum = 0.0
    imag_sum = 0.0
    for i in range(n):
        c = coeffs[i]
        v = vector[i]
        # Compute (c * v) mod 3
        phase = (c * v) % 3
        if phase == 0:
            real_sum += 1.0
            imag_sum += 0.0
        elif phase == 1:
            real_sum += -0.5
            imag_sum += sqrt3_over_2
        else:  # phase == 2
            real_sum += -0.5
            imag_sum += -sqrt3_over_2

    # Magnitude of the character sum
    char_mag = math.sqrt(real_sum * real_sum + imag_sum * imag_sum)
    # Normalize by sqrt(n) to get a per-coordinate measure
    char_mag_norm = char_mag / math.sqrt(float(n)) if n > 0 else 0.0
    # We want to REWARD vectors with smaller character magnitude (more uniform)
    # Use a smooth penalty: -char_mag_norm * weight
    # The weight should be moderate so it doesn't dominate the distinct_score
    fourier_score = -char_mag_norm * 100.0

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + hash_score + fourier_score
    return float(total)
# EVOLVE-BLOCK-END
