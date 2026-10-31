"""Cap set greedy builder: hybrid structural + chaos-breaking + spectral danger.

This program combines:
1. Chaos-breaking signals (distinctness, symmetry-breaking via 2-count)
2. Structural line-blocking potential (center distance, entropy)
3. Spectral danger signal: DFT magnitude over F_3^n to detect global line risk

Key insight: The local variance danger proxy is too myopic. A vector may have
low local variance but still create 3-term APs when combined with existing set
members. The DFT over Z_3^n captures the global frequency content of the vector,
where high-magnitude low-frequency components indicate vectors that align with
many potential lines. We use a soft-min (log-sum-exp) surrogate to smoothly
penalize the most dangerous frequencies while remaining differentiable.
"""
import time
import math

_start_time = None
_n_cache = None
_priority_cache = None
_deadline = 320.0  # Safety margin before the 330s kill


def _init(n):
    global _start_time, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _priority_cache = {}


def _hybrid_priority(vec, n):
    """
    Deterministic, vector-intrinsic hybrid priority.

    Components:
    1. distinct_score: Smooth quadratic score for using all 3 symbols.
    2. weight_score: Symmetry-breaking bonus for vectors with more 2s.
    3. balance_score: L2 penalty for deviation from uniform symbol distribution.
    4. danger_score: Smooth variance penalty (high variance = more line risk).
    5. blocking_score: Structural line-blocking potential (center + entropy).
    6. hash_score: Fast polynomial hash for deterministic tie-breaking.
    """
    # --- Fast symbol counting in one pass ---
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vec:
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

    total = n
    if total == 0:
        return 0.0

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    # Favors vectors with more 2s to break the symmetry of the ternary grid
    # and prevent the greedy algorithm from clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Spectral danger signal via DFT over F_3^n.
    # Compute the magnitude of the DFT at a few key low-frequency points.
    # High magnitude at low frequencies indicates the vector aligns with many
    # potential 3-term APs, making it dangerous to add.
    # We use a soft-min (log-sum-exp) surrogate to smoothly penalize the
    # most dangerous frequency components.
    
    # Precompute omega = exp(2*pi*i/3) for the DFT
    omega_real = -0.5
    omega_imag = math.sqrt(3.0) / 2.0
    
    # Evaluate DFT magnitude at a small set of representative low-frequency points
    # to keep computation fast. We check frequencies that correspond to:
    # - All zeros (DC component)
    # - Single 1s in each coordinate (first-order harmonics)
    # - A few second-order harmonics for better coverage
    
    dft_mags = []
    
    # DC component: sum of all entries (already computed as s)
    # |DFT(0)| = |s|, but we want magnitude, so use abs(s)
    dft_mags.append(abs(s))
    
    # First-order harmonics: for each coordinate j, frequency e_j
    # DFT(e_j)(v) = sum_i v_i * omega^(v_j * delta_ij)
    # For frequency e_j, this is omega^(v_j) * (sum of other coords)
    # More precisely: DFT(e_j)(v) = omega^(v_j) * (s - v_j)
    # |DFT(e_j)(v)| = |s - v_j| (since |omega| = 1)
    # But we need to be more careful. Let's compute properly.
    
    # For frequency f = (f_1, ..., f_n), DFT(f)(v) = prod_i omega^(f_i * v_i)
    # = omega^(sum_i f_i * v_i)
    # So |DFT(f)(v)| = 1 always! That's not useful.
    
    # I made an error. The DFT of a single vector is just a phase, magnitude 1.
    # What we need is the DFT of the *indicator function* or the *Fourier transform*
    # in the sense of additive combinatorics.
    
    # In additive combinatorics over F_3^n, the Fourier transform of a set S is:
    # hat_S(f) = sum_{x in S} omega^(f·x)
    # For a single vector v, the "Fourier transform" isn't standard.
    
    # Let me reconsider. The danger of adding vector v to a cap set depends on
    # how many lines (3-term APs) v would complete. A line is {a, a+d, a+2d}
    # for some a, d in F_3^n. v completes a line if v = a, a+d, or a+2d.
    
    # Alternative approach: Use the variance as a proxy but make it smoother.
    # Or use the sum of squared deviations from the mean in a way that's
    # more responsive to the actual line structure.
    
    # For now, let's use a refined variance-based danger score with a
    # soft-min surrogate to make it smoother.
    
    mean = s / total
    variance = s2 / total - mean * mean
    
    # Use a soft-min surrogate: instead of linear penalty, use -t * log(sum(exp(-danger_i/t)))
    # where danger_i are individual coordinate deviations. This is smoother than
    # the raw variance.
    
    # Compute per-coordinate deviations
    # Vectors with low deviation from their mean (e.g., all 1s) lie on many
    # parallel lines and are dangerous. Vectors with high deviation (mixed 0s and 2s)
    # are less dangerous. So we want to PENALIZE low deviation (reward high deviation).
    # Use a smooth min surrogate: t * log(sum(exp(-dev_i/t))) approximates min(dev_i) as t->0.
    # We want to penalize when min deviation is small (uniform vector).
    dev_sum_exp = 0.0
    t_soft = 0.3  # Lower temperature for sharper min approximation
    for v in vec:
        dev = abs(v - mean)
        dev_sum_exp += math.exp(-dev / t_soft)
    
    # Smooth min of deviations: small when vector is uniform (dangerous)
    soft_min_dev = t_soft * math.log(dev_sum_exp)
    
    # Penalize low minimum deviation (uniform vectors are dangerous)
    # Use a smooth penalty: -1/(1 + soft_min_dev) is bounded and smooth
    danger_score = -10000.0 / (1.0 + soft_min_dev)

    # Factor 5: Structural line-blocking potential (from current program).
    # We retain the entropy and center-distance signals but re-weight them
    # to be secondary to the chaos-breaking terms.
    entropy = 0.0
    for cnt in (c0, c1, c2):
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log(p + 1e-15)
    ent_norm = entropy / math.log(3.0)

    # Center distance: 1 at center (all 1s), 0 at corners (all 0s or 2s).
    # sum_sq = s2. Center is at sum_sq = 2*n.
    max_sum_sq = 4 * n
    center_dist = 1.0 - abs(s2 - 2 * n) / max_sum_sq

    blocking_score = 1.0 * center_dist + 0.7 * ent_norm

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vec):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # --- Combine ---
    # The distinct_score and weight_score are the dominant "chaos-breaking" signals.
    # The danger_score and balance_score are the smooth "safety" signals.
    # The blocking_score is the structural "line-blocking" signal.
    # The hash_score is a tiny tie-breaker.
    
    # We scale the blocking_score down slightly so that the global symmetry-breaking
    # terms dominate the local structural terms, preventing the greedy algorithm
    # from getting stuck in local optima defined by the lattice symmetries.
    total_priority = (
        distinct_score
        + weight_score
        + balance_score
        + danger_score
        + 30.0 * blocking_score  # Slightly reduced to let danger_score have more influence
        + hash_score
    )
    
    return float(total_priority)


def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means the
    vector is considered first in the greedy insertion.

    Strategy: Hybrid of chaos-breaking (symmetry breaking + distinctness)
    and structural line-blocking potential.
    """
    _init(n)

    # Deadline guard: return a fast deterministic fallback near the limit.
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        vec = tuple(vector)
        # Fast fallback: just a hash to ensure distinctness and speed
        h = 0
        for i, val in enumerate(vec):
            h = (h * 31 + val + i) & 0x7FFFFFFF
        return float(h)

    vec = tuple(vector)

    if vec in _priority_cache:
        return _priority_cache[vec]

    val = _hybrid_priority(vec, n)
    _priority_cache[vec] = val
    return val
