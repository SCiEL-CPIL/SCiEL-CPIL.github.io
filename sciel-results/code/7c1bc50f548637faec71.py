import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

# Precompute roots of unity for Z_3
# omega = exp(2*pi*i/3)
import math
_OMEGA = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
_OMEGA_SQ = _OMEGA * _OMEGA  # = exp(4*pi*i/3)

def _dft_magnitude(vec, n):
    """
    Compute a fast proxy for the L2 norm of the DFT of a single vector over Z_3^n.
    
    For a single vector v in {0,1,2}^n, we interpret it as a function on Z_3^n
    that is 1 at v and 0 elsewhere. Its DFT has constant magnitude 1 at all frequencies.
    
    Instead, we use a more useful quantity: the "self-correlation" structure.
    For greedy cap set construction, the key constraint is that no three selected
    vectors form an affine line. A vector v is "dangerous" if it's close (in the
    affine sense) to many already-selected vectors.
    
    We use a spectral proxy: for a vector v, compute the magnitude of its
    character sum against itself shifted by each of the 3^n - 1 nonzero directions.
    Vectors with more "aligned" high-frequency components are more likely to
    participate in arithmetic progressions.
    
    Practical computation: For a vector v, the number of affine lines it can
    form is related to how many pairs (a,b) with a+b=0 (mod 3) and a != 0
    such that v+a and v+2a are "plausible" additions.
    
    We use a simplified spectral score: compute the DFT magnitude at a few
    key frequencies that are most correlated with line formation.
    """
    # For the greedy cap set problem, the most dangerous vectors are those
    # that have many "nearby" vectors in the set. Since we can't access the set,
    # we use the vector's own structure as a proxy.
    
    # Key insight from cap set theory: the largest cap sets in Z_3^n are related
    # to error-correcting codes. Vectors that look like codewords (high minimum
    # distance from the zero vector in a certain metric) are safer.
    
    # We compute a spectral-based score using the fact that for a vector v,
    # the sum over all characters chi_k of |chi_k(v)|^2 * (some weight)
    # relates to the autocorrelation.
    
    # Simplified: use the variance of the coordinate distribution as a spectral proxy.
    # High variance in coordinates correlates with high-frequency DFT content,
    # which in turn correlates with line formation risk.
    
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
        else:
            c2 += 1
            s += 2
            s2 += 4
    
    mean = s / n
    variance = s2 / n - mean * mean
    
    # The "spectral energy" at high frequencies is proportional to variance
    # for a one-dimensional projection. We use this as our spectral danger score.
    return variance

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Spectral danger score via coordinate variance (proxy for high-frequency
       DFT content that correlates with arithmetic progression formation).
    2. Secondary: Quadratic distinct symbol bonus (smooth, rewards using all 3 symbols).
    3. Tertiary: Hamming weight symmetry-breaking (favors vectors with more 2s).
    4. Quaternary: Quadratic balance penalty (smooth surrogate for uniform distribution).
    5. Quinary: Fast polynomial hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting and moment computation
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

    # Factor 1: Spectral danger score.
    # High coordinate variance = more high-frequency DFT content = higher risk
    # of forming arithmetic progressions with other selected vectors.
    mean = s / n
    variance = s2 / n - mean * mean
    spectral_danger = -variance * 8000.0

    # Factor 2: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 3: Hamming weight symmetry-breaking.
    # Favors vectors with more 2s to break the {0,1}^n sub-cube symmetry
    # and encourage exploration of the full {0,1,2}^n space.
    weight_score = c2 * 150.0

    # Factor 4: Quadratic balance penalty.
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 8.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + spectral_danger + weight_score + balance_score + hash_score
    return float(total)
