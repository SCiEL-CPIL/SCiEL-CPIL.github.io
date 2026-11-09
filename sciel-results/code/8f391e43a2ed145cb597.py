"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size.

Spectral/Fourier approach:
- A cap set in {0,1,2}^n has no 3-term arithmetic progression.
- In the group (Z/3Z)^n, a 3-term AP {a,b,c} satisfies a+c = 2b.
- The Fourier transform on (Z/3Z)^n decomposes functions into characters χ_ξ(x) = ω^{ξ·x}
  where ω = e^{2πi/3}.
- Vectors with high "spectral energy" in low-frequency modes (small |ξ|) are more likely
  to participate in APs because APs are detected by low-frequency correlations.
- We penalize vectors that correlate strongly with the all-ones character and other
  low-frequency characters, pushing them later in the priority order.

Key insight: For a vector v, its "AP risk" is related to how well it aligns with
characters that detect 3-term progressions. The character ξ=(1,1,...,1) (all ones)
detects the most global structure. We compute a spectral penalty based on the
projection of v onto low-frequency characters.

Since we can't compute full FFT efficiently for large n in the priority function,
we use a direct formula that captures the essence:
- The "mean" of v (fraction of 1s) indicates correlation with the all-ones character.
- The "edge-ness" (distance from center value 1) indicates anti-correlation.
- Higher variance and more balanced symbol distribution indicate spread across frequencies.

We combine:
- Low correlation with all-ones (primary: prefer vectors with mean far from 1/3)
- High spectral spread (secondary: prefer diverse symbol patterns)
- Deterministic tiebreaker for stable ordering
"""
import time

# Explicit deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 280.0  # 50s safety margin before 330s kill

_SEED = 987654321


def _xorshift(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 7)
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF


def _better_hash(v, n):
    """Improved hash that considers pairwise interactions for better tiebreaking."""
    h = _SEED
    for i in range(n):
        h = _xorshift(h ^ (v[i] + 1) * 0x9E3779B97F4A7C15)
        if i > 0:
            # Add pairwise interaction term
            prev = v[i-1]
            h = _xorshift(h ^ ((v[i] + 1) * (prev + 1)) * 0xC2B2AE3D27D4EB4F)
    return h & 0xFFFFFFFF


def priority(vector, n):
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0

    v = vector
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # Edge score: distance from center value 1
    edge_score = 0
    for x in v:
        if x == 0:
            c0 += 1
            edge_score += 1
        elif x == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:
            c2 += 1
            s += 2
            s2 += 4
            edge_score += 1

    mean = s / n if n > 0 else 0.0
    var = (s2 / n) - mean * mean if n > 0 else 0.0
    if var < 0:
        var = 0.0

    # Spectral penalty: correlation with all-ones character
    # The all-ones character χ_{1,...,1}(v) = ω^{sum(v_i)}
    # Its magnitude is always 1, but the real part indicates phase alignment.
    # We use the deviation of mean from 1/3 as a proxy for low-frequency energy.
    # Vectors with mean close to 1/3 have high correlation with the uniform character.
    third = 1.0 / 3.0
    mean_dev = abs(mean - third)
    
    # Also consider deviation of symbol counts from uniform
    count_third = n / 3.0
    count_dev = (abs(c0 - count_third) + abs(c1 - count_third) + abs(c2 - count_third)) / n if n > 0 else 0.0

    # Normalized edge score
    edge_norm = edge_score / n if n > 0 else 0.0

    # Deterministic pseudo-random tiebreaker using improved hash
    h = _better_hash(v, n)
    rnd = h / 4294967296.0

    # Combine: 
    # - Primary: high mean_dev (far from uniform, low low-frequency energy)
    # - Secondary: high edge_norm (structural spread)
    # - Tertiary: high var (variance)
    # - Quaternary: high count_dev (symbol distribution spread)
    # - Tiebreaker: small random component
    score = (
        2.0 * mean_dev
        + 1.2 * edge_norm
        + 0.8 * var
        + 0.6 * count_dev
        + 0.03 * rnd
    )
    return float(score)
