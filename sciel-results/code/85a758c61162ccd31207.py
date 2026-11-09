"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size.

Fourier-spectral approach:
- A cap set in {0,1,2}^n is a set with no 3-term arithmetic progression.
- In {0,1,2}, a 3-term AP {a,b,c} exists iff a+c = 2*b (mod 3) with a,b,c not all equal.
- The greedy algorithm adds vectors in descending priority order, keeping only those
  that don't create a 3-AP with any two already-added vectors.

Spectral insight:
- 3-term APs in F_3^n correspond to low-frequency Fourier characters.
- Vectors with high correlation to the all-ones character (i.e., near the center
  point (1,1,...,1)) are more likely to serve as midpoints of 3-APs.
- We penalize vectors based on their "spectral mass" at low frequencies,
  which in coordinate space manifests as:
  1. Many coordinates equal to 1 (the midpoint value) -> high AP risk
  2. Imbalance between 0 and 2 counts -> asymmetric correlation with characters
  3. Low overall variance -> concentrated in a small region of the cube

We combine:
- Midpoint risk penalty (primary): fraction of coordinates equal to 1
- 0/2 imbalance penalty (secondary): |c0 - c2| / n
- Squared edge distance (tertiary): reward for being far from center
- Variance (quaternary): structural spread
- Structured hash tiebreaker (quinary): stable ordering
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


def _better_hash(vector, n):
    """Structured hash that mixes coordinates with position-dependent weights."""
    h = _SEED
    for i, x in enumerate(vector):
        # Mix coordinate value with position to create structured variation
        pos_weight = (i + 1) * 0x9E3779B97F4A7C15
        h = _xorshift(h ^ (x * 0x85EBCA6B) ^ pos_weight)
    return (h & 0xFFFFFFFF) / 4294967296.0


def priority(vector, n):
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0

    if n == 0:
        return 0.0

    v = vector
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    sq_edge = 0
    for x in v:
        if x == 0:
            c0 += 1
            s += 0
            s2 += 0
            sq_edge += 1
        elif x == 1:
            c1 += 1
            s += 1
            s2 += 1
            sq_edge += 0
        else:
            c2 += 1
            s += 2
            s2 += 4
            sq_edge += 1

    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0:
        var = 0.0

    # Normalized squared edge score: range [0, n], normalize to [0,1]
    sq_edge_norm = sq_edge / n if n > 0 else 0.0

    # Midpoint risk: fraction of coordinates equal to 1
    # These are the coordinates where this vector could serve as a midpoint
    midpoint_risk = c1 / n

    # 0/2 imbalance: asymmetry between the two "edge" symbols
    # In F_3, a 3-AP has form {a, a+d, a+2d}. The midpoint is (a + a+2d)/2 = a+d.
    # Vectors with balanced 0/2 counts are less likely to create APs
    imbalance = abs(c0 - c2) / n

    # Structured deterministic tiebreaker
    rnd = _better_hash(v, n)

    # Combine: penalize midpoint risk and imbalance, reward edge distance and variance
    # The spectral view: low-frequency characters correlate with vectors near center
    # and with imbalanced distributions. We suppress these.
    score = (
        -2.5 * midpoint_risk
        - 1.2 * imbalance
        + 2.0 * sq_edge_norm
        + 0.5 * var
        + 0.02 * rnd
    )
    return float(score)
