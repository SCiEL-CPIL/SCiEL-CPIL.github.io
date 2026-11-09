"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size.

PROBABILISTIC INDEPENDENT-SET APPROXIMATION:
- A cap set in {0,1,2}^n is a set with no 3-term arithmetic progression.
- In {0,1,2}, a 3-term AP {a,b,c} exists iff a+c = 2*b (mod 3) with a,b,c not all equal.
- The greedy algorithm adds vectors in descending priority order, keeping only those
  that don't create a 3-AP with any two already-added vectors.

Key insight: For a given vector v, the "collision density" (probability of forming a
3-AP with two other randomly-chosen vectors) depends on the distribution of symbols
in v. We estimate this local collision probability and assign higher priority to
vectors with lower estimated collision density.

For a vector v, a 3-AP occurs when there exist u, w in the set such that:
  v + w = 2*u (mod 3)  =>  u = (v + w) / 2 (mod 3)

The number of potential midpoints u for a given pair (v, w) is determined by the
coordinate-wise structure. Vectors with more "extreme" coordinates (0s and 2s)
tend to have fewer valid midpoints because the midpoint calculation (v+w)/2 mod 3
yields specific values that may not align well with other vectors.

We model the local 3-AP collision probability as:
- For each coordinate i, the contribution to collision risk depends on v[i].
- If v[i] = 1 (center), it's more likely to be a midpoint (higher risk).
- If v[i] = 0 or 2, it's less likely to be a midpoint (lower risk).
- The overall collision probability is approximated by a product/sum of per-coordinate
  risks, weighted by the entropy of the symbol distribution.

We combine:
- Per-coordinate collision risk (primary: quadratic penalty for center coordinates)
- Symbol distribution entropy (secondary: uniform distribution = higher diversity = lower local clustering)
- Pairwise collision estimate (tertiary: based on Hamming weight of non-center coords)
- Structured deterministic tiebreaker (quaternary: stable ordering)
"""
import time
import math

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
        pos_weight = (i + 1) * 0x9E3779B97F4A7C15
        h = _xorshift(h ^ (x * 0x85EBCA6B) ^ pos_weight)
    return (h & 0xFFFFFFFF) / 4294967296.0


def _collision_risk(vector, n):
    """
    Estimate the local 3-AP collision probability for this vector.
    
    For a vector v, the number of ways it can participate in a 3-AP as a midpoint
    depends on how many coordinates are 'center' (value 1). A coordinate with value 1
    can be the midpoint of (0,2) or (2,0) or (1,1), while a coordinate with value 0
    or 2 has fewer midpoint partners.
    
    We approximate the collision risk as:
    - Base risk from center coordinates: each center coord increases risk
    - Entropy of symbol distribution: more uniform = lower local clustering risk
    - Pairwise structure: vectors with many 0s and 2s in similar positions have
      lower collision risk with each other
    """
    c0 = 0
    c1 = 0
    c2 = 0
    for x in vector:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Per-coordinate collision risk: center coordinates (1) are more likely to be midpoints
    # The risk of being a midpoint in coordinate i is higher when v[i] = 1
    # Approximate per-coordinate risk: 1.0 for center, 0.5 for edge
    center_fraction = c1 / n if n > 0 else 0.0
    edge_fraction = (c0 + c2) / n if n > 0 else 0.0
    
    # Base collision risk: quadratic in center fraction
    base_risk = center_fraction ** 2
    
    # Entropy of symbol distribution: higher entropy = more diverse = lower local clustering
    # H = -sum(p_i * log(p_i)) for i in {0,1,2}
    entropy = 0.0
    for c in [c0, c1, c2]:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p + 1e-15)
    # Normalize entropy: max entropy = log(3) for uniform distribution
    max_entropy = math.log(3)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Pairwise collision estimate: vectors with many edges (0s and 2s) have
    # lower collision risk with other edge-heavy vectors
    # The "edge pair density" is proportional to (c0+c2)^2 / n^2
    edge_pair_density = (edge_fraction ** 2)
    
    # Combined collision risk: higher base_risk, lower entropy, lower edge_pair_density => higher risk
    # We want to MINIMIZE this, so priority should be inversely related
    collision_risk = (
        1.5 * base_risk
        - 0.5 * norm_entropy
        - 0.3 * edge_pair_density
    )
    
    return collision_risk


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
    for x in v:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Squared edge distance from center: sum of (x-1)^2
    sq_edge = 0
    for x in v:
        if x != 1:
            sq_edge += 1
    sq_edge_norm = sq_edge / n if n > 0 else 0.0
    
    # Variance across coordinates
    s = 0
    s2 = 0
    for x in v:
        s += x
        s2 += x * x
    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0:
        var = 0.0
    
    # Balance deviation (L2 norm of deviation from uniform)
    third = n / 3.0
    bal_l2 = math.sqrt((c0 - third) ** 2 + (c1 - third) ** 2 + (c2 - third) ** 2) / n
    
    # Symbol diversity: reward using all three symbols
    diversity = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        diversity = 1.0
    elif c0 > 0 and c1 > 0:
        diversity = 0.5
    elif c0 > 0 and c2 > 0:
        diversity = 0.5
    elif c1 > 0 and c2 > 0:
        diversity = 0.5
    else:
        diversity = 0.25
    
    # Collision risk estimate
    collision_risk = _collision_risk(v, n)
    
    # Structured deterministic tiebreaker
    rnd = _better_hash(v, n)
    
    # Combine: prefer low collision risk, high squared edge, high variance,
    # high diversity, low balance deviation, structured tiebreak
    score = (
        -2.5 * collision_risk
        + 1.5 * sq_edge_norm
        + 0.5 * var
        + 0.3 * diversity
        - 0.6 * bal_l2
        + 0.02 * rnd
    )
    return float(score)
