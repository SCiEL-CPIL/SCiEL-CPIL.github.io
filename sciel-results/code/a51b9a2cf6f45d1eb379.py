import time

_START_TIME = time.time()
_TIME_LIMIT = 270.0  # Safety margin before 300s hard limit


def priority(vector, n):
    # Deadline guard: if we've overshot, return a neutral low score
    # so the evaluator can still finish its sort/insertion pass.
    if time.time() - _START_TIME > _TIME_LIMIT:
        return 0.0

    # Count symbols 0, 1, 2
    n0 = 0
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        else:
            n2 += 1

    # ---- Spectral / low-frequency analysis ----
    # A 3-term AP a, b, c satisfies a + c == 2b (mod 3), i.e. b = (a+c)/2.
    # Vectors with many 1s sit near the "midpoint" direction and are more
    # likely to act as the center of a 3-AP.  Vectors close to the all-ones
    # or all-twos directions (low Fourier frequency) also cluster into
    # structured subspaces that harbour many 3-APs.
    #
    # We score a vector by how "spread out" it is in the Fourier domain:
    # high entropy / high frequency  =>  less structured  =>  safer.

    # 1) Midpoint risk: n1 is the count of coordinates equal to 1.
    #    A vector with many 1s is a natural midpoint of pairs (0,2) in those
    #    coordinates, creating many 3-APs.
    midpoint_risk = (n1 * n1) / max(n, 1)

    # 2) Low-frequency alignment: correlation with the all-ones direction.
    #    Map {0,1,2} -> {+1, -1/2, -1/2} so that 1 is "far" from the
    #    all-ones vector in a balanced way; then the sum measures alignment.
    #    Actually, use the standard Z3 Fourier basis: character chi_0(v) = 1.
    #    The "energy" in the trivial character is just |S| for a set S,
    #    but for a single vector we look at its projection onto each of the
    #    three constant characters.  A simpler proxy: how close the symbol
    #    counts are to (n/3, n/3, n/3).  Balanced = high frequency = safe.
    target = n / 3.0
    balance_var = (n0 - target) ** 2 + (n1 - target) ** 2 + (n2 - target) ** 2

    # 3) 0/2 asymmetry: in a 3-AP over Z3, the pair (0,2) is symmetric around 1.
    #    If n0 and n2 are very different, the vector is "polarised" toward one
    #    endpoint, which also correlates with structured APs.
    zero_two_imbalance = (n0 - n2) ** 2 / max(n, 1)

    # 4) Coordinate-level spread: count how many distinct "types" of
    #    coordinate patterns exist.  A vector that is mostly one symbol
    #    (e.g. all 0s) is extremely structured.  We penalise concentration.
    #    Use the largest symbol count as a concentration measure.
    max_count = max(n0, n1, n2)
    concentration = max_count / max(n, 1)  # in [1/3, 1]

    # ---- Combine into a single score (higher = picked earlier) ----
    # We want to AVOID: high midpoint_risk, high balance_var, high
    # zero_two_imbalance, high concentration.
    # So the priority is the negative of a weighted risk.

    risk = (
        3.0 * midpoint_risk
        + 1.5 * (balance_var / max(n, 1))
        + 0.8 * (zero_two_imbalance / max(n, 1))
        + 2.0 * concentration
    )

    # ---- Deterministic tie-breaker via FNV-1a hash ----
    h = 0x811c9dc5
    for v in vector:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFF
    hash_score = (h % 1000003) / 1000003.0  # small perturbation

    # Return: higher priority for lower-risk vectors.
    # Scale the risk term so the hash only breaks ties.
    return -risk * 1000.0 + hash_score
