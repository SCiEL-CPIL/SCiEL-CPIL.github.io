"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # All candidate (m, r) pairs
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
    n = len(candidates)

    # Precompute gcd for all modulus pairs (moduli 20..200)
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for a in range(20, max_m + 1):
        for b in range(a, max_m + 1):
            g = math.gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g

    # Precompute weight 1/m as Fraction for exact scoring
    weight = {}
    for m in range(20, max_m + 1):
        weight[m] = Fraction(1, m)

    def compatible(m1, r1, m2, r2):
        g = gcd_table[m1][m2]
        return (r1 % g) != (r2 % g)

    def score_of(pairs):
        s = Fraction(0, 1)
        for (m, r) in pairs:
            s += weight[m]
        return s

    # Greedy: pick pairs in order of smallest modulus, choosing the residue
    # that is compatible with all already-chosen pairs.
    def greedy(seed=None):
        rng = random.Random(seed)
        chosen = []
        chosen_set = set()
        # process moduli from smallest to largest
        for m in range(20, max_m + 1):
            best_r = None
            # try residues; prefer those that keep options open (fewest conflicts later)
            # simple: pick first compatible residue
            for r in range(m):
                ok = True
                for (m2, r2) in chosen:
                    if not compatible(m, r, m2, r2):
                        ok = False
                        break
                if ok:
                    best_r = r
                    break
            if best_r is not None:
                chosen.append((m, best_r))
                chosen_set.add(m)
        return chosen

    # Randomized greedy with multiple seeds
    best_pairs = None
    best_score = Fraction(0, 1)

    for seed in range(2000):
        if time.time() > deadline - 5.0:
            break
        pairs = greedy(seed)
        s = score_of(pairs)
        if s > best_score:
            best_score = s
            best_pairs = pairs

    # Local improvement: try to add a pair for a modulus that was skipped,
    # or swap residues to allow more pairs.
    if best_pairs is not None and time.time() < deadline - 5.0:
        chosen = list(best_pairs)
        chosen_moduli = {m for (m, r) in chosen}
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # try adding any modulus not yet present
            for m in range(20, max_m + 1):
                if m in chosen_moduli:
                    continue
                for r in range(m):
                    ok = True
                    for (m2, r2) in chosen:
                        if not compatible(m, r, m2, r2):
                            ok = False
                            break
                    if ok:
                        chosen.append((m, r))
                        chosen_moduli.add(m)
                        improved = True
                        break
        best_pairs = chosen
        best_score = score_of(best_pairs)

    if best_pairs is None:
        # fallback to the seed program's construction
        return [[20 * j, j - 1] for j in range(1, 11)]

    return [[m, r] for (m, r) in best_pairs]
# EVOLVE-BLOCK-END
