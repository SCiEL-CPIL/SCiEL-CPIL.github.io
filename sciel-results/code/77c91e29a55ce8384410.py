"""Improved program: construct pairwise-disjoint residue classes with distinct moduli in 2..128.

Strategy:
  1. Use ALL even moduli 4..128 (64 classes) with a carefully searched residue assignment.
  2. Attempt to add feasible odd moduli.
  3. Time-guarded at 80s.
"""

import math
import time
import random

def _feasible_pair(m_i, r_i, m_j, r_j):
    g = math.gcd(m_i, m_j)
    return (r_i - r_j) % g != 0

def construct():
    deadline = time.time() + 80.0
    rng = random.Random(12345)

    # All even moduli from 4 to 128
    even_mods = list(range(4, 129, 2))
    n = len(even_mods)

    # Precompute gcd matrix
    gcd_mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcd_mat[i][j] = gcd_mat[j][i] = math.gcd(even_mods[i], even_mods[j])

    def full_check(residues):
        for i in range(n):
            for j in range(i+1, n):
                if (residues[i] - residues[j]) % gcd_mat[i][j] == 0:
                    return False
        return True

    # Local search for even modulus residues
    solution = None
    for restart in range(100):
        residues = [rng.randint(0, m-1) for m in even_mods]
        for it in range(300000 // 100):
            # Sample random pairs to find violations
            violation_found = False
            for _ in range(300):
                i = rng.randint(0, n-1)
                j = rng.randint(0, n-1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                g = gcd_mat[i][j]
                if (residues[i] - residues[j]) % g == 0:
                    # Fix: change one endpoint
                    target = rng.choice([i, j])
                    other = j if target == i else i
                    m_t = even_mods[target]
                    r_o = residues[other]
                    # Try random new values for target
                    fixed = False
                    for _attempt in range(80):
                        new_r = rng.randint(0, m_t - 1)
                        if (new_r - r_o) % g == 0:
                            continue
                        # Quick check against a few others
                        ok = True
                        for k in rng.sample(range(n), min(8, n-1)):
                            if k == target or k == other:
                                continue
                            gk = math.gcd(even_mods[target], even_mods[k])
                            if (new_r - residues[k]) % gk == 0:
                                ok = False
                                break
                        if ok:
                            residues[target] = new_r
                            fixed = True
                            break
                    if not fixed:
                        residues[target] = rng.randint(0, m_t - 1)
                    violation_found = True
                    break

            if it % 5000 == 0:
                if full_check(residues):
                    solution = residues[:]
                    break

        if solution:
            break

        if time.time() > deadline - 30:
            break

    if not solution:
        # Fallback: tiny valid set
        return [[1, 4], [0, 6]]

    # Build result from even moduli
    result_mods = even_mods[:]
    result_res = solution[:]

    # Try adding odd moduli
    odd_mods_sorted = sorted(range(3, 129, 2))
    for m_odd in odd_mods_sorted:
        if time.time() > deadline - 5:
            break
        found_r = None
        for r in range(m_odd):
            ok = True
            for idx in range(len(result_mods)):
                if (r - result_res[idx]) % math.gcd(m_odd, result_mods[idx]) == 0:
                    ok = False
                    break
            if ok:
                found_r = r
                break
        if found_r is not None:
            result_mods.append(m_odd)
            result_res.append(found_r)

    result = [[result_res[i], result_mods[i]] for i in range(len(result_mods))]
    return result
