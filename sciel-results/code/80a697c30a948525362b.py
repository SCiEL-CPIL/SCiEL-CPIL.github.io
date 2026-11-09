"""Rewritten program: greedy maximal disjoint-residue construction."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_disjoint_set(classes):
        """Check if all pairs are disjoint. classes: list of (m, r)."""
        n = len(classes)
        for a in range(n):
            ma, ra = classes[a]
            ia = mod_idx[ma]
            for b in range(a + 1, n):
                mb, rb = classes[b]
                ib = mod_idx[mb]
                g = gcd_table[ia][ib]
                if g > 1 and (ra - rb) % g == 0:
                    return False
        return True

    def can_add(classes, m, r):
        """Check if adding (m, r) keeps the set disjoint."""
        im = mod_idx[m]
        for (mm, rm) in classes:
            im2 = mod_idx[mm]
            g = gcd_table[im][im2]
            if g > 1 and (r - rm) % g == 0:
                return False
        return True

    def score(classes):
        s = Fraction(0)
        for (m, r) in classes:
            s += Fraction(1, m)
        return s

    def greedy_construct():
        classes = []
        used_moduli = set()
        # Try to add the smallest available modulus that admits some residue.
        while time.time() < deadline:
            added = False
            for m in moduli:
                if m in used_moduli:
                    continue
                # Find a valid residue; prefer the one with most slack.
                best_r = None
                best_slack = -1
                for r in range(m):
                    if can_add(classes, m, r):
                        # slack = gcd(m, m) is m; measure how many distinct
                        # residue classes mod m remain compatible with existing.
                        # Simpler heuristic: prefer small r (stable, easy).
                        # Use slack = number of r' in [0,m) that are compatible
                        # with all existing classes (approx future freedom).
                        slack = 0
                        for rp in range(m):
                            ok = True
                            for (mm, rm) in classes:
                                g = gcd_table[mod_idx[m]][mod_idx[mm]]
                                if g > 1 and (rp - rm) % g == 0:
                                    ok = False
                                    break
                            if ok:
                                slack += 1
                        # Choose residue with max slack; tie -> smallest r.
                        if slack > best_slack or (slack == best_slack and best_r is not None and r < best_r):
                            best_slack = slack
                            best_r = r
                if best_r is not None:
                    classes.append([m, best_r])
                    used_moduli.add(m)
                    added = True
                    break
            if not added:
                break
        return classes

    result = greedy_construct()

    # Final validity check (defensive).
    if not is_disjoint_set(result):
        # Fallback to a guaranteed-valid small construction.
        result = [[20 * j, j - 1] for j in range(1, 11)]

    return result
# EVOLVE-BLOCK-END
