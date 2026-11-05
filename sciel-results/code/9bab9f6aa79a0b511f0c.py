"""Stronger algorithm for odd covering systems with moduli dividing 51975."""
import time
import numpy as np

_BASE = 51975
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _build():
    # All divisors of 51975 (odd, so all divisors are odd)
    divs = [d for d in range(1, _BASE + 1) if _BASE % d == 0]
    moduli = [m for m in divs if m > 1]

    # For each modulus, precompute integer index slices for every residue class.
    # class_slices[m] = list of (r, np.array(indices)) where indices = positions x in [0,_BASE)
    # with x % m == r.
    class_slices = {}
    for m in moduli:
        slices = []
        for r in range(m):
            idx = np.arange(r, _BASE, m, dtype=np.int64)
            slices.append((r, idx))
        class_slices[m] = slices

    # For each residue x, the list of (m, r) classes that cover x.
    # x is covered by (m, r) iff x % m == r.
    covers = [[] for _ in range(_BASE)]
    for m in moduli:
        for x in range(_BASE):
            covers[x].append((m, x % m))
    return moduli, class_slices, covers


def construct():
    start = time.time()
    moduli, class_slices, covers = _build()

    # covered[x] = 1 if residue x is already covered by some chosen class.
    covered = np.zeros(_BASE, dtype=bool)
    uncovered_count = np.full(_BASE, 1, dtype=np.int64)  # 1 if uncovered
    total_uncovered = _BASE

    # Track chosen classes: chosen[(m, r)] = True
    chosen = set()
    # chosen_moduli: set of moduli already used (each divisor used at most once)
    used_moduli = set()

    # Precompute, for each modulus m, the list of (r, indices) and the count of
    # currently-uncovered residues in that class (recomputed each iteration).
    # We'll iterate over all candidate (m, r) not yet chosen.

    # To make greedy efficient, maintain for each (m, r) a count of uncovered residues.
    # Initialize counts.
    # cand: dict (m, r) -> count of uncovered residues in that class
    cand = {}
    for m in moduli:
        for r, idx in class_slices[m]:
            cand[(m, r)] = int(np.count_nonzero(uncovered_count[idx]))

    best_classes = []

    while total_uncovered > 0:
        if time.time() > _DEADLINE:
            break

        # Find the candidate (m, r) with max uncovered count; tie-break by smaller m,
        # then smaller r (deterministic).
        best_key = None
        best_count = -1
        best_m = None
        best_r = None
        for (m, r), cnt in cand.items():
            if cnt <= 0:
                continue
            if (best_key is None or
                    cnt > best_count or
                    (cnt == best_count and m < best_m) or
                    (cnt == best_count and m == best_m and r < best_r)):
                best_key = (m, r)
                best_count = cnt
                best_m = m
                best_r = r

        if best_key is None:
            break

        m, r = best_key
        if m in used_moduli:
            # Should not happen since each (m,r) is unique per m and we only add
            # one r per m? Actually we may add multiple r for the same m? No:
            # each modulus m can be used at most once (distinct moduli). So once
            # we pick a class with modulus m, we must exclude all other classes
            # with modulus m.
            # Re-scan: this means best_key should never be a used modulus. Guard.
            continue

        # Commit this class.
        used_moduli.add(m)
        chosen.add((m, r))
        best_classes.append([m, r])

        # Update covered and uncovered_count for this class.
        idx = class_slices[m][r][1]
        newly = ~covered[idx]
        if newly.any():
            gain = int(newly.sum())
            covered[idx] = True
            uncovered_count[idx] = 0
            total_uncovered -= gain
        else:
            gain = 0

        # Remove all other classes with the same modulus from candidates
        # (modulus m is now used).
        for r2, _ in class_slices[m]:
            if (m, r2) != (m, r):
                cand.pop((m, r2), None)

        # Update counts for all remaining candidate classes that intersect this
        # class's indices. For efficiency, recompute counts only for classes whose
        # index set overlaps idx. Overlap happens when the two classes share some
        # residue x, i.e., when there exists x with x%m==r and x%m2==r2.
        # Simpler: recompute counts for all remaining candidates each iteration is
        # O(num_candidates * avg_slice_len). num_candidates ~ sum(m for m in moduli not used).
        # That's large. Instead, update incrementally: for each remaining candidate
        # (m2, r2), the number of newly covered residues in its slice = number of
        # positions in idx2 that are in idx (i.e., x with x%m==r and x%m2==r2).
        # We can compute this as the size of the intersection of the two arithmetic
        # progressions. That equals the number of x in [0,_BASE) satisfying both
        # congruences, which is either 0 or floor((_BASE-1 - x0)/lcm)+1 if solvable.
        # To keep it simple and fast, recompute counts lazily: only recompute for
        # candidates that might have changed. A candidate (m2, r2) changes iff it
        # shares a residue with the new class, i.e., the congruence system
        # x%m==r, x%m2==r2 is solvable. That is solvable iff gcd(m, m2) | (r - r2).
        # When solvable, the shared residues form one residue class mod lcm(m, m2),
        # so the count of shared x in [0,_BASE) is either floor((_BASE-1 - x0)/lcm)+1
        # or 0. We subtract that from the candidate's count.
        for (m2, r2) in list(cand.keys()):
            if m2 == m:
                continue
            g = _gcd(m, m2)
            if (r - r2) % g != 0:
                continue
            # Solvable. Find one solution x0 in [0, lcm) using CRT, then count.
            lcm = m * m2 // g
            x0 = _crt(m, r, m2, r2, g)
            if x0 is None:
                continue
            # Count x in [0, _BASE) with x % lcm == x0.
            if x0 >= _BASE:
                continue
            cnt_shared = (_BASE - 1 - x0) // lcm + 1
            cand[(m2, r2)] -= cnt_shared

        # Also update the count of the chosen class itself (now 0) - already removed?
        # We kept (m, r) in cand; set to 0.
        cand[(m, r)] = 0

        if total_uncovered == 0:
            break

    # Return the list of [m, r] classes.
    return best_classes


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _crt(m1, r1, m2, r2, g):
    # Solve x % m1 == r1, x % m2 == r2. g = gcd(m1, m2). Returns x0 in [0, lcm) or None.
    # We require (r1 - r2) % g == 0 (caller ensures).
    m1p = m1 // g
    m2p = m2 // g
    # Solve m1p * t ≡ (r2 - r1)/g (mod m2p)
    diff = (r2 - r1) // g
    # Inverse of m1p mod m2p (they are coprime)
    inv = _modinv(m1p % m2p, m2p)
    t = (diff * inv) % m2p
    x0 = r1 + m1 * t
    lcm = m1 * m2p
    return x0 % lcm


def _modinv(a, m):
    # Extended Euclid; m > 1, gcd(a, m) == 1.
    if m == 1:
        return 0
    t, newt = 0, 1
    r, newr = m, a % m
    while newr != 0:
        q = r // newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    if r > 1:
        return None
    if t < 0:
        t += m
    return t
