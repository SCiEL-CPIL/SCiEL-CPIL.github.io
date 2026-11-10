"""Maximize pairwise-disjoint residue classes with distinct moduli 2..128."""
# EVOLVE-BLOCK-START
import time
from math import gcd

MAX_MOD = 128
DEADLINE = 85.0  # safety margin before the 90s kill


def _is_pairwise_disjoint(classes):
    """Exact gcd criterion: classes (r,m),(r',m') disjoint iff gcd(m,m') ∤ (r-r')."""
    n = len(classes)
    for i in range(n):
        r, m = classes[i]
        for j in range(i + 1, n):
            r2, m2 = classes[j]
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
    return True


def _is_valid(classes):
    if not classes:
        return False
    seen = set()
    for r, m in classes:
        if not (2 <= m <= MAX_MOD):
            return False
        if not (0 <= r < m):
            return False
        if m in seen:
            return False
        seen.add(m)
    return _is_pairwise_disjoint(classes)


def _greedy_assign_residues(moduli):
    """
    Given a list of moduli (all pairwise gcd > 1), assign residues greedily
    such that for each pair, gcd(m_i, m_j) does not divide (r_i - r_j).
    Returns list of [residue, modulus] pairs or None if stuck.
    """
    n = len(moduli)
    residues = [None] * n
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    for i in range(n):
        m = moduli[i]
        # Try residues 0..m-1, find one that satisfies all constraints with prior
        found = False
        for r in range(m):
            ok = True
            for j in range(i):
                g = gcds[i][j]
                if g > 1 and (r - residues[j]) % g == 0:
                    ok = False
                    break
            if ok:
                residues[i] = r
                found = True
                break
        if not found:
            return None
    return [[residues[i], moduli[i]] for i in range(n)]


def construct():
    t0 = time.time()

    # Strategy: use all even moduli (2,4,...,128) — 64 moduli, all pairwise gcd >= 2.
    # Assign residues greedily.
    even_moduli = list(range(2, MAX_MOD + 1, 2))
    result = _greedy_assign_residues(even_moduli)
    if result is not None and _is_valid(result):
        best = result
    else:
        # Fallback: try odd multiples of 3
        odd3_moduli = [m for m in range(3, MAX_MOD + 1, 2) if m % 3 == 0]
        result = _greedy_assign_residues(odd3_moduli)
        if result is not None and _is_valid(result):
            best = result
        else:
            best = [[0, 2]]

    # Try to improve: use all even moduli but with different ordering
    if time.time() - t0 < DEADLINE - 1.0:
        # Try descending order for even moduli
        even_desc = list(range(MAX_MOD, 0, -2))
        result = _greedy_assign_residues(even_desc)
        if result is not None and _is_valid(result) and len(result) > len(best):
            best = result

    # Try to add more moduli: mix evens with some odds that share factors
    # Actually, any odd number is coprime to 2, so we can't add odds to the even set.
    # But we could try a different base: all multiples of 2 or all multiples of 3.
    # 64 evens is the max for a single-prime-based set.
    
    # Alternative: try to find a larger set not based on a single prime.
    # The maximum size of a subset of [2,128] with all pairs gcd>1 is 64 (all evens).
    # Proof sketch: any set with an odd number n, then n is coprime to 2, 
    # but 2 might not be in the set. However, the max clique in the non-coprime graph
    # on [2,128] is indeed 64 (all evens).
    
    return best
# EVOLVE-BLOCK-END
