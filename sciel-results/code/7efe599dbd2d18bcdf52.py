"""Rewritten: multi-start randomized greedy for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import math
import random
import time

_DEADLINE = 140.0  # safety margin under 150s kill


def _score(classes):
    return sum(1.0 / m for m, r in classes)


def _try_greedy(order, rng=None):
    """Greedy: iterate moduli in `order`; add if a residue exists disjoint from all placed."""
    placed = []
    for m in order:
        # forbidden residues: union over placed of {r_i mod g} lifted to [0, m)
        forbidden = set()
        for m_i, r_i in placed:
            g = math.gcd(m, m_i)
            base = r_i % g
            # all r in [0,m) with r ≡ base (mod g)
            forbidden.update(range(base, m, g))
        if len(forbidden) < m:
            # pick a residue not in forbidden (deterministic: smallest; or random)
            all_res = set(range(m)) - forbidden
            if rng is not None:
                r = rng.choice(sorted(all_res))
            else:
                r = min(all_res)
            placed.append([m, r])
    return placed


def _random_order(moduli, rng):
    """Order moduli: smaller ones first with some random perturbation to escape traps."""
    # Weight: prefer small moduli but allow some randomness
    # Use a "jittered" rank: rank = index + random*spread
    n = len(moduli)
    ranks = [(i + rng.random() * 0.5, m) for i, m in enumerate(moduli)]
    ranks.sort(key=lambda x: x[0])
    return [m for _, m in ranks]


def _local_search(classes, moduli_set, rng, time_budget):
    """Try to swap a placed modulus for a smaller unplaced one to improve score."""
    placed_moduli = {m for m, r in classes}
    unplaced = [m for m in moduli_set if m not in placed_moduli]
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        # Try replacing each placed with a smaller unplaced
        for i, (m_i, r_i) in enumerate(classes):
            for m_new in unplaced:
                if m_new >= m_i:
                    break
                # try adding m_new with some residue, then check if m_i can be removed
                # Simpler: rebuild without m_i, then add m_new
                others = [c for j, c in enumerate(classes) if j != i]
                # Can m_new be added to others?
                forbidden = set()
                for m_o, r_o in others:
                    g = math.gcd(m_new, m_o)
                    base = r_o % g
                    forbidden.update(range(base, m_new, g))
                if len(forbidden) < m_new:
                    avail = [r for r in range(m_new) if r not in forbidden]
                    r_new = rng.choice(avail)
                    new_classes = others + [[m_new, r_new]]
                    if _score(new_classes) > _score(classes) + 1e-15:
                        classes = new_classes
                        unplaced = [m for m in moduli_set if m not in {c[0] for c in classes}]
                        improved = True
                        break
            if improved:
                break
    return classes


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    moduli = list(range(20, 201))
    rng = random.Random(12345)

    best = None
    best_score = -1.0

    # Deterministic baseline: smallest-first
    base = _try_greedy(sorted(moduli))
    best = base
    best_score = _score(base)

    # Multi-start randomized greedy
    n_starts = 0
    while time.time() < deadline - 5.0:
        order = _random_order(moduli, rng)
        cand = _try_greedy(order, rng)
        s = _score(cand)
        if s > best_score + 1e-15:
            best = cand
            best_score = s
        n_starts += 1
        # Occasionally do local search on the best
        if n_starts % 10 == 0 and time.time() < deadline - 10.0:
            improved = _local_search([c[:] for c in best], set(moduli), rng, deadline - 10.0)
            is_ = _score(improved)
            if is_ > best_score + 1e-15:
                best = improved
                best_score = is_

    # Final local search on the best if time permits
    if time.time() < deadline - 2.0:
        improved = _local_search([c[:] for c in best], set(moduli), rng, deadline - 2.0)
        is_ = _score(improved)
        if is_ > best_score + 1e-15:
            best = improved
            best_score = is_

    return best
# EVOLVE-BLOCK-END
