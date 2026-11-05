"""HP lattice protein folding: greedy construction + local-search refinement."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = time.time() + 300.0   # safety margin before the 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(p):
    """Return the six face-neighbours of lattice point p (as tuples)."""
    out = []
    for ax in range(3):
        for s in (-1, 1):
            q = list(p)
            q[ax] += s
            out.append(tuple(q))
    return out


def _energy(sequence, chain):
    """Energy = -# of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, rng):
    """Randomized greedy chain growth; returns list of tuples or None on trap."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    hpos = [i for i, c in enumerate(sequence) if c == "H"]
    for i in range(1, n):
        last = chain[-1]
        cand = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0.0
            if sequence[i] == "H":
                # immediate H-H contacts with already-placed non-adjacent H
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 2.0
                # look-ahead: H residues later that will be near this site
                for j in hpos:
                    if j <= i:
                        continue
                    # rough distance heuristic: count free neighbours
                # free-space bonus (helps avoid dead ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free + rng.random() * 0.5
            cand.append((score, site))
        if not cand:
            return None
        cand.sort(reverse=True)
        # pick from top few for diversity
        k = min(3, len(cand))
        site = cand[rng.randrange(k)][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_reverse(chain, sequence, a, b):
    """Try reversing sub-segment [a..b]; return new chain if self-avoiding & valid."""
    # After reversal, internal adjacency is preserved. We must check the
    # two boundary edges and that no new self-contact is created.
    n = len(chain)
    # Build new chain
    new = chain[:a] + list(reversed(chain[a:b + 1])) + chain[b + 1:]
    # Check self-avoidance
    seen = set()
    for p in new:
        if p in seen:
            return None
        seen.add(p)
    # Check the two boundary adjacencies are unit steps
    if a > 0:
        if abs(new[a - 1][0] - new[a][0]) + abs(new[a - 1][1] - new[a][1]) + abs(new[a - 1][2] - new[a][2]) != 1:
            return None
    if b + 1 < n:
        if abs(new[b][0] - new[b + 1][0]) + abs(new[b][1] - new[b + 1][1]) + abs(new[b][2] - new[b + 1][2]) != 1:
            return None
    return new


def _local_search(chain, sequence, max_iters=400):
    """2-opt reversals + single-step moves; return improved chain."""
    n = len(chain)
    cur_energy = _energy(sequence, chain)
    rng = random.Random(999)
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        improved = False
        # Try reversals
        for _attempt in range(8):
            if _time_left() < 5.0:
                break
            span = rng.randint(2, min(12, n - 2))
            a = rng.randint(1, n - span - 1)
            b = a + span - 1
            new = _try_reverse(chain, sequence, a, b)
            if new is None:
                continue
            e = _energy(sequence, new)
            if e < cur_energy:
                chain, cur_energy = new, e
                improved = True
                break
        if improved:
            continue
        # Try single-step perturbations
        for _attempt in range(6):
            if _time_left() < 5.0:
                break
            i = rng.randint(1, n - 2)
            for site in _neighbours(chain[i]):
                if site in chain:
                    continue
                # check adjacency with i-1 and i+1
                if abs(site[0] - chain[i - 1][0]) + abs(site[1] - chain[i - 1][1]) + abs(site[2] - chain[i - 1][2]) != 1:
                    continue
                if abs(site[0] - chain[i + 1][0]) + abs(site[1] - chain[i + 1][1]) + abs(site[2] - chain[i + 1][2]) != 1:
                    continue
                new = list(chain)
                new[i] = site
                # self-avoid check (only i changed)
                ok = True
                for k in range(n):
                    if k != i and new[k] == site:
                        ok = False
                        break
                if not ok:
                    continue
                e = _energy(sequence, new)
                if e < cur_energy:
                    chain, cur_energy = new, e
                    improved = True
                    break
        if not improved:
            break
    return chain


def fold(sequence, dimension):
    """Best conformation over multiple greedy constructions + local search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]

    best = None
    best_energy = 0  # energy is negative; start at 0 (no contacts)

    # Deterministic seed + time-based seed for variety
    base_seed = 12345 + n
    rng = random.Random(base_seed)

    # Number of constructions scales with time budget
    max_constructions = 60
    for it in range(max_constructions):
        if _time_left() < 15.0:
            break
        # vary seed
        rng = random.Random(base_seed + it * 7919)
        chain = _grow(sequence, rng)
        if chain is None:
            continue
        # local search refinement
        chain = _local_search(chain, sequence, max_iters=200)
        e = _energy(sequence, chain)
        if e < best_energy:
            best, best_energy = chain, e

    if best is None:
        # fallback: straight line (always self-avoiding)
        best = [(i, 0, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
