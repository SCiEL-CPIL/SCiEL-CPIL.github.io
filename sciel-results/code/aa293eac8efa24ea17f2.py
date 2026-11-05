"""Rewritten solver: greedy seed + contact-aware local search (MC + SA) with deadline guard."""
import time
import random
import heapq

# EVOLVE-BLOCK-START

# Global wall-clock budget for the whole evaluation (safety margin under 300s).
_TOTAL_DEADLINE = 245.0
_start_wall = time.time()


def _remaining():
    return _TOTAL_DEADLINE - (time.time() - _start_wall)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _hmap(sequence):
    return [1 if c == "H" else 0 for c in sequence]


def _initial_energy(sequence, chain):
    """Energy (negative) of a conformation, counting H-H non-chain contacts."""
    where = {p: i for i, p in enumerate(chain)}
    h = _hmap(sequence)
    contacts = 0
    for i, p in enumerate(chain):
        if h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and h[j]:
                    contacts += 1
    return -contacts


def _greedy_seed(sequence, dimension, rng):
    """Contact-aware greedy growth; returns (chain, occupied, energy) or None."""
    n = len(sequence)
    h = _hmap(sequence)
    # Precompute, for each position i, the set of positions j (j>i+1) that are H
    # and could become a contact if i were placed next to j's site.
    # We approximate future benefit by counting H residues in a window ahead.
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        options = []
        cur = chain[-1]
        for site in _neighbours(cur):
            if site in occupied:
                continue
            gain = 0
            if h[i]:
                # new H-H contacts with already placed H residues (not adjacent in chain)
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and h[j]:
                        gain += 1
            # future potential: how many H residues are coming up soon (heuristic)
            future = 0
            if h[i]:
                for k in range(i + 1, min(i + 14, n)):
                    if h[k]:
                        future += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain + 0.03 * future + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        # pick best, with a bit of randomness among near-ties
        options.sort(reverse=True)
        top = options[0][0]
        pool = [s for s in options if s[0] >= top - 0.6]
        site = rng.choice(pool)[1]
        occupied[site] = i
        chain.append(site)
    return chain, occupied, _initial_energy(sequence, chain)


def _local_search(sequence, dimension, rng, chain, occupied, energy, time_budget):
    """Monte-Carlo + simulated-annealing local search over single-residue moves."""
    n = len(sequence)
    h = _hmap(sequence)
    best_chain = list(chain)
    best_energy = energy
    deadline = time.time() + time_budget

    # Precompute per-residue contact contribution is complex; we recompute energy
    # incrementally using a where-map and neighbour checks for the moved residue
    # and its neighbours.
    where = {p: i for i, p in enumerate(chain)}

    # Temperature schedule
    t_start = 1.5
    t_end = 0.05
    steps = 0
    total_steps = max(2000, int(time_budget * 4000))

    # Indices we can try to move (avoid ends to keep unit steps valid easily)
    movable = list(range(1, n - 1))

    cur = list(chain)
    cur_occ = dict(occupied)
    cur_where = dict(where)
    cur_e = energy
    T = t_start

    while time.time() < deadline:
        steps += 1
        T = t_start * (t_end / t_start) ** (steps / total_steps)
        i = rng.choice(movable)
        old_site = cur[i]
        # candidate sites: neighbours of old_site (excluding where the chain
        # currently occupies: old_site itself, and the two chain-neighbour sites)
        prev_site = cur[i - 1]
        next_site = cur[i + 1]
        cands = []
        for s in _neighbours(old_site):
            if s == prev_site or s == next_site:
                continue
            if s in cur_occ:
                continue
            cands.append(s)
        if not cands:
            continue
        new_site = rng.choice(cands)

        # Compute energy delta from moving residue i from old_site to new_site.
        # Only contacts involving residue i change.
        # Remove old contributions of i, add new contributions of i.
        delta = 0
        if h[i]:
            # old: H-H contacts between i and placed neighbours of old_site
            for q in _neighbours(old_site):
                j = cur_where.get(q)
                if j is not None and j != i - 1 and j != i + 1 and h[j]:
                    delta -= 1  # losing a contact -> energy increases by 1
            # new
            for q in _neighbours(new_site):
                j = cur_where.get(q)
                # exclude the two chain neighbours and the old_site occupant (i itself)
                if j is not None and j != i - 1 and j != i + 1 and j != i and h[j]:
                    delta += -1  # gaining a contact -> energy decreases by 1
        # Apply move
        cur_occ.pop(old_site)
        cur_occ[new_site] = i
        cur_where.pop(old_site)
        cur_where[new_site] = i
        cur[i] = new_site
        new_e = cur_e + delta

        if new_e < cur_e or rng.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            cur_e = new_e
            if new_e < best_energy:
                best_energy = new_e
                best_chain = list(cur)
        else:
            # revert
            cur_occ.pop(new_site)
            cur_occ[old_site] = i
            cur_where.pop(new_site)
            cur_where[old_site] = i
            cur[i] = old_site

    return best_chain, best_energy


def fold(sequence, dimension):
    n = len(sequence)
    h = _hmap(sequence)

    # Allocate time: longer/harder chains get more, but keep a floor.
    # Rough: base 4s + 0.06s per residue, capped so total fits budget.
    per = 4.0 + 0.06 * n
    per = min(per, 60.0)

    rem = _remaining()
    if rem <= 1.0:
        # No time left: return a straight line (always valid).
        return [[i] + [0] * (dimension - 1) for i in range(n)]

    # Number of greedy seeds to try within the time budget.
    # Estimate per-seed cost ~ (greedy + one short LS). We'll do a few seeds.
    best_chain = None
    best_energy = 0  # energy is <= 0; start at 0 (unfolded)
    rng = random.Random(12345 + len(sequence))

    # Phase 1: generate several greedy seeds, keep the best.
    seed_budget = min(rem * 0.35, 12.0)
    seed_deadline = time.time() + seed_budget
    seeds = []
    while time.time() < seed_deadline and len(seeds) < 40:
        res = _greedy_seed(sequence, dimension, rng)
        if res is None:
            continue
        chain, occ, e = res
        seeds.append((e, chain, occ))
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
    if not seeds:
        # fallback straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]

    # Phase 2: local search on the best seeds (top few), split remaining time.
    seeds.sort(key=lambda x: x[0])
    top = seeds[:4]
    ls_rem = _remaining() - 2.0
    if ls_rem > 1.0:
        # Give the best seed the most time.
        weights = [0.5, 0.2, 0.15, 0.15][:len(top)]
        for (e, chain, occ), w in zip(top, weights):
            tb = ls_rem * w
            if tb < 0.5:
                break
            if _remaining() < 2.0:
                break
            bl, be = _local_search(sequence, dimension, rng, chain, occ, e, tb)
            if be < best_energy:
                best_energy = be
                best_chain = bl
        # Optional: a final LS on the global best if time remains.
        if _remaining() > 3.0 and best_chain is not None:
            occ = {p: i for i, p in enumerate(best_chain)}
            bl, be = _local_search(sequence, dimension, rng, best_chain, occ,
                                   best_energy, _remaining() - 3.0)
            if be < best_energy:
                best_energy = be
                best_chain = bl

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
