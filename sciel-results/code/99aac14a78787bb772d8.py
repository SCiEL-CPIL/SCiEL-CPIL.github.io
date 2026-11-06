"""HP lattice protein folding: randomized greedy growth + monotonic local search polish."""
# EVOLVE-BLOCK-START
import random
import math
import time

_START = time.time()
_BUDGET = 110.0  # shared CPU budget across all sequences, with safety margin


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, dimension, rng, deadline):
    """Monotonic local search: only accept moves that strictly reduce energy."""
    n = len(chain)
    if n < 4:
        return chain
    current = [tuple(p) for p in chain]
    cur_energy = _energy(sequence, current)
    best = list(current)
    best_energy = cur_energy
    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        i = rng.randrange(n)
        nbrs = list(_neighbours(current[i]))
        if not nbrs:
            no_improve += 1
            continue
        new_pos = nbrs[rng.randrange(len(nbrs))]
        old_pos = current[i]
        if new_pos == old_pos:
            no_improve += 1
            continue
        occupied_set = set(current)
        occupied_set.discard(old_pos)
        if new_pos in occupied_set:
            no_improve += 1
            continue

        current[i] = new_pos
        new_energy = _energy(sequence, current)
        if new_energy < cur_energy:
            cur_energy = new_energy
            no_improve = 0
            if cur_energy < best_energy:
                best_energy = cur_energy
                best = list(current)
        else:
            current[i] = old_pos
            no_improve += 1

    return best


def fold(sequence, dimension):
    """Best conformation via randomized greedy growths + monotonic local search polish."""
    n = len(sequence)
    global_deadline = _START + _BUDGET
    remaining = global_deadline - time.time()
    if remaining <= 0:
        # out of budget: return a quick straight line (always valid)
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]

    rng = random.Random(12345 + len(sequence))

    best = None
    best_energy = 1

    # Phase 1: greedy growths (fast, reliable)
    # Scale the number of growths to the remaining budget
    num_growth = min(200, max(20, int(remaining * 20)))
    seeds = []
    for _ in range(num_growth):
        if time.time() > global_deadline:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
        seeds.append((value, chain))

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best)
        return [list(p) for p in best]

    # Phase 2: monotonic local search polish on best seeds
    remaining = global_deadline - time.time()
    if remaining > 3.0:
        seeds.sort(key=lambda x: x[0])
        num_polish = min(3, len(seeds))
        per_polish = (remaining * 0.6) / max(1, num_polish)
        for k in range(num_polish):
            if time.time() > global_deadline - 2.0:
                break
            seed_energy, seed_chain = seeds[k]
            if seed_energy > best_energy:
                continue
            polish_deadline = time.time() + per_polish
            polished = _local_search(sequence, seed_chain, dimension, rng, polish_deadline)
            pe = _energy(sequence, polished)
            if pe < best_energy:
                best, best_energy = polished, pe

    # Phase 3: spare time -> more greedy growths
    remaining = global_deadline - 2.0 - time.time()
    if remaining > 1.0:
        extra = int(min(200, remaining * 20))
        for _ in range(max(0, extra)):
            if time.time() > global_deadline - 2.0:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
