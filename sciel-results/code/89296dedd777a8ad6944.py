"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


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


def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    # Candidate new sites: neighbours of old_site that are adjacent to both prev and next
    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        # Must be adjacent to prev and next
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        # Self-avoidance check: site must not be occupied by any index other than idx
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    # Perform move
    del where[old_site]
    where[new_site] = idx
    chain[idx] = new_site

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        del where[new_site]
        where[old_site] = idx
        chain[idx] = old_site
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    # Check adjacency after reversal
    # Original: ... chain[i-1] - chain[i] ... chain[j] - chain[j+1] ...
    # Reversed: ... chain[i-1] - chain[j] ... chain[i] - chain[j+1] ...
    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Self-avoidance: the set of sites is unchanged, so no new collisions with outside.
    # But we must ensure that the internal adjacencies are valid.
    # The internal adjacencies are the same pairs, just in reverse order, so they remain valid.
    # So self-avoidance is preserved.

    # Perform reversal
    chain[i:j + 1] = chain[i:j + 1][::-1]
    # Update where dict
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        chain[i:j + 1] = chain[i:j + 1][::-1]
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0):
    """Simulated annealing with local moves and reversals."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    rng = random.Random(12345 + n + int(time.time() * 1000))

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        # Alternate between local moves and reversals
        if rng.random() < 0.5:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)

        if accepted and current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)

        T *= cooling

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths followed by annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1

    # Phase 1: Greedy growth to get initial conformations
    initial_chains = []
    for _ in range(50):
        if _remaining_time() < 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation
    time_per_anneal = max(0.5, _remaining_time() / len(initial_chains) * 0.8)
    for chain in initial_chains:
        if _remaining_time() < 5:
            break
        annealed_chain, annealed_energy = _anneal(sequence, dimension, chain, max_time=time_per_anneal)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
