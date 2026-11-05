"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90  # Use 90% of 330s budget for safety margin


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE


def _is_deadline_exceeded():
    return time.time() > _get_deadline()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


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


def _fast_local_search(sequence, chain, dimension, rng, time_budget=1.0):
    """
    Fast local search using single-residue moves with incremental energy updates.
    """
    n = len(chain)
    if n < 5:
        return chain, _energy(sequence, chain)

    chain = [tuple(p) for p in chain]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    best_energy = energy
    best_chain = list(chain)
    
    start_time = time.time()
    max_time = start_time + time_budget

    for _ in range(50000):
        if time.time() > max_time or _is_deadline_exceeded():
            break
        
        i = rng.randint(1, n - 1)
        p_old = chain[i]
        
        for axis in range(dimension):
            for step in (-1, 1):
                p_new = list(p_old)
                p_new[axis] += step
                p_new = tuple(p_new)
                
                if p_new in pos_to_idx:
                    j = pos_to_idx[p_new]
                    if abs(j - i) > 1:
                        continue
                    else:
                        continue
                else:
                    if i > 0:
                        p_prev = chain[i-1]
                        if sum(abs(a-b) for a,b in zip(p_new, p_prev)) != 1:
                            continue
                    if i < n-1:
                        p_next = chain[i+1]
                        if sum(abs(a-b) for a,b in zip(p_new, p_next)) != 1:
                            continue

                old_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(p_old):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i and abs(j-i) > 1 and sequence[j] == 'H':
                            old_contacts += 1
                
                new_contacts = 0
                if sequence[i] == 'H':
                    for q in _neighbours(p_new):
                        j = pos_to_idx.get(q)
                        if j is not None and j != i and abs(j-i) > 1 and sequence[j] == 'H':
                            new_contacts += 1
                
                delta_e = -(new_contacts - old_contacts)
                
                if delta_e < 0:
                    pos_to_idx.pop(p_old)
                    pos_to_idx[p_new] = i
                    chain[i] = p_new
                    energy += delta_e
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = list(chain)
                    break

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation of randomized greedy growths followed by Monte Carlo local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    n = len(sequence)
    # Allocate time per sequence: total budget / 26 sequences, with margin
    per_sequence_time = (330 * _BUDGET_FRACTION) / 26.0
    start_time = time.time()

    # Number of greedy restarts: scale with time available
    # For a 48-residue chain, one greedy growth is fast (~1ms)
    # For a 136-residue chain, one greedy growth is slower
    # Estimate: ~n^2 operations per growth, so time per growth ~ n^2 * c
    # Let's do a fixed number of restarts, then local search
    n_restarts = 20
    if n > 100:
        n_restarts = 10
    elif n > 64:
        n_restarts = 15

    for r in range(n_restarts):
        if _is_deadline_exceeded():
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        # Local search to improve this conformation
        chain_ls, energy_ls = _fast_local_search(sequence, chain, dimension, rng, time_budget=0.5)
        if energy_ls < best_energy:
            best, best_energy = chain_ls, energy_ls

    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
