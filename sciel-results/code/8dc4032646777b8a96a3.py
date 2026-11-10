"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 175.0  # 35s safety margin from 210s


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(seq, chain):
    """Compute HP energy (negative number of H-H contacts)."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(seq, rng):
    """Generate a self-avoiding walk via greedy growth."""
    n = len(seq)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            score = rng.random()
            if seq[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 3.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            options.append((score, site))
        if not options:
            return None
        _, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
    return chain


def _sa_refine(seq, chain, rng, time_budget):
    """Simulated annealing refinement. Returns (best_chain, best_energy)."""
    n = len(seq)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    energy = _compute_energy(seq, chain)
    best_chain = list(chain)
    best_energy = energy

    T_start = 1.5
    T_end = 0.005
    total_steps = max(2000, n * 100)
    start_t = time.time()

    for step in range(total_steps):
        if time.time() - start_t > time_budget:
            break
        if _time_left() < 3.0:
            break

        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress

        # Choose move type: single residue move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        candidates = list(_neighbours(old_pos))
        rng.shuffle(candidates)

        for new_pos in candidates:
            if new_pos in where:
                continue

            # Compute delta energy
            delta_e = 0
            if seq[i] == 'H':
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and seq[j] == 'H':
                        delta_e += 1
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and abs(j - i) > 1 and seq[j] == 'H':
                        delta_e -= 1

            if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
                chain[i] = new_pos
                del where[old_pos]
                where[new_pos] = i
                energy += delta_e
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                break

    return best_chain, best_energy


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]

    if _time_left() < 3.0:
        return [[i, 0] for i in range(n)]

    best_chain = None
    best_energy = 0

    # Allocate time for restarts
    # Use a fraction of remaining time for each restart
    base_time = max(0.3, min(5.0, _time_left() * 0.15))
    rng = random.Random(12345 + n)

    while _time_left() > base_time + 2.0:
        chain = _greedy_grow(sequence, rng)
        if chain is None:
            continue

        refined, energy = _sa_refine(sequence, chain, rng, base_time)

        if energy < best_energy:
            best_energy = energy
            best_chain = refined

        # Early stop if very good
        if best_energy <= -n * 0.4:
            break

    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    else:
        best_chain = [list(p) for p in best_chain]

    return best_chain
# EVOLVE-BLOCK-END
