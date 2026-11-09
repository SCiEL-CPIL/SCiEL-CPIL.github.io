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
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _compute_contact_map(sequence, chain):
    """
    Compute a contact map: for each position i, list of positions j that are in contact with i.
    Returns a list of lists, where contact_map[i] contains j such that i and j are H-H contacts.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contact_map = [[] for _ in range(n)]
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contact_map[i].append(j)
                    contact_map[j].append(i)
    return contact_map


def _energy_from_contact_map(contact_map):
    """Compute energy from contact map."""
    contacts = sum(len(cm) for cm in contact_map)
    return -contacts // 2


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
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
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move (pivot). Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
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

    del where[old_site]
    where[new_site] = idx
    chain[idx] = new_site

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
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

    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    chain[i:j + 1] = chain[i:j + 1][::-1]
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = chain[i:j + 1][::-1]
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _rotate_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a 90-degree rotation of a segment. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    # The segment is chain[i..j]. We rotate it 90 degrees around chain[i].
    # After rotation, chain[i] stays fixed, and the rest of the segment rotates.
    # We need to check self-avoidance and boundary connectivity.

    pivot = chain[i]
    seg = chain[i:j + 1]

    # Rotate each point in seg (except pivot) 90 degrees clockwise around pivot
    # In 2D: (dx, dy) -> (dy, -dx)
    rotated_seg = [pivot]
    for pt in seg[1:]:
        dx = pt[0] - pivot[0]
        dy = pt[1] - pivot[1]
        new_pt = (pivot[0] + dy, pivot[1] - dx)
        rotated_seg.append(new_pt)

    # Check self-avoidance for the rotated segment
    new_where = dict(where)
    # Remove old segment points (except pivot which stays)
    for pt in seg[1:]:
        if pt in new_where:
            del new_where[pt]

    # Check if rotated points collide
    for pt in rotated_seg[1:]:
        if pt in new_where:
            return chain, where, current_energy, False

    # Check boundary connectivity
    # Left boundary: chain[i-1] must be adjacent to chain[i] (pivot, unchanged) - OK
    # Right boundary: chain[j] (old) becomes rotated_seg[-1], must be adjacent to chain[j+1]
    if j < n - 1:
        if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Perform rotation
    chain[i:j + 1] = rotated_seg
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = seg
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0, seed=12345):
    """Simulated annealing with local moves, reversals, and rotations."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 3.0
    T_min = 0.005
    cooling = 0.999
    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 5000

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.45:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.75:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _rotate_move(sequence, chain, where, current_energy, T, rng)

        if accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Restart if stuck for too long
        if no_improve_count > max_no_improve:
            # Cool down faster
            T *= 0.95
            no_improve_count = 0

        T *= cooling

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies and annealing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 500
    biases = [1.0, 1.5, 2.0, 0.5, 0.8, 1.2, 1.8, 0.6, 1.4, 0.9]
    for trial in range(growth_budget):
        if _remaining_time() < 20:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation
    # Allocate time proportionally, but give more to the first few
    time_per_anneal = max(0.3, _remaining_time() / max(1, len(initial_chains)) * 0.8)
    
    # Sort by initial energy to prioritize better starts
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    for idx, chain in enumerate(initial_chains):
        if _remaining_time() < 10:
            break
        # Give more time to earlier (better) chains
        if idx < 3:
            t = time_per_anneal * 2.0
        elif idx < 10:
            t = time_per_anneal * 1.5
        else:
            t = time_per_anneal
        t = min(t, _remaining_time() - 5)
        if t <= 0:
            break
        annealed_chain, annealed_energy = _anneal(sequence, dimension, chain, max_time=t, seed=12345 + idx)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Phase 3: Final polish - dedicated local search at very low temperature
    if _remaining_time() > 5 and best is not None:
        polish_time = min(5.0, _remaining_time() - 3)
        best, best_energy = _anneal(sequence, dimension, best, max_time=polish_time, seed=99999)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
