"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    """Yield the 4 neighbors of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == "H":
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, idx, old_site, new_site, where):
    """
    Compute energy delta for moving residue idx from old_site to new_site.
    Returns delta_e (new_energy - old_energy).
    """
    delta = 0
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                delta += 1
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                delta -= 1
    return delta


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    occ = occupied.get(q)
                    if occ is not None and occ < i - 1 and sequence[occ] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * bias + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        if rng.random() < 0.7:
            site = options[0][1]
        elif len(options) > 1:
            site = options[rng.randint(0, min(2, len(options) - 1))][1]
        else:
            site = options[0][1]
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
        # Must be adjacent to prev
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        # Must be adjacent to next
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        # Must not be occupied by another residue
        if site in where:
            if where[site] != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        del where[old_site]
        where[new_site] = idx
        chain[idx] = new_site
        return chain, where, current_energy + delta_e, True
    else:
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    # Bias towards shorter segments
    max_len = min(n - 1, rng.choice([2, 3, 4, 5, 6, 8, 10, 15]))
    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, min(i + max_len, n - 1))

    # Check left boundary
    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    # Check right boundary
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    old_segment = list(chain[i:j + 1])
    new_segment = old_segment[::-1]

    # Check self-avoidance
    new_where = dict(where)
    for pt in old_segment:
        del new_where[pt]

    valid = True
    for pt in new_segment:
        if pt in new_where:
            valid = False
            break

    if not valid:
        return chain, where, current_energy, False

    # Apply
    chain[i:j + 1] = new_segment
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = old_segment
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

    max_len = min(n - 1, rng.choice([2, 3, 4, 5, 7]))
    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, min(i + max_len, n - 1))

    pivot = chain[i]
    seg = list(chain[i:j + 1])

    # Rotate 90 degrees clockwise around pivot: (dx,dy) -> (dy, -dx)
    direction = rng.choice([1, -1])  # clockwise or counterclockwise
    rotated_seg = [pivot]
    for pt in seg[1:]:
        dx = pt[0] - pivot[0]
        dy = pt[1] - pivot[1]
        if direction == 1:
            new_pt = (pivot[0] + dy, pivot[1] - dx)
        else:
            new_pt = (pivot[0] - dy, pivot[1] + dx)
        rotated_seg.append(new_pt)

    # Check self-avoidance
    new_where = dict(where)
    for pt in seg[1:]:
        if pt in new_where:
            del new_where[pt]

    valid = True
    for pt in rotated_seg[1:]:
        if pt in new_where:
            valid = False
            break

    if not valid:
        return chain, where, current_energy, False

    # Check right boundary connectivity
    if j < n - 1:
        if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Apply
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

    T = 2.5
    T_min = 0.003
    cooling = 0.9992
    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 4000

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.40:
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

        if no_improve_count > max_no_improve:
            T *= 0.93
            no_improve_count = 0

        T *= cooling

    return best_chain, best_energy


def _polish(sequence, dimension, chain, max_time=5.0, seed=99999):
    """
    High-precision polish pass: only accept moves that improve energy,
    with very tight temperature to escape shallow local minima.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 5000

    start = time.time()
    while (time.time() - start) < max_time and no_improve_count < max_no_improve:
        r = rng.random()
        if r < 0.55:
            # Local move with strict acceptance
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
                    if where[site] != idx:
                        continue
                candidates.append(site)

            if candidates:
                new_site = rng.choice(candidates)
                if new_site != old_site:
                    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)
                    if delta_e <= 0 or (delta_e == 1 and rng.random() < 0.03):
                        del where[old_site]
                        where[new_site] = idx
                        chain[idx] = new_site
                        current_energy += delta_e
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        elif r < 0.82:
            # Short reversal for fine-tuning
            if n >= 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, min(i + 5, n - 1))

                if i > 0:
                    if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
                        no_improve_count += 1
                        continue
                if j < n - 1:
                    if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
                        no_improve_count += 1
                        continue

                old_segment = list(chain[i:j + 1])
                new_segment = old_segment[::-1]

                new_where = dict(where)
                for pt in old_segment:
                    del new_where[pt]

                valid = True
                for pt in new_segment:
                    if pt in new_where:
                        valid = False
                        break

                if valid:
                    chain[i:j + 1] = new_segment
                    for k in range(i, j + 1):
                        where[chain[k]] = k

                    new_energy = _energy(sequence, chain)
                    delta_e = new_energy - current_energy

                    if delta_e <= 0:
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        chain[i:j + 1] = old_segment
                        for k in range(i, j + 1):
                            where[chain[k]] = k
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Try a 90-degree rotation of a short segment
            if n >= 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, min(i + 4, n - 1))

                pivot = chain[i]
                seg = list(chain[i:j + 1])
                direction = rng.choice([1, -1])

                rotated_seg = [pivot]
                for pt in seg[1:]:
                    dx = pt[0] - pivot[0]
                    dy = pt[1] - pivot[1]
                    if direction == 1:
                        new_pt = (pivot[0] + dy, pivot[1] - dx)
                    else:
                        new_pt = (pivot[0] - dy, pivot[1] + dx)
                    rotated_seg.append(new_pt)

                new_where = dict(where)
                for pt in seg[1:]:
                    if pt in new_where:
                        del new_where[pt]

                valid = True
                for pt in rotated_seg[1:]:
                    if pt in new_where:
                        valid = False
                        break

                if valid and j < n - 1:
                    if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
                        no_improve_count += 1
                        continue

                    chain[i:j + 1] = rotated_seg
                    for k in range(i, j + 1):
                        where[chain[k]] = k

                    new_energy = _energy(sequence, chain)
                    delta_e = new_energy - current_energy

                    if delta_e <= 0:
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        chain[i:j + 1] = seg
                        for k in range(i, j + 1):
                            where[chain[k]] = k
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

    return best_chain, best_energy


def _anneal_multi_start(sequence, dimension, initial_chains, total_time, seed_base=12345):
    """
    Run annealing on multiple initial chains, allocating time adaptively.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    # Sort by initial energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    best_chain = None
    best_energy = 0
    
    num_starts = len(initial_chains)
    if num_starts == 0:
        return None, 0
    
    # Allocate time: give more to better starts
    weights = []
    for i in range(num_starts):
        if i < 3:
            weights.append(2.0)
        elif i < 8:
            weights.append(1.3)
        else:
            weights.append(0.8)
    total_weight = sum(weights)
    
    for idx in range(num_starts):
        if _remaining_time() < 15:
            break
        t = total_time * weights[idx] / total_weight
        t = min(t, _remaining_time() - 10)
        if t < 0.5:
            break
        
        seed = seed_base + idx * 7919
        annealed_chain, annealed_energy = _anneal(sequence, dimension, initial_chains[idx], max_time=t, seed=seed)
        
        if best_chain is None or annealed_energy < best_energy:
            best_chain = annealed_chain
            best_energy = annealed_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies and annealing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 400
    biases = [1.0, 1.5, 2.0, 0.5, 0.8, 1.2, 1.8, 0.6, 1.4, 0.9, 2.5, 0.3, 1.1, 1.7, 0.7]
    
    for trial in range(growth_budget):
        if _remaining_time() < 30:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]
        return [list(p) for p in best]
    
    # Deduplicate by energy and keep top unique ones
    seen_energies = set()
    unique_chains = []
    for c in initial_chains:
        e = _energy(sequence, c)
        # Use a hash of the chain for dedup
        ch = tuple(c)
        if ch not in seen_energies:
            seen_energies.add(ch)
            unique_chains.append(c)
    
    # Keep at most 25 best unique chains
    unique_chains.sort(key=lambda c: _energy(sequence, c))
    initial_chains = unique_chains[:25]
    
    # Phase 2: Anneal each initial conformation
    anneal_time = min(12.0, _remaining_time() - 20)
    best_chain, best_energy = _anneal_multi_start(sequence, dimension, initial_chains, anneal_time)
    
    if best_chain is None:
        best_chain = initial_chains[0]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Final polish
    if _remaining_time() > 8 and best_chain is not None:
        polish_time = min(10.0, _remaining_time() - 5)
        best_chain, best_energy = _polish(sequence, dimension, best_chain, max_time=polish_time, seed=99999)
    
    # Phase 4: One more short anneal + polish cycle if time permits
    if _remaining_time() > 15:
        extra_time = min(8.0, _remaining_time() - 8)
        best_chain, best_energy = _anneal(sequence, dimension, best_chain, max_time=extra_time, seed=777777)
        if _remaining_time() > 5:
            polish_time2 = min(6.0, _remaining_time() - 3)
            best_chain, best_energy = _polish(sequence, dimension, best_chain, max_time=polish_time2, seed=555555)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
