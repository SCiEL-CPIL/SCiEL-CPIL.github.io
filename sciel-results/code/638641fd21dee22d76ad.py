"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _energy_delta(sequence, old_chain, new_chain, changed_indices):
    """Compute the change in energy when moving residues at changed_indices."""
    old_where = {p: i for i, p in enumerate(old_chain)}
    new_where = {p: i for i, p in enumerate(new_chain)}
    
    # Residues whose neighbours may have changed
    affected = set()
    for idx in changed_indices:
        p_old = old_chain[idx]
        p_new = new_chain[idx]
        affected.add(idx)
        for q in _neighbours(p_old):
            j = old_where.get(q)
            if j is not None:
                affected.add(j)
        for q in _neighbours(p_new):
            j = new_where.get(q)
            if j is not None:
                affected.add(j)
    
    old_contacts = 0
    for i in affected:
        if sequence[i] == 'H':
            p = old_chain[i]
            for q in _neighbours(p):
                j = old_where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    old_contacts += 1
    
    new_contacts = 0
    for i in affected:
        if sequence[i] == 'H':
            p = new_chain[i]
            for q in _neighbours(p):
                j = new_where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    new_contacts += 1
    
    return -(new_contacts) - (-(old_contacts))


def _grow_greedy(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    h_positions = []
    
    for i in range(1, len(sequence)):
        options = []
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            
            dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain * math.exp(-0.2 * dist) + 0.15 * free + rng.random() * 0.8
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        if sequence[i] == 'H':
            h_positions.append(site)
    return chain


def _grow_constrained(sequence, dimension, rng):
    """Growth with stronger constraint on H-H proximity."""
    n = len(sequence)
    if n < 4:
        return _grow_greedy(sequence, dimension, rng, bias=2.0)
    
    # Identify H positions that should be close
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                # Count nearby H residues
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
                # Also check second-nearest for bonus
                for q in _neighbours(site):
                    if q in occupied:
                        for r in _neighbours(q):
                            j = occupied.get(r)
                            if j is not None and j < i - 2 and sequence[j] == 'H':
                                gain += 0.5
            
            dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Strong bias for H-H contacts
            score = 2.5 * gain * math.exp(-0.15 * dist) + 0.1 * free + rng.random() * 0.6
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Pivot move: rotate a subchain."""
    n = len(chain)
    if n < 4:
        return None, None

    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None, None

    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    seen_rotated = set()

    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set or new_p in seen_rotated:
                valid = False
                break
            seen_rotated.add(new_p)
            rotated.append(new_p)

        if not valid:
            continue

        new_chain = chain[:segment_start] + rotated
        changed = list(range(segment_start, n))
        return new_chain, changed

    return None, None


def _slither_move(chain, rng):
    """Slither move: move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None, None

    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])

    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain, [n - 1]
    return None, None


def _crankshaft_move(chain, rng):
    """Crankshaft move: rotate a middle segment."""
    n = len(chain)
    if n < 6:
        return None, None

    # Pick a segment of length 2-4 in the middle
    seg_len = rng.randint(2, min(4, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    if start < 2 or end > n - 2:
        return None, None

    pivot = chain[start]
    segment = chain[start:end]
    fixed_set = set(chain[:start]) | set(chain[end:])
    seen_rotated = set()

    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set or new_p in seen_rotated:
                valid = False
                break
            seen_rotated.add(new_p)
            rotated.append(new_p)

        if not valid:
            continue

        new_chain = chain[:start] + rotated + chain[end:]
        changed = list(range(start, end))
        return new_chain, changed

    return None, None


def _local_search(sequence, chain, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.9995
    min_temp = 0.1

    while time.time() - start_time < time_limit and _time_left() > 15:
        r = rng.random()
        if r < 0.4:
            new_chain, changed = _pivot_move(current_chain, rng)
        elif r < 0.6:
            new_chain, changed = _slither_move(current_chain, rng)
        else:
            new_chain, changed = _crankshaft_move(current_chain, rng)

        if new_chain is None:
            continue

        # Compute energy delta
        if changed is not None:
            delta = _energy_delta(sequence, current_chain, new_chain, changed)
        else:
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, min_temp)):
            current_chain = new_chain
            current_energy += delta

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature = max(min_temp, temperature * cooling)

    return best_chain


def _polish(sequence, chain, time_limit=3.0, rng=None):
    """High-precision polish pass with low temperature."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 0.5
    cooling = 0.9999
    min_temp = 0.05

    while time.time() - start_time < time_limit and _time_left() > 10:
        r = rng.random()
        if r < 0.5:
            new_chain, changed = _pivot_move(current_chain, rng)
        elif r < 0.7:
            new_chain, changed = _slither_move(current_chain, rng)
        else:
            new_chain, changed = _crankshaft_move(current_chain, rng)

        if new_chain is None:
            continue

        if changed is not None:
            delta = _energy_delta(sequence, current_chain, new_chain, changed)
        else:
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

        # Only accept improving or equal moves, or small improvements
        if delta <= 0 or (delta <= 1 and rng.random() < math.exp(-delta / max(temperature, min_temp))):
            current_chain = new_chain
            current_energy += delta

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature = max(min_temp, temperature * cooling)

    # Final deterministic improvement: try all possible end moves
    n = len(current_chain)
    for _ in range(50):
        if _time_left() < 10:
            break
        prev = current_chain[-2]
        last = current_chain[-1]
        occupied = set(current_chain[:-1])
        improved = False
        for site in _neighbours(prev):
            if site in occupied or site == last:
                continue
            new_chain = current_chain[:-1] + [site]
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                improved = True
                break
        if not improved:
            break

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies."""
    seq_len = len(sequence)

    # Time allocation based on sequence length
    if seq_len <= 30:
        num_grow = 200
        local_search_time = 8.0
        polish_time = 3.0
    elif seq_len <= 50:
        num_grow = 150
        local_search_time = 6.0
        polish_time = 2.5
    elif seq_len <= 70:
        num_grow = 100
        local_search_time = 5.0
        polish_time = 2.0
    else:
        num_grow = 80
        local_search_time = 4.0
        polish_time = 1.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Multi-start: diverse growth strategies
    for i in range(num_grow):
        if _time_left() < 25:
            break

        # Alternate between growth strategies
        if i % 3 == 0:
            bias = 0.5 + rng.random() * 1.5
            chain = _grow_greedy(sequence, dimension, rng, bias=bias)
        elif i % 3 == 1:
            chain = _grow_constrained(sequence, dimension, rng)
        else:
            bias = 1.0 + rng.random() * 2.0
            chain = _grow_greedy(sequence, dimension, rng, bias=bias)

        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Local search on best chain
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Polish pass
        if _time_left() > 20:
            actual_polish_time = min(polish_time, _time_left() - 15)
            if actual_polish_time > 0.5:
                rng3 = random.Random(rng.random() * 1000000)
                polished = _polish(sequence, best_chain, time_limit=actual_polish_time, rng=rng3)
                if polished is not None:
                    pe = _energy(sequence, polished)
                    if pe < best_energy:
                        best_energy = pe
                        best_chain = polished

    # Final fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
