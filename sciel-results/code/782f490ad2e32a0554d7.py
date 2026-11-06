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


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
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
            score = bias * gain * math.exp(-0.3 * dist) + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain."""
    n = len(chain)
    if n < 4:
        return None

    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None

    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])

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
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)

        if not valid:
            continue

        new_chain = chain[:segment_start] + rotated
        return new_chain

    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None

    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])

    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain
    return None


def _contact_map(sequence):
    """Pre-compute a contact map: for each H-H pair, the number of steps between them."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            if j - i > 1:  # Not adjacent along chain
                pairs.append((i, j, j - i))
    return pairs


def _constraint_growth(sequence, dimension, rng, contact_pairs, max_contacts_to_enforce=5):
    """
    Backtracking growth with hard constraints on top contact pairs.
    We try to enforce spatial proximity for the top-k H-H pairs (by closeness in sequence).
    """
    n = len(sequence)
    if n < 4:
        return _grow(sequence, dimension, rng, bias=1.0)

    # Sort contact pairs by sequence distance (closer pairs are easier to satisfy first? or prioritize long range?)
    # Let's prioritize pairs that are close in sequence but far in space potential, or just top by some heuristic.
    # For now, let's just pick the first few H-H pairs that are non-adjacent.
    # A better heuristic: prioritize pairs that are "distant" in sequence but likely to be close in space in compact folds.
    # Let's just use the first few valid pairs.
    
    # We will try to grow the chain such that at the end, specific pairs are in contact.
    # This is hard to do directly in growth. 
    # Alternative: Use the contact map to bias the growth.
    # If we are placing residue i, and there is a future H residue j such that (i,j) is a key pair,
    # we might want to place i near where j will eventually be. But we don't know where j will be.
    
    # Let's stick to a simpler improvement: 
    # 1. Better energy function with precomputed H set for speed.
    # 2. More aggressive local search with multiple restarts.
    # 3. "Crane" moves or other moves?
    
    # Actually, let's implement a "pair-based" local search.
    # Instead of just pivot/slither, we can try to move a specific residue to make a contact with another.
    
    # For now, let's just return a standard growth but with a bias towards compactness.
    return _grow(sequence, dimension, rng, bias=1.5)


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
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
    temperature = 5.0
    cooling = 0.999

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.6:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.8:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move_fast(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, time_limit=3.0, rng=None):
    """High-precision polish pass with low temperature for fine-tuning."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 1.0
    cooling = 0.9995

    while time.time() - start_time < time_limit and _time_left() > 15:
        r = rng.random()
        if r < 0.7:
            new_chain = _pivot_move_fast(current_chain, rng)
        else:
            new_chain = _slither_move(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or (delta <= 1 and rng.random() < math.exp(-delta / max(temperature, 0.05))):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    n = len(current_chain)
    for pivot_idx in range(max(1, n - 8), n - 1):
        if _time_left() < 15:
            break
        pivot = current_chain[pivot_idx]
        segment = current_chain[pivot_idx + 1:]
        fixed_set = set(current_chain[:pivot_idx + 1])

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
                if new_p in fixed_set:
                    valid = False
                    break
                rotated.append(new_p)

            if not valid:
                continue

            new_chain = current_chain[:pivot_idx + 1] + rotated
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    if seq_len <= 30:
        num_grow = 400
        local_search_time = 10.0
    elif seq_len <= 50:
        num_grow = 300
        local_search_time = 8.0
    elif seq_len <= 70:
        num_grow = 200
        local_search_time = 6.0
    else:
        num_grow = 150
        local_search_time = 5.0

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Greedy Growth
    for i in range(num_grow):
        if _time_left() < 20:
            break

        bias = 0.5 + rng.random() * 2.0
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Local Search on the best chain found
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, max_iterations=10000,
                                 time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Polish pass
        if _time_left() > 25:
            polish_time = min(4.0, _time_left() - 20)
            if polish_time > 1.0:
                rng3 = random.Random(rng.random() * 1000000)
                polished = _polish(sequence, best_chain, time_limit=polish_time, rng=rng3)
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
