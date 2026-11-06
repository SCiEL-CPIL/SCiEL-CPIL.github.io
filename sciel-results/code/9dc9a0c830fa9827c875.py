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
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, old_chain, new_chain, pivot_idx, segment_start):
    """Compute energy delta for a pivot move efficiently.
    Only contacts involving the moved segment can change.
    Returns delta = new_energy - old_energy.
    """
    n = len(old_chain)
    # Build position map for old and new
    old_where = {p: i for i, p in enumerate(old_chain)}
    new_where = {p: i for i, p in enumerate(new_chain)}
    
    # Contacts that involve at least one residue in the moved segment
    # Moved segment: indices [segment_start, n-1]
    # We need to check all H-H pairs where at least one index >= segment_start
    
    # Count contacts in old chain involving moved segment
    old_contacts = 0
    for i in range(segment_start, n):
        if sequence[i] == "H":
            for q in _neighbours(old_chain[i]):
                j = old_where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    old_contacts += 1
    # Also count contacts where the H is before segment_start but neighbor is in moved segment
    for i in range(0, segment_start):
        if sequence[i] == "H":
            for q in _neighbours(old_chain[i]):
                j = old_where.get(q)
                if j is not None and j >= segment_start and sequence[j] == "H":
                    old_contacts += 1
    
    # Count contacts in new chain involving moved segment
    new_contacts = 0
    for i in range(segment_start, n):
        if sequence[i] == "H":
            for q in _neighbours(new_chain[i]):
                j = new_where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    new_contacts += 1
    for i in range(0, segment_start):
        if sequence[i] == "H":
            for q in _neighbours(new_chain[i]):
                j = new_where.get(q)
                if j is not None and j >= segment_start and sequence[j] == "H":
                    new_contacts += 1
    
    return -(new_contacts) - (-(old_contacts))  # new_energy - old_energy = -new_contacts - (-old_contacts)


def _polish(sequence, chain, max_iterations=5000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish: try all pivot moves, accept best improving one."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        best_move = None
        best_delta = 0

        # Try all pivot moves
        for pivot_idx in range(1, n - 1):
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            if segment_start >= n:
                continue

            segment = current_chain[segment_start:]
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

                new_chain = current_chain[:segment_start] + rotated
                delta = _energy_delta(sequence, current_chain, new_chain, pivot_idx, segment_start)
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Try slither move
        if best_move is None or best_delta == 0:
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                delta = _energy_delta(sequence, current_chain, new_chain, n - 1, n - 1)
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        if best_move is None or best_delta == 0:
            break

        current_chain = best_move
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain. Validity check is O(segment_length)."""
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


def _crankshaft_move(chain, rng):
    """Crankshaft move: rotate only a 2-residue segment (indices pivot+1 and pivot+2)."""
    n = len(chain)
    if n < 5:
        return None

    pivot_idx = rng.randint(1, n - 3)
    pivot = chain[pivot_idx]
    
    # Segment is chain[pivot_idx+1] and chain[pivot_idx+2]
    p1 = chain[pivot_idx + 1]
    p2 = chain[pivot_idx + 2]
    
    fixed_set = set(chain[:pivot_idx + 1])
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in (p1, p2):
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
        
        new_chain = chain[:pivot_idx + 1] + rotated + chain[pivot_idx + 3:]
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


def _local_search(sequence, chain, max_iterations=10000, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 5.0
    cooling = 0.9995  # Slower cooling to explore more
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move: 40% Pivot, 20% Crankshaft, 20% Slither, 20% Pivot (small segment)
        r = rng.random()
        if r < 0.4:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.6:
            new_chain = _crankshaft_move(current_chain, rng)
        elif r < 0.8:
            new_chain = _slither_move(current_chain, rng)
        else:
            # Small segment pivot
            n = len(current_chain)
            if n < 5:
                new_chain = _pivot_move_fast(current_chain, rng)
            else:
                pivot_idx = rng.randint(1, n - 3)
                new_chain = _crankshaft_move(current_chain, rng)
                if new_chain is None:
                    new_chain = _pivot_move_fast(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search + polish."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_time = 8.0
        polish_time = 5.0
    elif seq_len <= 50:
        num_grow = 250
        local_search_time = 6.0
        polish_time = 4.0
    elif seq_len <= 70:
        num_grow = 180
        local_search_time = 5.0
        polish_time = 3.5
    else:
        num_grow = 120
        local_search_time = 4.0
        polish_time = 3.0

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
    if best_chain is not None and _time_left() > 25:
        improved = _local_search(sequence, best_chain, max_iterations=10000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Deterministic polish
        if _time_left() > 20:
            polished = _polish(sequence, best_chain, max_iterations=5000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=5000,
                                       time_limit=local_search_time * 0.6, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 20:
                final_polish = _polish(sequence, best_chain, max_iterations=3000,
                                       time_limit=polish_time * 0.5, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
