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


def _polish(sequence, chain, max_iterations=2000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves."""
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

        # Try all pivot moves and accept the best improving one
        n = len(current_chain)
        best_move = None
        best_delta = 0

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
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        if best_move is None:
            # Try slither move
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        if best_move is None:
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
            # Heuristic: Prefer sites that are "inside" the current bounding box or close to centroid
            # Calculate centroid of chain so far
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Strong bias towards gaining contacts, secondary bias towards compactness (lower dist)
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
    
    # Fixed part is chain[:pivot_idx+1] (including pivot)
    # The points to check are the rotated chain[pivot_idx+1:]
    # They must not collide with chain[:pivot_idx+1]
    # Note: The pivot itself is fixed, so it's not part of the moving segment to check for collision with fixed part,
    # but the first point of the moving segment (index pivot_idx+1) will move.
    
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
        
    segment = chain[segment_start:]
    
    # Set of fixed points (including the pivot point which acts as the anchor)
    # We need to ensure the rotated segment doesn't hit any point in chain[:pivot_idx+1]
    # Actually, the pivot point chain[pivot_idx] is fixed.
    # The points chain[segment_start:] move.
    # They cannot occupy any position in chain[:pivot_idx] (the part before pivot).
    # They also cannot occupy the pivot position chain[pivot_idx] (obviously, as it's already taken and they move away).
    # So fixed_set = set(chain[:pivot_idx+1]) is correct.
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
            else:  # 180
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if not valid:
            continue
            
        # If no collision with fixed set, we must also ensure the rotated segment itself is self-avoiding.
        # Since the original segment was self-avoiding, and rotation is a rigid transform, 
        # the rotated segment is self-avoiding internally.
        # However, we must ensure the new chain is a valid path (unit steps).
        # Rotation preserves relative distances, so unit steps are preserved.
        # The only potential issue is the connection at the pivot.
        # chain[pivot_idx] is fixed. The first point of the new segment is rotated[0].
        # Is rotated[0] a unit step away from pivot?
        # Yes, because p[0] (which is chain[segment_start]) was a unit step away from pivot originally.
        # Rotation preserves distance.
        
        new_chain = chain[:segment_start] + rotated
        return new_chain

    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    # Try moving the last residue
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain
    return None


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
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
    cooling = 0.999  # Slower cooling to explore more
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move: 60% Pivot, 20% Slither, 20% Crankshaft (simplified pivot)
        r = rng.random()
        if r < 0.6:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.8:
            new_chain = _slither_move(current_chain, rng)
        else:
            # Fallback to a simple random pivot if fast one fails or for variety
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
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_time = 8.0
        polish_time = 4.0
    elif seq_len <= 50:
        num_grow = 250
        local_search_time = 6.0
        polish_time = 3.5
    elif seq_len <= 70:
        num_grow = 180
        local_search_time = 5.0
        polish_time = 3.0
    else:
        num_grow = 120
        local_search_time = 4.0
        polish_time = 2.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Greedy Growth
    # We use a range of biases to explore different compactness levels
    for i in range(num_grow):
        if _time_left() < 20:
            break

        # Bias between 0.5 and 2.5
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
            polished = _polish(sequence, best_chain, max_iterations=3000,
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
                final_polish = _polish(sequence, best_chain, max_iterations=2000,
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
