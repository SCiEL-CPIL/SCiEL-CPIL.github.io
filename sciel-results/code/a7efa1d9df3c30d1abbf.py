"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


_NEIGHBOUR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in _NEIGHBOUR_OFFSETS:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _build_occupied(chain):
    return {p: i for i, p in enumerate(chain)}


def _contact_count(sequence, chain, occupied):
    """Count H-H contacts using pre-built occupied dict."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    h_set = set(h_indices)
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in _NEIGHBOUR_OFFSETS:
            j = occupied.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return contacts


def _spiral_fold(sequence):
    """Generate a compact spiral conformation. Returns None if invalid."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    # Spiral pattern: right, up, left, left, down, down, right, right, right, up, up...
    # Actually, let's use a proper square spiral
    dx, dy = 1, 0
    step_len = 1
    steps_in_dir = 1
    steps_taken = 0
    direction_changes = 0
    
    x, y = 0, 0
    for i in range(1, n):
        # Try current direction
        nx, ny = x + dx, y + dy
        if (nx, ny) not in occupied:
            x, y = nx, ny
            occupied[(x, y)] = i
            chain.append((x, y))
            steps_taken += 1
            if steps_taken >= step_len:
                steps_taken = 0
                direction_changes += 1
                # Turn left
                dx, dy = -dy, dx
                if direction_changes % 2 == 0:
                    step_len += 1
        else:
            # Turn left and try again
            dx, dy = -dy, dx
            if direction_changes % 2 == 0:
                step_len += 1
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                x, y = nx, ny
                occupied[(x, y)] = i
                chain.append((x, y))
                steps_taken += 1
            else:
                # Try other directions
                found = False
                for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    tx, ty = x + ndx, y + ndy
                    if (tx, ty) not in occupied:
                        x, y = tx, ty
                        occupied[(x, y)] = i
                        chain.append((x, y))
                        dx, dy = ndx, ndy
                        found = True
                        break
                if not found:
                    return None
    return chain


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                x, y = site
                for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _apply_pivot(chain, pivot_idx, angle):
    """Apply a pivot move and return the new chain if valid, else None."""
    n = len(chain)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
    
    fixed_set = set(chain[:pivot_idx + 1])
    segment = chain[segment_start:]
    
    rotated = []
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
            return None
        rotated.append(new_p)
    
    # Check for self-intersection within the rotated segment
    rotated_set = set()
    for p in rotated:
        if p in rotated_set:
            return None
        rotated_set.add(p)
    
    return chain[:segment_start] + rotated


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain."""
    n = len(chain)
    if n < 4:
        return None
    pivot_idx = rng.randint(1, n - 2)
    angle = rng.choice((90, -90, 180))
    return _apply_pivot(chain, pivot_idx, angle)


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    candidates = [site for site in _neighbours(prev) if site not in occupied]
    if not candidates:
        return None
    site = rng.choice(candidates)
    return chain[:-1] + [site]


def _segment_shift(chain, rng):
    """Shift a segment by one unit in a random direction."""
    n = len(chain)
    if n < 5:
        return None
    seg_start = rng.randint(1, n - 3)
    seg_end = rng.randint(seg_start + 1, n - 1)
    direction = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
    
    # Check boundaries
    if seg_start > 0 and seg_end < n - 1:
        # Both sides have fixed parts
        fixed_before = set(chain[:seg_start])
        fixed_after = set(chain[seg_end + 1:])
    elif seg_start == 0:
        fixed_before = set()
        fixed_after = set(chain[seg_end + 1:])
    else:
        fixed_before = set(chain[:seg_start])
        fixed_after = set()
    
    fixed_all = fixed_before | fixed_after
    
    segment = chain[seg_start:seg_end + 1]
    shifted = []
    for p in segment:
        np_ = (p[0] + direction[0], p[1] + direction[1])
        if np_ in fixed_all:
            return None
        shifted.append(np_)
    
    # Check self-intersection within shifted segment
    shifted_set = set()
    for p in shifted:
        if p in shifted_set:
            return None
        shifted_set.add(p)
    
    return chain[:seg_start] + shifted + chain[seg_end + 1:]


def _local_search(sequence, chain, max_iterations=20000, time_limit=5.0, rng=None):
    """Simulated annealing local search with multiple move types."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_occupied = _build_occupied(current_chain)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    h_set = set(h_indices)
    current_contacts = _contact_count(sequence, current_chain, current_occupied)
    current_energy = -current_contacts
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.9997

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.55:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.80:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _segment_shift(current_chain, rng)

        if new_chain is None:
            continue

        new_occupied = _build_occupied(new_chain)
        new_contacts = _contact_count(sequence, new_chain, new_occupied)
        new_energy = -new_contacts
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_contacts = new_contacts
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, max_iterations=3000, time_limit=3.0, rng=None):
    """Hill-climbing polish: accept only improving moves, try all pivot positions."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    h_set = set(h_indices)
    current_occupied = _build_occupied(current_chain)
    current_contacts = _contact_count(sequence, current_chain, current_occupied)
    current_energy = -current_contacts
    best_chain = list(current_chain)
    best_energy = current_energy

    n = len(current_chain)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        best_move = None
        best_delta = 0
        best_occupied = None

        # Try pivot moves at various positions
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain is None:
                    continue
                new_occupied = _build_occupied(new_chain)
                new_contacts = _contact_count(sequence, new_chain, new_occupied)
                new_energy = -new_contacts
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain
                    best_occupied = new_occupied

        # Try slither
        slither = _slither_move(current_chain, rng)
        if slither is not None:
            sl_occupied = _build_occupied(slither)
            sl_contacts = _contact_count(sequence, slither, sl_occupied)
            sl_energy = -sl_contacts
            delta = sl_energy - current_energy
            if delta < best_delta:
                best_delta = delta
                best_move = slither
                best_occupied = sl_occupied

        if best_move is None:
            break

        current_chain = best_move
        current_occupied = best_occupied
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_time = 9.0
        polish_time = 5.0
    elif seq_len <= 50:
        num_grow = 250
        local_search_time = 7.0
        polish_time = 4.0
    elif seq_len <= 70:
        num_grow = 180
        local_search_time = 6.0
        polish_time = 3.5
    else:
        num_grow = 120
        local_search_time = 5.0
        polish_time = 3.0

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Try spiral fold first
    spiral = _spiral_fold(sequence)
    if spiral is not None:
        e = _energy(sequence, spiral)
        if e < best_energy:
            best_energy = e
            best_chain = spiral

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

    # Pass 2: Local Search on the best chain
    if best_chain is not None and _time_left() > 25:
        improved = _local_search(sequence, best_chain, max_iterations=20000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Polish
        if _time_left() > 20:
            polished = _polish(sequence, best_chain, max_iterations=5000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=12000,
                                       time_limit=local_search_time * 0.6, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 20:
                final_polish = _polish(sequence, best_chain, max_iterations=4000,
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
