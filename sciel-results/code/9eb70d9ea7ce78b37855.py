# EVOLVE-BLOCK-START
import random
import time

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 190.0  # Safety margin for 210s kill


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


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias: weight for new contacts vs free space.
    """
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
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: emphasize contacts (gain) heavily, slight preference for free space to avoid dead ends
            score = bias * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a suffix of the chain by 90, 180, or 270 degrees."""
    n = len(chain)
    if n < 4:
        return None

    # Choose a random pivot index (not the first or last)
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]

    # Get the segment to rotate (from pivot_idx to end)
    segment = chain[pivot_idx:]

    # Try different rotations
    for angle in (90, -90, 180):
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:  # 180
                ndx, ndy = -dx, -dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))

        # Check if the rotated segment is self-avoiding and doesn't collide with the fixed part
        new_chain = chain[:pivot_idx] + rotated

        # Quick check: all points must be unique
        if len(set(new_chain)) != len(new_chain):
            continue

        # Check unit steps
        valid = True
        for i in range(1, len(new_chain)):
            dx = abs(new_chain[i][0] - new_chain[i - 1][0])
            dy = abs(new_chain[i][1] - new_chain[i - 1][1])
            if dx + dy != 1:
                valid = False
                break
        if valid:
            return new_chain

    return None


def _crankshaft_move(chain, rng):
    """
    Perform a crankshaft move: rotate a short internal subchain (length 2 or 3).
    """
    n = len(chain)
    if n < 4:
        return None

    # Choose a random start index for the subchain to rotate
    # Subchain length can be 2 or 3
    sub_len = rng.choice([2, 3])
    if n < sub_len + 2:
        return None

    start_idx = rng.randint(1, n - sub_len - 1)
    pivot = chain[start_idx - 1]  # The fixed point before the subchain

    # The subchain to rotate is chain[start_idx : start_idx + sub_len]
    segment = chain[start_idx : start_idx + sub_len]

    for angle in (90, -90, 180):
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))

        new_chain = chain[:start_idx] + rotated + chain[start_idx + sub_len:]

        if len(set(new_chain)) != len(new_chain):
            continue

        valid = True
        for i in range(1, len(new_chain)):
            dx = abs(new_chain[i][0] - new_chain[i - 1][0])
            dy = abs(new_chain[i][1] - new_chain[i - 1][1])
            if dx + dy != 1:
                valid = False
                break
        if valid:
            return new_chain

    return None


def _local_search(sequence, chain, time_limit, rng):
    """
    Simulated annealing local search with pivot and crankshaft moves.
    """
    if chain is None:
        return None, 0

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.997

    while time.time() - start_time < time_limit:
        if _time_left() < 5:
            break

        # Try a pivot or crankshaft move
        if rng.random() < 0.7:
            new_chain = _pivot_move(current_chain, rng)
        else:
            new_chain = _crankshaft_move(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple strategies: 
    1. Generate diverse initial chains via greedy growth with varying bias.
    2. Refine top candidates using SA with pivot/crankshaft moves.
    """
    seq_len = len(sequence)

    # Time budgeting
    # We have ~190s total. 
    # Allocate based on sequence length.
    # Shorter sequences can be solved faster, but we want to maximize score.
    # Let's reserve ~15s per sequence on average, but adjust.
    if seq_len <= 30:
        num_grow = 100
        sa_time_per_candidate = 3.0
        num_sa_candidates = 5
    elif seq_len <= 50:
        num_grow = 80
        sa_time_per_candidate = 2.5
        num_sa_candidates = 4
    elif seq_len <= 70:
        num_grow = 60
        sa_time_per_candidate = 2.0
        num_sa_candidates = 3
    else:
        num_grow = 40
        sa_time_per_candidate = 1.5
        num_sa_candidates = 2

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Contact-aware greedy growth with varying bias
    candidates = []
    for i in range(num_grow):
        if _time_left() < 10:
            break
        
        # Vary bias to explore different compactness levels
        # Higher bias = more compact/contact-focused
        # Lower bias = more spread out
        bias = 0.5 + rng.random() * 2.0 
        
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
        candidates.append((value, chain))

    if best_chain is None:
        # Fallback: linear chain
        return [list(p) for p in [(i, 0) for i in range(seq_len)]]

    # Pass 2: SA on the best few chains
    # Sort by energy (ascending, since negative is better)
    candidates.sort(key=lambda x: x[0])
    
    # Deduplicate to avoid wasting time on identical structures
    seen = set()
    sa_candidates = []
    for value, chain in candidates:
        key = tuple(chain)
        if key in seen:
            continue
        seen.add(key)
        sa_candidates.append(chain)
        if len(sa_candidates) >= num_sa_candidates:
            break

    for idx, chain in enumerate(sa_candidates):
        if _time_left() < 5:
            break
            
        # Allocate time for SA
        t = sa_time_per_candidate
        # Give slightly more time to the first candidate as it's the best initial
        if idx == 0:
            t *= 1.5
            
        improved_chain, improved_energy = _local_search(sequence, chain, t, rng)
        
        if improved_chain is not None and improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
