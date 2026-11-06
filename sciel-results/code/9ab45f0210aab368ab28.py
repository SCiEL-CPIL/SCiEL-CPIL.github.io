# EVOLVE-BLOCK-START
import random
import time

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


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
            # Heuristic: prefer tighter packing (fewer free neighbors) to maximize future contacts
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Increase bias for gain, decrease for free space to encourage compactness
            score = 2.0 * gain - 0.5 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _shift_move(chain, rng):
    """
    Perform a shift move: move a short subchain (length 2-5) by 1 unit in one of 4 directions.
    This is cheaper than pivot moves.
    """
    n = len(chain)
    if n < 4:
        return None

    # Random segment length and start
    max_seg_len = min(5, n - 2)
    seg_len = rng.randint(2, max_seg_len)
    start_idx = rng.randint(1, n - seg_len - 1)
    
    # Directions
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = rng.choice(directions)

    # The segment to move is chain[start_idx : start_idx + seg_len]
    # It is attached at start_idx-1 and start_idx+seg_len
    
    segment = chain[start_idx : start_idx + seg_len]
    
    # Check if the new positions are valid
    # 1. New positions must not be occupied by fixed parts
    # 2. New positions must not be occupied by the segment itself (trivial if shift is 1 unit? No, could overlap if seg_len > 1? No, a rigid shift of a straight-ish chain by 1 unit usually doesn't self-intersect unless it's very tight, but we must check against the rest of the chain)
    
    new_segment = []
    for p in segment:
        new_segment.append((p[0] + dx, p[1] + dy))
        
    # Check self-avoidance with the rest of the chain
    # The rest of the chain is chain[:start_idx] + chain[start_idx+seg_len:]
    # We can create a set of occupied points from the rest of the chain
    # To be fast, we only check the new_segment points against the occupied set excluding the old segment points
    
    # For simplicity and speed, we'll just build the new chain and check uniqueness.
    # This is O(N). But N is small (<=100).
    
    new_chain = chain[:start_idx] + new_segment + chain[start_idx+seg_len:]
    
    # Check self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return None
        
    # Check connectivity (unit steps)
    # The connections at the boundaries:
    # 1. chain[start_idx-1] to new_segment[0]
    # 2. new_segment[-1] to chain[start_idx+seg_len]
    # Internal connections in new_segment are preserved by rigid shift.
    
    p1 = chain[start_idx-1]
    p2 = new_segment[0]
    if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) != 1:
        return None
        
    p3 = new_segment[-1]
    p4 = chain[start_idx+seg_len]
    if abs(p3[0]-p4[0]) + abs(p3[1]-p4[1]) != 1:
        return None
        
    return new_chain


def _sa_run(sequence, chain, time_limit, rng, T0=2.0, cooling=0.999):
    """
    Simulated annealing with shift moves.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = T0

    while time.time() - start_time < time_limit:
        if _time_left() < 5:
            break
        new_chain = _shift_move(current_chain, rng)
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
    """Best conformation using multiple strategies: contact-aware greedy growth + fast SA local search."""
    seq_len = len(sequence)

    # Aggressive time budgeting
    # Total 10 sequences. 200s limit.
    # Allocate ~15s per sequence on average, less for short, more for long.
    # But we must finish all 10.
    # Let's use a fixed 12s per sequence to be safe (120s total).
    seq_budget = 12.0
    seq_start = time.time()

    def _seq_left():
        return seq_budget - (time.time() - seq_start)

    # Parameters scaled by length
    if seq_len <= 30:
        num_grow = 100
        sa_time = 4.0
        sa_starts = 5
    elif seq_len <= 50:
        num_grow = 80
        sa_time = 3.0
        sa_starts = 4
    elif seq_len <= 70:
        num_grow = 60
        sa_time = 2.0
        sa_starts = 3
    else:
        num_grow = 40
        sa_time = 1.5
        sa_starts = 2

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Contact-aware greedy growth
    candidates = []
    for i in range(num_grow):
        if _time_left() < 10 or _seq_left() < 2.0:
            break
        bias = 1.0 # Bias is hardcoded in _grow now
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
        candidates.append((value, chain))

    if best_chain is None:
        return [list(p) for p in [(i, 0) for i in range(seq_len)]]

    # Pass 2: SA on the best few chains
    candidates.sort(key=lambda x: x[0])
    seen = set()
    sa_candidates = []
    for value, chain in candidates:
        key = tuple(chain)
        if key in seen:
            continue
        seen.add(key)
        sa_candidates.append(chain)
        if len(sa_candidates) >= sa_starts:
            break

    for idx, chain in enumerate(sa_candidates):
        if _time_left() < 5 or _seq_left() < 1.0:
            break
        t = sa_time if idx == 0 else sa_time * 0.5
        improved, improved_energy = _sa_run(sequence, chain, t, rng)
        if improved is not None and improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
