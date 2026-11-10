"""HP lattice protein folding on the cubic lattice – fast local-search design."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 330.0  # evaluator kills at 330s wall-clock
_START_TIME = time.time()
_SAFETY_MARGIN = 10.0  # seconds before deadline we must stop


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - _elapsed()


def _neighbors3d(p):
    """Yield the 6 face-neighbours of a 3D lattice point (tuple of 3 ints)."""
    x, y, z = p
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _build_occupied(chain):
    """Build dict: lattice point -> index in chain."""
    return {p: i for i, p in enumerate(chain)}


def _energy_from_occupied(sequence, occupied):
    """Count H-H contacts (non-adjacent along chain) using occupied dict."""
    contacts = 0
    for i, p in enumerate(occupied):
        if sequence[i] != 'H':
            continue
        for q in _neighbors3d(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _energy(sequence, chain):
    occupied = _build_occupied(chain)
    return _energy_from_occupied(sequence, occupied)


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = abs(p[0] - prev[0]) + abs(p[1] - prev[1]) + abs(p[2] - prev[2])
            if dist != 1:
                return False
    return True


def _spiral_init(sequence, dimension):
    """
    Generate an initial 3D chain using a compact spiral/growth heuristic.
    Place H residues in a dense core to maximize contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    # Start with a small 3D spiral that packs tightly
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}

    # Direction cycle for 3D spiral: right, up, left, down, forward, back
    # We'll use a simple expanding square spiral in 3D
    # For compactness, alternate between xy-plane spirals and z-steps

    directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]

    # Build a compact 3D path
    # Strategy: grow in a pattern that keeps the chain close to origin
    # Use a "snake" pattern in 3D

    # Simple approach: interleave moves to stay compact
    # Alternate between +x, +y, +z, -x, -y, -z in a pattern that avoids self-contact
    # This is a heuristic; local search will fix it

    rng = random.Random(42)
    for i in range(1, n):
        # Try to place next residue in a direction that maximizes H-H contacts
        best_dir = None
        best_score = -1
        for dx, dy, dz in directions:
            np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
            if np_ in occupied:
                continue
            # Score: prefer directions that are near existing H residues
            score = 0
            if sequence[i] == 'H':
                for q in _neighbors3d(np_):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 2
            # Also prefer staying compact (smaller norm)
            norm = abs(np_[0]) + abs(np_[1]) + abs(np_[2])
            score -= norm * 0.1
            score += rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_dir = np_
        if best_dir is None:
            # Fallback: try any non-occupied neighbor
            for dx, dy, dz in directions:
                np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
                if np_ not in occupied:
                    best_dir = np_
                    break
        if best_dir is None:
            # Give up, return what we have (shouldn't happen for reasonable n)
            break
        chain.append(best_dir)
        occupied[best_dir] = i

    return chain


def _local_search(sequence, chain, max_iters=5000, time_limit=5.0):
    """
    Local search using single-residue moves (pick a residue, try to move it
    to a nearby empty site) and 180-degree turns.
    Uses O(1) occupancy checks via dict.
    """
    n = len(chain)
    if n < 3:
        return chain

    occupied = _build_occupied(chain)
    best_energy = _energy_from_occupied(sequence, occupied)
    best_chain = list(chain)

    start_time = time.time()
    rng = random.Random(12345)

    # Precompute H positions
    h_indices = [i for i in range(n) if sequence[i] == 'H']

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 2.0:
            break

        # Pick a random residue to move (prefer H residues for more impact)
        if h_indices and rng.random() < 0.7:
            i = rng.choice(h_indices)
        else:
            i = rng.randint(1, n - 2)  # avoid ends for connectivity

        if i == 0 or i == n - 1:
            continue

        p = chain[i]
        prev_p = chain[i - 1]
        next_p = chain[i + 1]

        # Try moving residue i to a neighbor of its current position (or nearby)
        # Must maintain connectivity: new position must be adjacent to prev_p and next_p
        # OR we can do a "pull" move: move i to a site adjacent to both prev and next

        # Candidate sites: neighbors of prev_p that are also neighbors of next_p (180-turn)
        # OR neighbors of prev_p (if we also allow next to shift – too complex)
        # Simplest: 180-degree turns – move i to a site that is adjacent to both prev and next

        candidates = []
        for q in _neighbors3d(prev_p):
            if q == p:
                continue
            if q == next_p:
                continue
            if q in occupied:
                continue
            # Check q is adjacent to next_p
            dist = abs(q[0] - next_p[0]) + abs(q[1] - next_p[1]) + abs(q[2] - next_p[2])
            if dist == 1:
                candidates.append(q)

        if not candidates:
            # Try moving i to any neighbor of prev_p (this breaks connectivity with next)
            # Instead, try a different move: shift a segment
            # For simplicity, skip this iteration
            continue

        new_p = rng.choice(candidates)

        # Compute energy delta
        # Remove p from occupied, add new_p
        old_energy = best_energy

        # Check if move improves energy
        # Delta: contacts involving p and new_p
        delta = 0
        # Contacts lost at p
        if sequence[i] == 'H':
            for q in _neighbors3d(p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    delta -= 1  # losing a contact
        # Contacts gained at new_p
        if sequence[i] == 'H':
            for q in _neighbors3d(new_p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    delta += 1  # gaining a contact

        new_energy = old_energy + delta

        if new_energy < best_energy or (new_energy == best_energy and rng.random() < 0.1):
            # Accept move
            occupied.pop(p)
            occupied[new_p] = i
            chain[i] = new_p
            best_energy = new_energy
            best_chain = list(chain)

    return best_chain


def _simulated_annealing(sequence, chain, max_iters=20000, time_limit=8.0):
    """
    Simulated annealing with single-residue 180-degree turn moves.
    """
    n = len(chain)
    if n < 3:
        return chain

    occupied = _build_occupied(chain)
    best_energy = _energy_from_occupied(sequence, occupied)
    best_chain = list(chain)

    current_chain = list(chain)
    current_energy = best_energy

    start_time = time.time()
    rng = random.Random(999)

    T = 2.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(1, max_iters))

    h_indices = [i for i in range(n) if sequence[i] == 'H']

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 2.0:
            break

        T *= cooling

        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        if i == 0 or i == n - 1:
            continue

        p = current_chain[i]
        prev_p = current_chain[i - 1]
        next_p = current_chain[i + 1]

        # Find 180-turn candidates
        candidates = []
        for q in _neighbors3d(prev_p):
            if q == p or q == next_p:
                continue
            if q in occupied:
                continue
            dist = abs(q[0] - next_p[0]) + abs(q[1] - next_p[1]) + abs(q[2] - next_p[2])
            if dist == 1:
                candidates.append(q)

        if not candidates:
            continue

        new_p = rng.choice(candidates)

        # Compute energy delta
        delta = 0
        if sequence[i] == 'H':
            for q in _neighbors3d(p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    delta -= 1
        if sequence[i] == 'H':
            for q in _neighbors3d(new_p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    delta += 1

        new_energy = current_energy + delta

        if new_energy < current_energy or rng.random() < _exp(-delta / max(T, 1e-10)):
            # Accept
            occupied.pop(p)
            occupied[new_p] = i
            current_chain[i] = new_p
            current_energy = new_energy

            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)

    return best_chain


def _exp(x):
    """Safe exp that returns 0 for very negative x."""
    if x < -100:
        return 0.0
    import math
    return math.exp(x)


def _multi_start_fold(sequence, dimension):
    """
    Multi-start local search with time-aware budgeting.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    best_chain = None
    best_energy = 0

    # Time budget per sequence: we have 330s total for 26 sequences
    # So ~12s per sequence on average, but we adapt
    seq_time_budget = max(1.0, min(15.0, _time_left() / 26.0))

    # Generate several initial conformations
    inits = []

    # Spiral init
    c = _spiral_init(sequence, dimension)
    if len(c) == n:
        inits.append(c)

    # Random walks (a few)
    rng = random.Random(777 + n)
    for _ in range(3):
        c = _random_walk_init(sequence, dimension, rng)
        if c is not None and len(c) == n:
            inits.append(c)

    if not inits:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]

    # Local search on each init
    for c in inits:
        if _time_left() < 2.0:
            break

        # Local search first (fast)
        c_ls = _local_search(sequence, c, max_iters=2000, time_limit=min(3.0, seq_time_budget / 2))
        e_ls = _energy(sequence, c_ls)

        if best_chain is None or e_ls < best_energy:
            best_chain = c_ls
            best_energy = e_ls

        # Simulated annealing on the best so far
        if _time_left() > 3.0:
            c_sa = _simulated_annealing(sequence, c_ls, max_iters=10000, time_limit=min(5.0, seq_time_budget / 2))
            e_sa = _energy(sequence, c_sa)
            if e_sa < best_energy:
                best_chain = c_sa
                best_energy = e_sa

    if best_chain is None:
        best_chain = inits[0]

    return [list(p) for p in best_chain]


def _random_walk_init(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for i in range(1, n):
        # Shuffle directions for randomness
        dirs = directions[:]
        rng.shuffle(dirs)
        placed = False
        for dx, dy, dz in dirs:
            np_ = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
            if np_ not in occupied:
                chain.append(np_)
                occupied[np_] = i
                placed = True
                break
        if not placed:
            return None  # stuck
    return chain


def fold(sequence, dimension):
    """
    Fold the HP sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    if _time_left() < 1.0:
        # Emergency: return a straight line
        return [[i, 0, 0] for i in range(len(sequence))]

    result = _multi_start_fold(sequence, dimension)

    # Final validity check
    if result and len(result) == len(sequence):
        # Convert to tuples for validation
        chain_tuples = [tuple(p) for p in result]
        if not _is_valid_chain(chain_tuples):
            # Fallback to straight line
            return [[i, 0, 0] for i in range(len(sequence))]

    return result
# EVOLVE-BLOCK-END
