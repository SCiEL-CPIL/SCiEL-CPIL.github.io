"""HP lattice protein folding: multi-strategy with fast energy, diverse initialization, and targeted annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 118.0  # Safety margin for 120s budget


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0

    # Map position to index for O(1) lookup
    where = {}
    for i, p in enumerate(chain):
        where[p] = i

    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1

    return -contacts // 2


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def _grow_compact(sequence, rng, bias=1.0):
    """
    Compact growth heuristic: place residues one by one, preferring positions
    that maximize immediate H-H contacts and minimize free neighbors.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    chain = [(0, 0)]
    occupied = {(0, 0)}
    h_positions = set()
    if sequence[0] == "H":
        h_positions.add((0, 0))

    for idx in range(1, n):
        current_pos = chain[-1]
        options = []

        for site in _neighbours(current_pos):
            if site in occupied:
                continue

            # Heuristic score
            score = 0.0

            # 1. Immediate H-H contacts
            if sequence[idx] == "H":
                for q in _neighbours(site):
                    if q in h_positions:
                        score += 5.0 * bias

            # 2. Compactness: fewer free neighbors is better
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score -= free_count * 0.8

            # 3. Add some noise for diversity
            score += rng.random() * 0.5

            options.append((score, site))

        if not options:
            return None  # Dead end

        # Sort by score and pick the best
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]

        chain.append(best_site)
        occupied.add(best_site)
        if sequence[idx] == "H":
            h_positions.add(best_site)

    return chain


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    0 <= i < j < len(chain).
    """
    n = len(chain)
    if i >= j:
        return None

    # Check boundary conditions
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None

    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None

    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]

    # Check self-avoidance of the new segment
    new_segment = new_chain[i:j+1]
    prefix = new_chain[:i]
    suffix = new_chain[j+1:]

    prefix_set = set(prefix)
    suffix_set = set(suffix)

    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None

    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i].
    angle is 1, 2, or 3 (multiples of 90 degrees).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None

    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]

    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))

    new_chain = prefix + rotated_tail

    # Check self-avoidance
    prefix_set = set(prefix)
    for p in rotated_tail:
        if p in prefix_set:
            return None

    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng, T_start=1.0, cooling_rate=0.998, min_T=0.05):
    """
    Simulated Annealing starting from initial_chain.
    Uses tighter cooling schedule for finer local optimization.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)

    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    T = T_start
    cooling_rate = cooling_rate

    deadline = time.time() + time_limit

    stall_counter = 0
    max_stall = 50  # Stricter stall detection

    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break

        # Choose move type: bias towards reversals (shorter segments)
        r = rng.random()
        if r < 0.7:
            # Reversal: prefer shorter segments
            max_seg = min(n - 1, max(2, int(n * 0.3)))
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, min(n - 1, i + max_seg))
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
        else:
            # Pivot
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
            if new_chain is None:
                continue

        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy

        # Stricter Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            stall_counter = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            stall_counter += 1

        T *= cooling_rate
        if T < min_T:
            # Reheat with probability
            if rng.random() < 0.3:
                T = T_start * 0.5
            else:
                break

        if stall_counter > max_stall:
            # Perturb with a larger move
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is not None:
                current_chain = new_chain
                current_energy = _energy(sequence, current_chain)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            stall_counter = 0

        if _remaining_time() < 0.5:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation using diverse initialization and SA.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    best_chain = None
    best_energy = 0  # Energy is negative

    # Phase 1: Generate diverse initial chains using compact growth with different biases
    initial_chains = []

    # Use different bias values and seeds for diversity
    configs = [
        (1.0, 42),
        (0.8, 123),
        (1.2, 777),
        (0.9, 999),
        (1.1, 555)
    ]

    for bias, seed in configs:
        if _remaining_time() < 10.0:
            break
        local_rng = random.Random(seed)
        chain = _grow_compact(sequence, local_rng, bias=bias)
        if chain is not None and _is_valid_chain(chain):
            initial_chains.append(chain)

    # If no valid chains, fallback to a simple straight line
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    # Evaluate and sort initial chains by energy
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))

    scored_chains.sort(key=lambda x: x[0])

    # Keep top K candidates for annealing
    top_k = min(4, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]

    # Phase 2: Anneal each candidate with tighter cooling schedule
    total_time_remaining = _remaining_time()
    time_per_candidate = total_time_remaining / len(candidates)

    for idx, chain in enumerate(candidates):
        if _remaining_time() < 2.0:
            break

        # Allocate time
        alloc_time = min(time_per_candidate, _remaining_time() - 1.0)
        if alloc_time < 0.5:
            continue

        # Use tighter cooling schedule for polish
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng, T_start=1.0, cooling_rate=0.998, min_T=0.05)

        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c

    # Fallback
    if best_chain is None:
        if initial_chains:
            best_chain = initial_chains[0]
        else:
            best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
