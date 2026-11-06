"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_DEADLINE = time.time() + 315  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=0.0):
    """One randomized greedy chain growth with pocket-aware scoring."""
    start = (0,) * dimension
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
            dead_end_penalty = -2.0 if free == 0 else 0.0
            trap_penalty = -0.5 if free == 1 else 0.0
            score = gain * 1.5 + 0.12 * free + dead_end_penalty + trap_penalty + bias * rng.random() + (1 - bias) * rng.random() * 0.3
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, i, rng, dimension):
    """Fast pivot move with early collision abort."""
    if i <= 1 or i >= len(chain) - 1:
        return None

    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]

    axis = rng.randint(0, dimension - 1)
    sign = rng.choice([-1, 1])

    new_tail = []
    seen_in_tail = set()
    head_set = set(head)

    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        if axis == 0:
            new_vec = [vec[0], -sign * vec[2], sign * vec[1]]
        elif axis == 1:
            new_vec = [sign * vec[2], vec[1], -sign * vec[0]]
        else:
            new_vec = [-sign * vec[1], sign * vec[0], vec[2]]
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))

        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)

    return head + new_tail


def _reversal_move(chain, i, j, rng):
    """Reverse a segment of the chain from i to j (inclusive)."""
    if i < 1 or j >= len(chain) - 1 or j <= i:
        return None

    # Check boundary connectivity after reversal
    if i > 0:
        d1 = sum(abs(chain[i-1][k] - chain[j][k]) for k in range(len(chain[i])))
        if d1 != 1:
            return None
    if j < len(chain) - 1:
        d2 = sum(abs(chain[j+1][k] - chain[i][k]) for k in range(len(chain[i])))
        if d2 != 1:
            return None

    new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]

    if len(set(new_chain)) != len(new_chain):
        return None

    return new_chain


def _kink_move(chain, i, rng, dimension):
    """Try to move residue i to an adjacent free site (kink move)."""
    if i <= 0 or i >= len(chain) - 1:
        return None

    occupied = set(chain)
    occupied.discard(chain[i])

    prev_pos = chain[i-1]
    next_pos = chain[i+1]

    for site in _neighbours(chain[i]):
        if site in occupied:
            continue

        dist_prev = sum(abs(site[k] - prev_pos[k]) for k in range(dimension))
        dist_next = sum(abs(site[k] - next_pos[k]) for k in range(dimension))

        if dist_prev == 1 and dist_next == 1:
            new_chain = list(chain)
            new_chain[i] = site
            return new_chain

    return None


def _crank_move(chain, i, rng, dimension):
    """Crank move: rotate a local 4-residue segment.
    For residues i-1, i, i+1, i+2, rotate the tail (i+1, i+2) around the bond (i-1, i).
    This is a local move that preserves self-avoidance more often than full-tail pivots."""
    if i < 2 or i + 2 >= len(chain):
        return None

    p0 = chain[i-1]
    p1 = chain[i]
    p2 = chain[i+1]
    p3 = chain[i+2]

    # The bond vector from p0 to p1
    bond = tuple(p1[k] - p0[k] for k in range(dimension))

    # The vector from p1 to p2
    v1 = tuple(p2[k] - p1[k] for k in range(dimension))
    # The vector from p2 to p3
    v2 = tuple(p3[k] - p2[k] for k in range(dimension))

    # For a crank move, we rotate the segment (p2, p3) around the axis through p0-p1
    # In 3D lattice, if bond is along axis a, and v1 is along axis b (perpendicular),
    # then the possible 90° rotations of v1 give 3 options (including identity).
    # We try all 6 possible 90° rotations of the (v1, v2) pair.

    # Simpler: try all 6 rotations of the tail starting at p1
    head = chain[:i+1]  # includes p1
    tail = chain[i+1:]  # starts at p2

    pivot = p1
    head_set = set(head)

    rotations = [
        lambda v: (v[0], -v[2], v[1]),
        lambda v: (v[0], v[2], -v[1]),
        lambda v: (v[2], v[1], -v[0]),
        lambda v: (-v[2], v[1], v[0]),
        lambda v: (-v[1], v[0], v[2]),
        lambda v: (v[1], -v[0], v[2]),
    ]

    for rot in rotations:
        new_tail = []
        seen_in_tail = set()
        valid = True
        for pos in tail:
            vec = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
            new_vec = rot(vec)
            new_pos = (pivot[0] + new_vec[0], pivot[1] + new_vec[1], pivot[2] + new_vec[2])
            if new_pos in seen_in_tail or new_pos in head_set:
                valid = False
                break
            seen_in_tail.add(new_pos)
            new_tail.append(new_pos)

        if valid:
            return head + new_tail

    return None


def _a_star_search(sequence, dimension, max_nodes=50000):
    """A* search for optimal folding of short chains.
    Uses admissible heuristic: remaining H residues that could potentially form contacts."""
    n = len(sequence)
    if n > 35:
        return None

    start = (0,) * dimension
    start_state = (0, [start], {start: 0}, 0)  # (g, chain, occupied, index)

    # Admissible heuristic: count remaining H residues (each can contribute at most 1 contact per H)
    # More refined: count H-H pairs remaining that could be non-bonded
    h = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        h[i] = h[i + 1]
        if sequence[i] == 'H':
            # Each H can form at most (n - i - 2) contacts with later H's (excluding bonded)
            # But that's an overestimate; use a tighter bound
            h[i] += 1  # conservative: each H contributes at most 1

    open_set = []
    heapq.heappush(open_set, (h[0], 0, start_state))
    best_g = {start_state[1]: 0}

    nodes_explored = 0
    while open_set:
        if time.time() > _DEADLINE - 30:
            return None
        f, g, (g_val, chain, occupied, idx) = heapq.heappop(open_set)

        if idx == n - 1:
            # Complete chain found
            return chain

        nodes_explored += 1
        if nodes_explored > max_nodes:
            return None

        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue

            new_idx = idx + 1
            # Compute immediate gain
            gain = 0
            if sequence[new_idx] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < new_idx - 1 and sequence[occupied[q]] == 'H')

            new_g = g + gain  # We maximize gain, so g accumulates contacts
            new_occupied = dict(occupied)
            new_occupied[site] = new_idx
            new_chain = chain + [site]

            # Check if this state was seen with better g
            state_key = tuple(new_chain)
            if state_key in best_g and best_g[state_key] >= new_g:
                continue
            best_g[state_key] = new_g

            new_f = new_g + h[new_idx]
            heapq.heappush(open_set, (-new_f, new_g, (new_g, new_chain, new_occupied, new_idx)))

    return None


def _local_search(sequence, chain, dimension, rng, max_iters=300):
    """Fast local search: accept only improving moves."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    stall_count = 0

    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        move_type = rng.random()
        if move_type < 0.40:
            i = rng.randint(2, len(current) - 2)
            new_chain = _pivot_move_fast(current, i, rng, dimension)
        elif move_type < 0.65:
            i = rng.randint(1, len(current) - 2)
            j = rng.randint(i + 1, min(i + 6, len(current) - 1))
            new_chain = _reversal_move(current, i, j, rng)
        elif move_type < 0.85:
            i = rng.randint(2, len(current) - 3)
            new_chain = _crank_move(current, i, rng, dimension)
        else:
            i = rng.randint(1, len(current) - 2)
            new_chain = _kink_move(current, i, rng, dimension)

        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            stall_count = 0
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        else:
            stall_count += 1
            if stall_count > max_iters // 3:
                break

    return best, best_energy


def _anneal(sequence, chain, dimension, rng, max_iters=500, T_start=4.0, T_end=0.005):
    """Simulated annealing refinement of a chain with adaptive cooling."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy

    if max_iters > 1:
        alpha = (T_end / T_start) ** (1.0 / max_iters)
    else:
        alpha = 1.0
    T = T_start

    accepted = 0
    total = 0
    window_size = 50

    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        move_type = rng.random()
        if move_type < 0.40:
            i = rng.randint(2, len(current) - 2)
            new_chain = _pivot_move_fast(current, i, rng, dimension)
        elif move_type < 0.65:
            i = rng.randint(1, len(current) - 2)
            j = rng.randint(i + 1, min(i + 8, len(current) - 1))
            new_chain = _reversal_move(current, i, j, rng)
        elif move_type < 0.85:
            i = rng.randint(2, len(current) - 3)
            new_chain = _crank_move(current, i, rng, dimension)
        else:
            i = rng.randint(1, len(current) - 2)
            new_chain = _kink_move(current, i, rng, dimension)

        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        total += 1
        if delta < 0 or (T > 0.001 and rng.random() < math.exp(-delta / T)):
            current = new_chain
            current_energy = new_energy
            accepted += 1
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        T *= alpha

        if total >= window_size:
            rate = accepted / total
            if rate < 0.1 and T > 0.01:
                T = max(0.01, T * 1.5)
                alpha = (T_end / T) ** (1.0 / max(1, max_iters - it - 1)) if it < max_iters - 1 else 1.0
            elif rate > 0.9 and T < T_start:
                T = min(T_start, T * 0.8)
                alpha = (T_end / T) ** (1.0 / max(1, max_iters - it - 1)) if it < max_iters - 1 else 1.0
            accepted = 0
            total = 0

    return best, best_energy


def _beam_search(sequence, dimension, rng, beam_width=50, max_depth=60):
    """Beam search over partial chains with contact-aware scoring."""
    n = len(sequence)
    if n > max_depth:
        return None

    start = (0,) * dimension
    beam = [(0, [start], {start: 0})]

    for i in range(1, n):
        if time.time() > _DEADLINE - 30:
            return None
        candidates = []
        for score, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                dead_end = -2.0 if free == 0 else 0.0
                trap = -0.5 if free == 1 else 0.0
                new_score = score + gain * 2.0 + 0.1 * free + dead_end + trap
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_chain = chain + [site]
                candidates.append((new_score, new_chain, new_occupied))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        beam = candidates[:beam_width]

    if not beam:
        return None

    return beam[0][1]


def fold(sequence, dimension):
    """Best conformation using A*, beam search, greedy growth, and simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    n = len(sequence)

    # Determine search effort based on chain length
    if n <= 48:
        n_restarts = 200
        ls_iters = 150
        anneal_iters = 800
        use_beam = True
        use_a_star = True
    elif n <= 80:
        n_restarts = 120
        ls_iters = 200
        anneal_iters = 1000
        use_beam = True
        use_a_star = False
    else:
        n_restarts = 60
        ls_iters = 250
        anneal_iters = 1200
        use_beam = False
        use_a_star = False

    # Phase 0a: A* search for short chains
    if use_a_star and n <= 30:
        a_star_chain = _a_star_search(sequence, dimension, max_nodes=100000)
        if a_star_chain is not None:
            refined, refined_energy = _local_search(sequence, a_star_chain, dimension, rng, max_iters=ls_iters)
            best, best_energy = refined, refined_energy
            if time.time() < _DEADLINE - 60:
                refined2, refined_energy2 = _anneal(sequence, best, dimension, rng,
                                                     max_iters=300, T_start=3.0, T_end=0.01)
                if refined_energy2 < best_energy:
                    best, best_energy = refined2, refined_energy2

    # Phase 0b: Beam search for shorter chains
    if use_beam and n <= 64 and (best is None or time.time() < _DEADLINE - 90):
        beam_chain = _beam_search(sequence, dimension, rng, beam_width=80, max_depth=n)
        if beam_chain is not None:
            refined, refined_energy = _local_search(sequence, beam_chain, dimension, rng, max_iters=ls_iters)
            if best is None or refined_energy < best_energy:
                best, best_energy = refined, refined_energy
            if time.time() < _DEADLINE - 60:
                refined2, refined_energy2 = _anneal(sequence, best, dimension, rng,
                                                     max_iters=300, T_start=3.0, T_end=0.01)
                if refined_energy2 < best_energy:
                    best, best_energy = refined2, refined_energy2

    # Phase 1: Quick local search on multiple restarts with varying bias
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 120:
            break
        progress = r / max(1, n_restarts - 1)
        bias = 0.03 + 0.9 * progress
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=ls_iters)

        if best is None or refined_energy < best_energy:
            best, best_energy = refined, refined_energy

    if best is None:
        for r in range(30):
            chain = _grow(sequence, dimension, rng, bias=0.0)
            if chain is not None:
                best, best_energy = chain, _energy(sequence, chain)
                break

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
        return [list(p) for p in best]

    # Phase 2: Deep annealing on the best solution
    if time.time() < _DEADLINE - 60:
        refined, refined_energy = _anneal(sequence, best, dimension, rng,
                                           max_iters=anneal_iters, T_start=5.0, T_end=0.003)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy

    # Phase 3: Try fresh starts with annealing if time permits
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 50:
        n_extra = min(10, int(time_remaining / 12))
        for extra in range(n_extra):
            if time.time() > _DEADLINE - 25:
                break
            bias = 0.15 + 0.6 * rng.random()
            chain = _grow(sequence, dimension, rng, bias=bias)
            if chain is None:
                continue
            refined, refined_energy = _local_search(sequence, chain, dimension, rng, max_iters=120)
            if time.time() < _DEADLINE - 20:
                refined, refined_energy = _anneal(sequence, refined, dimension, rng,
                                                   max_iters=400, T_start=4.0, T_end=0.01)
            if best is None or refined_energy < best_energy:
                best, best_energy = refined, refined_energy

    # Phase 4: Final polish with local search + short anneal
    if time.time() < _DEADLINE - 5:
        refined, refined_energy = _local_search(sequence, best, dimension, rng, max_iters=200)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy
        if time.time() < _DEADLINE - 3:
            refined, refined_energy = _anneal(sequence, best, dimension, rng,
                                               max_iters=200, T_start=2.0, T_end=0.005)
            if refined_energy < best_energy:
                best, best_energy = refined, refined_energy

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
