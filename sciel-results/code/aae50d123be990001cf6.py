"""HP lattice protein folding: beam search over SAWs with contact-aware scoring + polish SA."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 200.0  # Safety margin for 210s wall-clock limit


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DELTA_IDX = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0

    where = {}
    for i, p in enumerate(chain):
        where[p] = i

    h_positions = [i for i in range(n) if sequence[i] == 'H']
    if len(h_positions) < 2:
        return 0

    contacts = 0
    for i in h_positions:
        p = chain[i]
        for dx, dy in _DELTAS:
            q = (p[0] + dx, p[1] + dy)
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1

    return -contacts


def _partial_contact_score(sequence, chain, occupied):
    """
    Score a partial chain: count H-H contacts already formed + potential future contacts.
    Returns a tuple (contacts, -free_h_neighbors) for lexicographic comparison.
    """
    n = len(chain)
    if n < 2:
        return (0, 0)

    where = {}
    for i, p in enumerate(chain):
        where[p] = i

    h_indices = [i for i in range(n) if sequence[i] == 'H']
    if len(h_indices) < 2:
        return (0, 0)

    contacts = 0
    for i in h_indices:
        p = chain[i]
        for dx, dy in _DELTAS:
            q = (p[0] + dx, p[1] + dy)
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1

    free_h = 0
    for i in h_indices:
        p = chain[i]
        for dx, dy in _DELTAS:
            q = (p[0] + dx, p[1] + dy)
            if q not in occupied:
                free_h += 1

    return (contacts, -free_h)


def _beam_search(sequence, beam_width=80, max_steps=None):
    """
    Beam search over self-avoiding walks with incremental scoring.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    if max_steps is None:
        max_steps = n

    h_mask = [1 if c == 'H' else 0 for c in sequence]

    initial_beams = []
    for d in _DELTAS:
        p0 = (0, 0)
        p1 = (d[0], d[1])
        chain = [p0, p1]
        occupied = {p0, p1}
        where = {p0: 0, p1: 1}
        contacts = 0
        free_h = 0
        if h_mask[0] and h_mask[1]:
            pass  # adjacent, no contact
        for idx in range(2):
            if h_mask[idx]:
                px, py = chain[idx]
                for dx, dy in _DELTAS:
                    q = (px + dx, py + dy)
                    if q not in occupied:
                        free_h += 1
        score = (contacts, -free_h)
        initial_beams.append((score, chain, occupied, where, d, contacts, free_h))

    initial_beams.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
    beam = initial_beams[:beam_width]

    for step in range(2, n):
        if _remaining_time() < 2.0:
            break

        next_beams = []
        for score, chain, occupied, where, last_dir, contacts, free_h in beam:
            current_pos = chain[-1]
            opp = (-last_dir[0], -last_dir[1])

            # Update free_h: remove neighbors of current position that were free
            cur_x, cur_y = current_pos
            if h_mask[step - 1]:
                for dx, dy in _DELTAS:
                    q = (cur_x + dx, cur_y + dy)
                    if q not in occupied:
                        free_h -= 1

            for d in _DELTAS:
                if d == opp:
                    continue
                new_pos = (current_pos[0] + d[0], current_pos[1] + d[1])
                if new_pos in occupied:
                    continue

                new_chain = chain + [new_pos]
                new_occupied = occupied | {new_pos}
                new_where = dict(where)
                new_where[new_pos] = step

                new_contacts = contacts
                new_free_h = free_h

                if h_mask[step]:
                    nx, ny = new_pos
                    for dx, dy in _DELTAS:
                        q = (nx + dx, ny + dy)
                        j = new_where.get(q)
                        if j is not None and j < step - 1 and h_mask[j]:
                            new_contacts += 1
                        elif q not in new_occupied:
                            new_free_h += 1

                new_score = (new_contacts, -new_free_h)
                next_beams.append((new_score, new_chain, new_occupied, new_where, d, new_contacts, new_free_h))

        if not next_beams:
            break

        next_beams.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
        beam = next_beams[:beam_width]

    if not beam:
        return None

    return beam[0][1]


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    """
    n = len(chain)
    if i >= j:
        return None

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

    # Check self-avoidance of the new segment against prefix and suffix
    new_segment = new_chain[i:j+1]
    prefix_set = set(new_chain[:i])
    suffix_set = set(new_chain[j+1:])

    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None

    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i].
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

    prefix_set = set(prefix)
    for p in rotated_tail:
        if p in prefix_set:
            return None

    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng, T_start=1.0, cooling_rate=0.9985, min_T=0.03):
    """
    Simulated Annealing with efficient energy delta computation.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)

    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    T = T_start
    deadline = time.time() + time_limit

    stall_counter = 0
    max_stall = 80

    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break

        r = rng.random()
        if r < 0.7:
            max_seg = min(n - 1, max(2, int(n * 0.25)))
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, min(n - 1, i + max_seg))
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
        else:
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
            if new_chain is None:
                continue

        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy

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
            if rng.random() < 0.25:
                T = T_start * 0.4
            else:
                break

        if stall_counter > max_stall:
            # Larger perturbation
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


def _local_polish(sequence, chain, time_limit, rng):
    """
    Intensive local search: try all short reversals and small pivots systematically.
    """
    n = len(chain)
    if n < 3:
        return chain[:], _energy(sequence, chain)

    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy

    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline and _remaining_time() > 1.0:
        improved = False
        # Try all reversals of length 2 to min(n-1, 15)
        max_len = min(n - 1, 15)
        for i in range(n - 1):
            for j in range(i + 1, min(n, i + 1 + max_len)):
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is None:
                    continue
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    if time.time() > deadline or _remaining_time() < 1.0:
                        return best_chain, best_energy

        # Try pivots at each position with all angles
        for i in range(1, n - 1):
            for angle in [1, 2, 3]:
                new_chain = _apply_pivot(current_chain, i, angle)
                if new_chain is None:
                    continue
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    if time.time() > deadline or _remaining_time() < 1.0:
                        return best_chain, best_energy

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation using beam search + polish.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    best_chain = None
    best_energy = 0  # Energy is negative

    # Phase 1: Beam search with multiple beam widths
    beam_configs = [
        (120, None),
        (80, None),
        (50, None),
    ]

    for beam_width, _ in beam_configs:
        if _remaining_time() < 5.0:
            break
        chain = _beam_search(sequence, beam_width=beam_width)
        if chain is not None and len(chain) == n:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_energy = e
                best_chain = chain

    # If beam search failed, fallback
    if best_chain is None:
        chain = [(0, 0)]
        occupied = {(0, 0)}
        for idx in range(1, n):
            cur = chain[-1]
            found = False
            for d in _DELTAS:
                np = (cur[0] + d[0], cur[1] + d[1])
                if np not in occupied:
                    chain.append(np)
                    occupied.add(np)
                    found = True
                    break
            if not found:
                break
        if len(chain) == n:
            best_chain = chain
            best_energy = _energy(sequence, best_chain)
        else:
            best_chain = [(i, 0) for i in range(n)]
            best_energy = _energy(sequence, best_chain)

    # Phase 2: Local polish
    polish_time = min(20.0, _remaining_time() * 0.35)
    if polish_time > 1.0:
        best_chain, best_energy = _local_polish(sequence, best_chain, polish_time, rng)

    # Phase 3: SA polish with remaining time
    sa_time = min(40.0, _remaining_time() * 0.5)
    if sa_time > 1.0:
        best_chain, best_energy = _anneal(sequence, best_chain, sa_time, rng, T_start=0.5, cooling_rate=0.999, min_T=0.02)

    # Final local polish
    final_time = min(15.0, _remaining_time() * 0.3)
    if final_time > 0.5:
        best_chain, best_energy = _local_polish(sequence, best_chain, final_time, rng)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
