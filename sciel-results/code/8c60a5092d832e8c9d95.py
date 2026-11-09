"""HP lattice protein folding: beam-search growth + local search + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0  # safety margin before 210 s kill


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _contact_count(sequence, chain):
    """Return the number of H-H contacts (energy = -contacts//2)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return contacts // 2


def _beam_search(sequence, dimension, beam_width=80, max_time=6.0):
    """Beam search over self-avoiding walks with contact-aware scoring."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]

    start_time = time.time()

    # Each beam entry: (score, chain, occupied_dict)
    # score = negative contacts so far (lower is better)
    initial_chain = [(0, 0)]
    initial_occupied = {(0, 0): 0}
    initial_score = 0

    beam = [(initial_score, initial_chain, initial_occupied)]

    for step in range(1, n):
        if time.time() - start_time > max_time or _remaining_time() < 1.0:
            break

        new_beam = []
        per_parent_keep = max(1, beam_width // 3)

        for score, chain, occupied in beam:
            last_pos = chain[-1]
            candidates = []

            for pos in _neighbours(last_pos):
                if pos in occupied:
                    continue

                # Compute new contacts formed by placing residue `step` at `pos`
                delta_contacts = 0
                if sequence[step] == "H":
                    for q in _neighbours(pos):
                        j = occupied.get(q)
                        if j is not None and j < step - 1 and sequence[j] == "H":
                            delta_contacts += 1

                new_score = score - delta_contacts
                new_chain = chain + [pos]
                new_occupied = dict(occupied)
                new_occupied[pos] = step
                candidates.append((new_score, new_chain, new_occupied))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                for c in candidates[:per_parent_keep]:
                    new_beam.append(c)

        if not new_beam:
            break

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return [(i, 0) for i in range(n)]

    # Return the best complete chain from the beam
    best_score, best_chain, _ = beam[0]
    return best_chain


def _greedy_complete(sequence, chain, occupied):
    """Greedily complete a partial chain by choosing the best next position."""
    n = len(sequence)
    current_len = len(chain)

    for step in range(current_len, n):
        last_pos = chain[-1]
        best_pos = None
        best_score = float("inf")

        for pos in _neighbours(last_pos):
            if pos in occupied:
                continue

            score = 0
            if sequence[step] == "H":
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and j < step - 1 and sequence[j] == "H":
                        score -= 1

            if score < best_score:
                best_score = score
                best_pos = pos

        if best_pos is None:
            break

        chain.append(best_pos)
        occupied[best_pos] = step

    if len(chain) == n:
        return chain

    return [(i, 0) for i in range(n)]


def _apply_reversal(chain, a, b):
    """
    Apply a reversal move: reverse chain[a:b+1].
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if a < 1 or b >= n or b <= a:
        return None

    # Check adjacency constraints at the junctions
    p_prev = chain[a - 1]
    p_end = chain[b]
    p_start = chain[a]
    p_next = chain[b + 1] if b + 1 < n else None

    if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
        return None
    if p_next is not None:
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None

    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]

    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None

    return new_chain


def _apply_pivot(chain, i, rotation):
    """
    Apply a pivot move: rotate the tail after position i by `rotation` degrees (90, 180, 270).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i < 1 or i >= n - 2:
        return None

    pivot = chain[i]
    tail = chain[i + 1:]
    prefix_set = set(chain[:i + 1])

    dx = tail[0][0] - pivot[0]
    dy = tail[0][1] - pivot[1]

    # Rotation mappings
    if rotation == 1:      # 90 deg CCW
        rx, ry = -dy, dx
    elif rotation == 2:    # 180 deg
        rx, ry = -dx, -dy
    elif rotation == 3:    # 270 deg CCW
        rx, ry = dy, -dx
    else:
        return None

    # Check the first step after pivot is a unit move
    if abs(rx) + abs(ry) != 1:
        return None

    new_tail = [None] * len(tail)
    new_tail[0] = (pivot[0] + rx, pivot[1] + ry)

    # Build the rotated tail
    for k in range(1, len(tail)):
        px, py = tail[k][0] - pivot[0], tail[k][1] - pivot[1]
        # Apply rotation to the vector from pivot
        if rotation == 1:
            nx, ny = -py, px
        elif rotation == 2:
            nx, ny = -px, -py
        else:
            nx, ny = py, -px
        new_tail[k] = (pivot[0] + nx, pivot[1] + ny)

    # Check self-avoidance: no tail point in prefix, no duplicates in tail
    for p in new_tail:
        if p in prefix_set:
            return None
    if len(set(new_tail)) != len(new_tail):
        return None

    # Check the junction at i+1 is valid (unit step from chain[i])
    if abs(new_tail[0][0] - pivot[0]) + abs(new_tail[0][1] - pivot[1]) != 1:
        return None

    new_chain = chain[:i + 1] + new_tail
    return new_chain


def _local_search(sequence, chain, time_limit):
    """Move-based local search with incremental energy updates. Mutates chain in place."""
    n = len(chain)
    if n < 4:
        return _energy(sequence, chain)

    best = _energy(sequence, chain)
    deadline = time.time() + time_limit

    while time.time() < deadline:
        improved = False

        # --- Reversal Moves ---
        for a in range(1, n - 2):
            for b in range(a + 1, n - 1):
                if time.time() > deadline:
                    break
                new_chain = _apply_reversal(chain, a, b)
                if new_chain is None:
                    continue
                e = _energy(sequence, new_chain)
                if e < best:
                    chain[:] = new_chain
                    best = e
                    improved = True
            if time.time() > deadline:
                break
        if improved:
            continue

        # --- Pivot Moves ---
        for i in range(1, n - 2):
            if time.time() > deadline:
                break
            for rotation in (1, 2, 3):
                new_chain = _apply_pivot(chain, i, rotation)
                if new_chain is None:
                    continue
                e = _energy(sequence, new_chain)
                if e < best:
                    chain[:] = new_chain
                    best = e
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best


def _simulated_annealing(sequence, chain, time_limit):
    """Simulated annealing to escape local minima. Mutates chain in place."""
    n = len(chain)
    if n < 4:
        return _energy(sequence, chain)

    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]

    deadline = time.time() + time_limit
    start_time = time.time()
    total_time = max(0.1, time_limit)
    t0 = 100.0
    t_min = 0.1

    while time.time() < deadline:
        # Temperature decay
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / total_time)
        temperature = max(t_min, t0 * (1.0 - progress) ** 2)

        # Pick a random move
        if random.random() < 0.5:
            # Reversal
            a = random.randint(1, n - 3)
            b = random.randint(a + 1, n - 2)
            new_chain = _apply_reversal(chain, a, b)
        else:
            # Pivot
            i = random.randint(1, n - 3)
            rotation = random.randint(1, 3)
            new_chain = _apply_pivot(chain, i, rotation)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or random.random() < pow(2.718281828, -delta / temperature):
            chain[:] = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]

    chain[:] = best_chain
    return best_energy


def fold(sequence, dimension):
    """Best conformation from beam-search growth, local search, and simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)

    best_chain = None
    best_energy = 0

    # Phase 1: Beam search for a good starting conformation
    if _remaining_time() > 2.0:
        beam_time = min(6.0, _remaining_time() * 0.3)
        chain = _beam_search(sequence, dimension, beam_width=80, max_time=beam_time)
        if chain and len(chain) == n:
            e = _energy(sequence, chain)
            best_chain = chain[:]
            best_energy = e

    # Phase 2: Local search polish
    if best_chain is not None and _remaining_time() > 1.0:
        ls_time = min(_remaining_time() * 0.3, 30.0)
        _local_search(sequence, best_chain, ls_time)
        e = _energy(sequence, best_chain)
        if e < best_energy:
            best_energy = e

    # Phase 3: Simulated annealing to escape local minima
    if best_chain is not None and _remaining_time() > 1.0:
        sa_time = _remaining_time() * 0.7
        _simulated_annealing(sequence, best_chain, sa_time)
        e = _energy(sequence, best_chain)
        if e < best_energy:
            best_energy = e

    # Fallback if nothing worked
    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
