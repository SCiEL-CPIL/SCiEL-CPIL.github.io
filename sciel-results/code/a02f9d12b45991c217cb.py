"""HP lattice protein folding on cubic lattice - beam search + incremental local search."""
# EVOLVE-BLOCK-START
import time
import random

START_TIME = time.time()
TIME_LIMIT = 230.0  # Safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _compute_energy(sequence, chain):
    """Compute energy of a full chain (negative number of H-H contacts)."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=80, time_budget=12.0):
    """Beam search for HP folding. Returns best full chain or None if timeout."""
    n = len(sequence)
    if n == 0:
        return []

    deadline = START_TIME + time_budget

    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_energy = 0

    beam = [(initial_chain, initial_occupied, initial_energy)]

    for i in range(1, n):
        next_beam = []

        for chain, occupied, energy in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue

                new_energy = energy
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            new_energy -= 1

                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i

                next_beam.append((new_chain, new_occupied, new_energy))

        if not next_beam:
            return None

        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]

        if time.time() > deadline:
            return None

    if not beam:
        return None

    return beam[0][0]


def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy growth."""
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
            score = gain * 2.0 + free * 0.3 + rng.random() * 0.5
            options.append((score, site))

        if not options:
            break

        options.sort(reverse=True)
        top_k = min(3, len(options))
        chosen = random.choice(options[:top_k])[1]
        occupied[chosen] = i
        chain.append(chosen)

    return chain


def _local_search(sequence, chain, max_iters=20000):
    """
    Local search with incremental energy updates and multiple move types:
    - 2-residue moves (move one interior residue to an adjacent free site)
    - 180-degree turns (flip a 3-residue segment)
    - 3-residue shifts (translate a 3-residue segment)
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 4:
        return list(chain), _compute_energy(sequence, chain)

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    h_set = set(i for i in range(n) if sequence[i] == 'H')

    def contact_count_at(pos_idx, position, occ):
        """Count H-H contacts involving residue pos_idx at 'position' (excluding chain-adjacent)."""
        if pos_idx not in h_set:
            return 0
        cnt = 0
        for q in _neighbours(position):
            j = occ.get(q)
            if j is not None and j != pos_idx:
                if j > pos_idx + 1 and j in h_set:
                    cnt += 1
                elif j < pos_idx - 1 and j in h_set:
                    cnt += 1
        return cnt

    def try_move_2residue(i):
        """Move residue i to an adjacent free site. Returns (new_pos, delta_energy) or None."""
        if i < 1 or i > n - 2:
            return None
        pos = best_chain[i]
        prev_pos = best_chain[i - 1]
        next_pos = best_chain[i + 1]

        for q in _neighbours(pos):
            if q == prev_pos or q == next_pos:
                continue
            if q in occupied:
                continue
            if _manhattan(prev_pos, q) != 1:
                continue
            if _manhattan(q, next_pos) != 1:
                continue

            # Compute energy delta
            old_occ = occupied
            # Remove i from old position
            old_cnt_i = contact_count_at(i, pos, old_occ)
            # New position
            # We need to simulate: remove i, add i at q
            # Contacts affected: those involving i at old pos, i at new pos,
            # and any other residue that was in contact with i at old pos or q.
            # Simpler: compute new contact count for i at q, and check if any
            # other H residue loses/gains a contact.
            # For efficiency, we compute delta as:
            #   delta = new_contacts_involving_i - old_contacts_involving_i
            # But we must also account for contacts between i and other residues
            # that change. Since only i moves, the only contacts that change
            # are those involving i.
            new_cnt_i = 0
            if i in h_set:
                for q2 in _neighbours(q):
                    j = old_occ.get(q2)
                    if j is not None and j != i:
                        if j > i + 1 and j in h_set:
                            new_cnt_i += 1
                        elif j < i - 1 and j in h_set:
                            new_cnt_i += 1

            delta = new_cnt_i - old_cnt_i
            return (q, delta)
        return None

    def try_move_180turn(i):
        """180-degree turn at position i (moves i-1, i, i+1). Returns (new_a, new_c, delta) or None."""
        if i < 1 or i > n - 2:
            return None
        a, b, c = best_chain[i - 1], best_chain[i], best_chain[i + 1]
        new_a = (2 * b[0] - c[0], 2 * b[1] - c[1], 2 * b[2] - c[2])
        new_c = (2 * b[0] - a[0], 2 * b[1] - a[1], 2 * b[2] - a[2])

        if new_a == new_c:
            return None
        if new_a == b or new_c == b:
            return None
        if new_a in occupied and occupied[new_a] != i - 1:
            return None
        if new_c in occupied and occupied[new_c] != i + 1:
            return None

        # Compute energy delta
        # Affected residues: i-1, i, i+1
        old_cnt = 0
        for idx in (i - 1, i, i + 1):
            old_cnt += contact_count_at(idx, best_chain[idx], occupied)

        # Simulate new positions
        # Temporarily update occupied
        old_occ = dict(occupied)
        occupied.pop(best_chain[i - 1])
        occupied.pop(best_chain[i])
        occupied.pop(best_chain[i + 1])
        occupied[new_a] = i - 1
        occupied[b] = i
        occupied[new_c] = i + 1

        new_cnt = 0
        for idx, pos in ((i - 1, new_a), (i, b), (i + 1, new_c)):
            new_cnt += contact_count_at(idx, pos, occupied)

        # Restore
        occupied.clear()
        occupied.update(old_occ)

        delta = new_cnt - old_cnt
        return (new_a, new_c, delta)

    def try_move_3shift(i):
        """Shift residues i-1, i, i+1 as a unit. Returns (new_a, new_b, new_c, delta) or None."""
        if i < 1 or i > n - 2:
            return None
        a, b, c = best_chain[i - 1], best_chain[i], best_chain[i + 1]
        for axis in range(3):
            for step in (-1, 1):
                new_a = list(a)
                new_b = list(b)
                new_c = list(c)
                new_a[axis] += step
                new_b[axis] += step
                new_c[axis] += step
                new_a = tuple(new_a)
                new_b = tuple(new_b)
                new_c = tuple(new_c)

                if new_a in occupied and occupied[new_a] != i - 1:
                    continue
                if new_b in occupied and occupied[new_b] != i:
                    continue
                if new_c in occupied and occupied[new_c] != i + 1:
                    continue

                if _manhattan(new_a, new_b) != 1:
                    continue
                if _manhattan(new_b, new_c) != 1:
                    continue

                # Compute energy delta
                old_cnt = 0
                for idx in (i - 1, i, i + 1):
                    old_cnt += contact_count_at(idx, best_chain[idx], occupied)

                old_occ = dict(occupied)
                occupied.pop(best_chain[i - 1])
                occupied.pop(best_chain[i])
                occupied.pop(best_chain[i + 1])
                occupied[new_a] = i - 1
                occupied[new_b] = i
                occupied[new_c] = i + 1

                new_cnt = 0
                for idx, pos in ((i - 1, new_a), (i, new_b), (i + 1, new_c)):
                    new_cnt += contact_count_at(idx, pos, occupied)

                occupied.clear()
                occupied.update(old_occ)

                delta = new_cnt - old_cnt
                return (new_a, new_b, new_c, delta)
        return None

    rng = random.Random(42)

    for it in range(max_iters):
        if time.time() > START_TIME + TIME_LIMIT * 0.92:
            break

        improved = False
        i = rng.randint(1, n - 2)
        move_type = rng.random()

        if move_type < 0.4:
            result = try_move_2residue(i)
            if result is not None:
                new_pos, delta = result
                if delta < 0:
                    old_pos = best_chain[i]
                    occupied.pop(old_pos)
                    occupied[new_pos] = i
                    best_chain[i] = new_pos
                    best_energy += delta
                    improved = True
        elif move_type < 0.7:
            result = try_move_180turn(i)
            if result is not None:
                new_a, new_c, delta = result
                if delta < 0:
                    old_a = best_chain[i - 1]
                    old_c = best_chain[i + 1]
                    occupied.pop(old_a)
                    occupied.pop(old_c)
                    occupied[new_a] = i - 1
                    occupied[new_c] = i + 1
                    best_chain[i - 1] = new_a
                    best_chain[i + 1] = new_c
                    best_energy += delta
                    improved = True
        else:
            result = try_move_3shift(i)
            if result is not None:
                new_a, new_b, new_c, delta = result
                if delta < 0:
                    old_a = best_chain[i - 1]
                    old_b = best_chain[i]
                    old_c = best_chain[i + 1]
                    occupied.pop(old_a)
                    occupied.pop(old_b)
                    occupied.pop(old_c)
                    occupied[new_a] = i - 1
                    occupied[new_b] = i
                    occupied[new_c] = i + 1
                    best_chain[i - 1] = new_a
                    best_chain[i] = new_b
                    best_chain[i + 1] = new_c
                    best_energy += delta
                    improved = True

        if not improved and it > 0 and it % 500 == 0:
            # Occasionally do a full sweep to avoid getting stuck
            pass

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using beam search with local refinement and multiple restarts.
    """
    n = len(sequence)

    # Determine time budget for beam search based on sequence length
    if n <= 48:
        beam_time = 10.0
        beam_width = 100
    elif n <= 80:
        beam_time = 8.0
        beam_width = 70
    else:
        beam_time = 6.0
        beam_width = 50

    # Try beam search first
    best_chain = _beam_search(sequence, dimension, beam_width, beam_time)

    if best_chain is None or len(best_chain) != n:
        # Fallback to greedy growth with randomization
        rng = random.Random(12345 + n)
        best_chain = _greedy_grow(sequence, dimension, rng)

        if len(best_chain) != n:
            # Last resort: straight line
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Local refinement with incremental energy
    best_chain, best_energy = _local_search(sequence, best_chain, max_iters=50000)

    # If we have time, try multiple greedy starts and keep the best
    if time.time() - START_TIME < TIME_LIMIT * 0.7:
        rng = random.Random(99999 + n)

        num_restarts = min(4, max(1, int((TIME_LIMIT * 0.7 - (time.time() - START_TIME)) / 3)))

        for restart in range(num_restarts):
            if time.time() - START_TIME > TIME_LIMIT * 0.88:
                break

            candidate = _greedy_grow(sequence, dimension, rng)
            if len(candidate) != n:
                continue

            candidate, cand_energy = _local_search(sequence, candidate, max_iters=20000)

            if cand_energy < best_energy:
                best_energy = cand_energy
                best_chain = candidate

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
