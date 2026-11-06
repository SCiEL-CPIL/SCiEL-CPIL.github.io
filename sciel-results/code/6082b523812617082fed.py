"""HP lattice protein folding on the square lattice: randomized greedy growth + local refinement."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_full(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -4)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            score = 2.0 * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_kink_move(seq, chain, pos_to_idx, i, rng):
    """
    Try a kink move at index i. Returns (new_chain, delta_energy) if successful and improves energy,
    or (None, 0) otherwise.
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return None, 0

    old_pos = chain[i]
    x, y = old_pos
    left = chain[i - 1]
    right = chain[i + 1]

    # Find valid new positions for kink
    candidates = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        new_pos = (x + dx, y + dy)
        if new_pos in pos_to_idx:
            continue
        # Must be adjacent to left and right
        if abs(new_pos[0] - left[0]) + abs(new_pos[1] - left[1]) != 1:
            continue
        if abs(new_pos[0] - right[0]) + abs(new_pos[1] - right[1]) != 1:
            continue
        candidates.append(new_pos)

    if not candidates:
        return None, 0

    new_pos = rng.choice(candidates)

    # Compute energy delta
    is_h = (seq[i] == "H")
    if not is_h:
        # If not H, no energy change from contacts involving i
        # But we still need to check if the move is valid (self-avoiding)
        # The new_pos is not in pos_to_idx, so it's self-avoiding
        # Update chain and return
        new_chain = list(chain)
        new_chain[i] = new_pos
        return new_chain, 0

    # H residue: compute delta
    delta = 0
    h_set = set(j for j, c in enumerate(seq) if c == "H")

    # Contacts lost: H-H contacts involving residue i at old position
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (x + dx, y + dy)
        j = pos_to_idx.get(nb)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if seq[j] == "H":
                delta += 1  # Lost a contact -> energy increases by 1

    # Contacts gained: H-H contacts involving residue i at new position
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (new_pos[0] + dx, new_pos[1] + dy)
        j = pos_to_idx.get(nb)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if seq[j] == "H":
                delta -= 1  # Gained a contact -> energy decreases by 1

    if delta >= 0:
        return None, 0

    new_chain = list(chain)
    new_chain[i] = new_pos
    return new_chain, delta


def _refine(seq, chain, rng, max_iter=500):
    """Local refinement using kink moves. Returns improved chain."""
    n = len(chain)
    current = list(chain)
    pos_to_idx = {p: i for i, p in enumerate(current)}
    current_energy = _energy_full(seq, current)

    for _ in range(max_iter):
        if _remaining() < 5.0:
            break

        i = rng.randint(1, n - 2)
        new_chain, delta = _try_kink_move(seq, current, pos_to_idx, i, rng)
        if new_chain is not None and delta < 0:
            # Update pos_to_idx
            old_pos = current[i]
            new_pos = new_chain[i]
            del pos_to_idx[old_pos]
            pos_to_idx[new_pos] = i
            current = new_chain
            current_energy += delta

    return current


def _segment_relocation(chain, sequence, rng, seg_len, max_attempts=20):
    """
    Try to relocate a segment of length seg_len. Cut out chain[a:a+seg_len],
    then attempt to regrow it connecting chain[a-1] to chain[a+seg_len].
    """
    n = len(chain)
    if n < seg_len + 2:
        return None

    a = rng.randint(1, n - seg_len - 1)
    b = a + seg_len  # b is the index after the segment

    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    # Check Manhattan distance feasibility
    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > seg_len or (seg_len - manhattan) % 2 != 0:
        return None

    rest_occupied = set(chain[:a] + chain[b:])

    for attempt in range(max_attempts):
        path = _dfs_rand(left_neighbor, seg_len, [left_neighbor], {left_neighbor} | rest_occupied, right_neighbor, rng)
        if path is not None and path[-1] == right_neighbor:
            new_frag = path[1:]
            if len(new_frag) == seg_len:
                new_chain = chain[:a] + new_frag + chain[b:]
                if _is_valid_chain(new_chain):
                    return new_chain
    return None


def _dfs_rand(current, remaining, path, occupied, target, rng):
    """Randomized DFS to find a self-avoiding path of length 'remaining' from current to target."""
    if remaining == 0:
        if current == target:
            return list(path)
        return None

    x, y = current
    neighbors = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in occupied:
            continue
        dist_to_end = abs(target[0] - site[0]) + abs(target[1] - site[1])
        if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
            continue
        neighbors.append((dist_to_end, site))

    if not neighbors:
        return None

    rng.shuffle(neighbors)
    neighbors.sort(key=lambda x: x[0])

    for _, site in neighbors:
        path.append(site)
        occupied.add(site)
        result = _dfs_rand(site, remaining - 1, path, occupied, target, rng)
        if result is not None:
            return result
        path.pop()
        occupied.discard(site)
    return None


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def fold(sequence, dimension):
    """Best conformation using randomized growth + local refinement + segment relocation."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    best = None
    best_energy = 0

    # Phase 1: Randomized greedy growth
    num_grow = min(200, max(50, int(3000 / n)))
    for _ in range(num_grow):
        if _remaining() < 5.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy_full(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value

    if best is None:
        best = [(i, 0) for i in range(n)]
        best_energy = _energy_full(sequence, best)

    # Phase 2: Local refinement on best chain
    if _remaining() > 5.0:
        refined = _refine(sequence, best, rng, max_iter=1000)
        refined_energy = _energy_full(sequence, refined)
        if refined_energy < best_energy:
            best, best_energy = refined, refined_energy

    # Phase 3: Segment relocation on best chain
    if _remaining() > 5.0:
        for seg_len in (4, 6, 8, 10):
            if _remaining() < 5.0:
                break
            for _ in range(20):
                if _remaining() < 5.0:
                    break
                new_chain = _segment_relocation(best, sequence, rng, seg_len, max_attempts=10)
                if new_chain is not None:
                    new_energy = _energy_full(sequence, new_chain)
                    if new_energy < best_energy:
                        best, best_energy = new_chain, new_energy
                        # Refine after relocation
                        if _remaining() > 5.0:
                            refined = _refine(sequence, best, rng, max_iter=500)
                            refined_energy = _energy_full(sequence, refined)
                            if refined_energy < best_energy:
                                best, best_energy = refined, refined_energy

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
