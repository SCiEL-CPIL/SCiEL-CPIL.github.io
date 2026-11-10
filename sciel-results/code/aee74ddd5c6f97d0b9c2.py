"""HP lattice protein folding: regrow-SA with efficient energy deltas."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 195.0


def _check_deadline():
    return time.time() < _DEADLINE


def _energy(sequence, chain):
    """Compute HP energy: -1 for each H-H contact (non-bonded neighbours)."""
    if len(chain) < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        x, y = p
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            j = pos_to_idx.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _count_contacts(sequence, chain, i, j):
    """Count H-H contacts between positions i and j (not bonded)."""
    if abs(i - j) <= 1:
        return 0
    x1, y1 = chain[i]
    x2, y2 = chain[j]
    if abs(x1 - x2) + abs(y1 - y2) != 1:
        return 0
    if sequence[i] == 'H' and sequence[j] == 'H':
        return 1
    return 0


def _energy_delta_insert(sequence, chain, pos, new_point):
    """
    Compute energy change when inserting new_point at position pos in chain.
    chain is the chain WITHOUT the inserted point.
    """
    n = len(chain)
    delta = 0
    # Contacts between new_point and existing chain positions
    x, y = new_point
    for idx in range(n):
        if abs(idx - pos) <= 1:
            continue  # bonded
        cx, cy = chain[idx]
        if abs(x - cx) + abs(y - cy) == 1:
            if sequence[pos] == 'H' and sequence[idx] == 'H':
                delta -= 1
    return delta


def _regrow_sa(sequence, chain, rng, iterations, t_start, t_end):
    """
    Regrow simulated annealing: pick a cut point, regrow the tail with SA-like moves.
    This allows large conformational changes that single-residue moves cannot reach.
    """
    n = len(chain)
    if n < 5:
        return chain, _energy(sequence, chain)

    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    t = t_start
    cooling = (t_start - t_end) / max(iterations, 1)

    for it in range(iterations):
        if not _check_deadline():
            break
        t = max(t - cooling, t_end)

        # Pick a cut point: keep prefix [0..cut-1], regrow suffix [cut..n-1]
        cut = rng.randint(3, n - 3)
        prefix = chain[:cut]
        occupied = {p: i for i, p in enumerate(prefix)}

        # Regrow the tail
        tail = [prefix[-1]]
        tail_occupied = set(occupied.keys())
        success = True

        for i in range(cut, n):
            last = tail[-1]
            x, y = last
            candidates = []
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) not in tail_occupied:
                    # Score: prefer H-H contacts
                    score = 0.0
                    if sequence[i] == 'H':
                        for j in range(cut):
                            if sequence[j] == 'H':
                                px, py = prefix[j]
                                if abs(nx - px) + abs(ny - py) == 1:
                                    score += 5.0
                        # Also check contacts with already-placed tail
                        for j in range(len(tail) - 1):
                            if sequence[cut + j] == 'H':
                                px, py = tail[j]
                                if abs(nx - px) + abs(ny - py) == 1 and abs((cut + j) - i) > 1:
                                    score += 5.0
                    # Free space bonus
                    free = 0
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if (nx + dx, ny + dy) not in tail_occupied:
                            free += 1
                    score += 0.1 * free
                    score += rng.random() * 2.0  # temperature-scaled randomness
                    candidates.append((score, (nx, ny)))

            if not candidates:
                success = False
                break

            # Pick with temperature: higher score = more likely
            candidates.sort(key=lambda c: c[0], reverse=True)
            # Boltzmann-like selection
            if t > 0.01 and rng.random() < 0.3:
                # Sometimes pick a random candidate for exploration
                idx = rng.randint(0, min(3, len(candidates) - 1))
            else:
                idx = 0
            site = candidates[idx][1]
            tail_occupied.add(site)
            tail.append(site)

        if not success:
            continue

        new_chain = prefix + tail[1:]
        if len(new_chain) != n:
            continue

        new_energy = _energy(sequence, new_chain)

        if new_energy < best_energy:
            best_chain = list(new_chain)
            best_energy = new_energy

        # Accept/reject with SA criterion
        if new_energy < current_energy:
            current_energy = new_energy
            chain = list(new_chain)
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < (10 ** (-delta_e / max(t, 0.01))):
                current_energy = new_energy
                chain = list(new_chain)
            # else: reject, keep old chain

    if best_energy < current_energy:
        return best_chain, best_energy
    return chain, current_energy


def _greedy_grow(sequence, rng):
    """Greedy growth with lookahead: prefer sites that create H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    h_positions = []
    if sequence[0] == 'H':
        h_positions.append((0, 0))

    for i in range(1, n):
        last = chain[-1]
        x, y = last
        candidates = []
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in occupied:
                continue
            score = 0.0
            if sequence[i] == 'H':
                for hx, hy in h_positions:
                    if abs(nx - hx) + abs(ny - hy) == 1:
                        score += 10
            free = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (nx + dx, ny + dy) not in occupied:
                    free += 1
            score += 0.1 * free
            score += rng.random() * 0.5
            candidates.append((score, (nx, ny)))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        site = candidates[0][1]
        occupied[site] = i
        chain.append(site)
        if sequence[i] == 'H':
            h_positions.append(site)

    return chain


def _random_walk(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    chain = [(0, 0)]
    occupied = {(0, 0): 0}

    for i in range(1, n):
        last = chain[-1]
        x, y = last
        candidates = []
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) not in occupied:
                candidates.append((nx, ny))
        if not candidates:
            return None
        site = rng.choice(candidates)
        occupied[site] = i
        chain.append(site)

    return chain


def _local_search(sequence, chain, rng, max_iter=3000):
    """
    Local search with segment relocation moves.
    Tries to move short segments to new positions to improve energy.
    """
    n = len(chain)
    if n < 5:
        return chain, _energy(sequence, chain)

    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    pos_to_idx = {p: i for i, p in enumerate(chain)}

    for it in range(max_iter):
        if not _check_deadline():
            break

        # Pick a segment to relocate
        seg_len = rng.randint(2, min(5, n - 2))
        i = rng.randint(1, n - seg_len - 1)
        j = i + seg_len - 1

        # Extract segment
        seg = chain[i:j + 1]
        anchor = chain[i - 1]  # position before segment

        # Remove segment
        for p in seg:
            pos_to_idx.pop(p, None)
        new_chain = chain[:i] + chain[j + 1:]

        # Try to reinsert segment at position i, adjacent to anchor
        ax, ay = anchor
        found = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # Try placing segment starting at (ax+dx, ay+dy)
            offsets = [(p[0] - seg[0][0], p[1] - seg[0][1]) for p in seg]
            new_positions = []
            valid = True
            sx, sy = ax + dx, ay + dy
            for ox, oy in offsets:
                px, py = sx + ox, sy + oy
                if (px, py) in pos_to_idx:
                    valid = False
                    break
                new_positions.append((px, py))

            if not valid or len(new_positions) != len(seg):
                continue

            # Check that new positions don't overlap with each other
            if len(set(new_positions)) != len(new_positions):
                continue

            # Build new chain
            candidate_chain = new_chain[:i] + new_positions + new_chain[i:]
            # Verify self-avoidance
            if len(set(candidate_chain)) != n:
                continue

            new_energy = _energy(sequence, candidate_chain)

            if new_energy < best_energy:
                best_chain = list(candidate_chain)
                best_energy = new_energy

            if new_energy < current_energy:
                current_energy = new_energy
                chain = list(candidate_chain)
                pos_to_idx = {p: idx for idx, p in enumerate(chain)}
                found = True
            else:
                # Accept with probability based on energy difference
                delta_e = new_energy - current_energy
                if delta_e < 0 or rng.random() < (10 ** (-delta_e / 0.5)):
                    current_energy = new_energy
                    chain = list(candidate_chain)
                    pos_to_idx = {p: idx for idx, p in enumerate(chain)}
                    found = True

            if found:
                break

        if not found:
            # Restore original
            for idx, p in enumerate(seg):
                pos_to_idx[p] = i + idx
            chain = new_chain[:i] + seg + new_chain[i:]

    if best_energy < current_energy:
        return best_chain, best_energy
    return chain, current_energy


def fold(sequence, dimension):
    """Fold an HP sequence into a low-energy conformation on the square lattice."""
    if not sequence:
        return []

    n = len(sequence)
    if n == 1:
        return [[0, 0]]

    rng = random.Random(42)

    best_chain = None
    best_energy = 0

    # Phase 1: Multiple greedy grows
    for attempt in range(30):
        if not _check_deadline():
            break
        chain = _greedy_grow(sequence, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = list(chain)
            best_energy = energy

    # Phase 2: Random walks with regrow-SA
    for attempt in range(20):
        if not _check_deadline():
            break
        chain = _random_walk(sequence, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)

        # Apply regrow-SA
        chain_sa, energy_sa = _regrow_sa(sequence, chain, rng, iterations=1500, t_start=1.0, t_end=0.05)
        if energy_sa < best_energy:
            best_chain = list(chain_sa)
            best_energy = energy_sa
        elif energy < best_energy:
            best_chain = list(chain)
            best_energy = energy

    # Phase 3: Regrow-SA on best solution with more iterations
    if best_chain is not None:
        chain_sa, energy_sa = _regrow_sa(sequence, best_chain, rng, iterations=3000, t_start=0.5, t_end=0.02)
        if energy_sa < best_energy:
            best_chain = list(chain_sa)
            best_energy = energy_sa

    # Phase 4: Local search on best
    if best_chain is not None:
        chain_ls, energy_ls = _local_search(sequence, best_chain, rng, max_iter=2000)
        if energy_ls < best_energy:
            best_chain = list(chain_ls)
            best_energy = energy_ls

    # Phase 5: Final regrow-SA polish
    if best_chain is not None and _check_deadline():
        chain_final, energy_final = _regrow_sa(sequence, best_chain, rng, iterations=2000, t_start=0.3, t_end=0.01)
        if energy_final < best_energy:
            best_chain = list(chain_final)
            best_energy = energy_final

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
