"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours2d(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied and q != last:
                    free += 1
            score = gain * 2.5 + 0.15 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_single(sequence, chain, occupied, i, new_site):
    """Compute energy delta for moving residue i to new_site. Returns delta or None if invalid."""
    old_site = chain[i]
    if new_site == old_site:
        return 0
    if new_site in occupied and occupied[new_site] != i:
        return None
    # Check adjacency to neighbors
    if i > 0:
        px, py = chain[i - 1]
        if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
            return None
    if i < len(chain) - 1:
        px, py = chain[i + 1]
        if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
            return None
    
    delta = 0
    if sequence[i] == "H":
        for q in _neighbours2d(old_site):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == "H":
                delta += 1
        for q in _neighbours2d(new_site):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == "H":
                delta -= 1
    return delta


def _apply_single(chain, occupied, i, new_site):
    old_site = chain[i]
    del occupied[old_site]
    occupied[new_site] = i
    chain[i] = new_site


def _local_optimize(sequence, chain, occupied, max_iters=5000):
    """Local optimization with single-residue, kink, and pivot moves using SA."""
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)

    best_energy = _energy(sequence, chain)
    current_energy = best_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)

    T = max(0.8, n * 0.05)
    T_min = 0.01
    cooling = 0.997

    iters = 0
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        move_type = random.random()

        if move_type < 0.45:
            # Single-residue move
            i = random.randint(1, n - 2)
            x, y = chain[i]
            candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            random.shuffle(candidates)
            for new_site in candidates:
                delta = _delta_single(sequence, chain, occupied, i, new_site)
                if delta is None:
                    continue
                if delta < 0 or random.random() < math.exp(-delta / T):
                    _apply_single(chain, occupied, i, new_site)
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_occupied = dict(occupied)
                    break

        elif move_type < 0.75:
            # Kink move: rotate consecutive pair (i, i+1)
            i = random.randint(1, n - 3)
            p1, p2 = chain[i], chain[i + 1]
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1

            if dx == 1 or dx == -1:
                for sign in (1, -1):
                    new_p1 = (x1, y1 + sign)
                    new_p2 = (x2, y2 + sign)
                    if new_p1 in occupied and occupied[new_p1] not in (i, i + 1):
                        continue
                    if new_p2 in occupied and occupied[new_p2] not in (i, i + 1):
                        continue
                    if i > 0:
                        px, py = chain[i - 1]
                        if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                            continue
                    if i + 2 < n:
                        px, py = chain[i + 2]
                        if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                            continue
                    if abs(new_p1[0] - new_p2[0]) + abs(new_p1[1] - new_p2[1]) != 1:
                        continue

                    delta = 0
                    for idx, old_s, new_s in ((i, p1, new_p1), (i + 1, p2, new_p2)):
                        if sequence[idx] == "H":
                            for q in _neighbours2d(old_s):
                                j = occupied.get(q)
                                if j is not None and j not in (i, i + 1) and sequence[j] == "H":
                                    delta += 1
                            for q in _neighbours2d(new_s):
                                j = occupied.get(q)
                                if j is not None and j not in (i, i + 1) and sequence[j] == "H":
                                    delta -= 1

                    if delta < 0 or random.random() < math.exp(-delta / T):
                        del occupied[p1]
                        del occupied[p2]
                        occupied[new_p1] = i
                        occupied[new_p2] = i + 1
                        chain[i] = new_p1
                        chain[i + 1] = new_p2
                        current_energy += delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            best_occupied = dict(occupied)
                        break

        else:
            # Pivot move: rotate segment (i+1..i+k) around chain[i]
            i = random.randint(1, n - 4)
            k = random.randint(2, min(6, n - i - 1))
            pivot = chain[i]
            seg = chain[i + 1:i + 1 + k]

            for angle in (1, 3):  # 90 or 270 degrees
                new_seg = []
                valid = True
                for (sx, sy) in seg:
                    if angle == 1:  # 90 deg CCW
                        nx = pivot[0] - (sy - pivot[1])
                        ny = pivot[1] + (sx - pivot[0])
                    else:  # 270 deg CCW
                        nx = pivot[0] + (sy - pivot[1])
                        ny = pivot[1] - (sx - pivot[0])
                    new_seg.append((nx, ny))

                if abs(new_seg[0][0] - pivot[0]) + abs(new_seg[0][1] - pivot[1]) != 1:
                    continue

                # Check self-avoidance
                conflict = False
                for idx, ns in enumerate(new_seg):
                    j = i + 1 + idx
                    if ns in occupied and occupied[ns] not in range(i + 1, i + 1 + k):
                        conflict = True
                        break
                if conflict:
                    continue

                # Check adjacency after the segment
                if i + 1 + k < n:
                    px, py = chain[i + 1 + k]
                    last_new = new_seg[-1]
                    if abs(last_new[0] - px) + abs(last_new[1] - py) != 1:
                        continue

                # Compute delta
                delta = 0
                for idx in range(k):
                    j = i + 1 + idx
                    old_s = seg[idx]
                    new_s = new_seg[idx]
                    if sequence[j] == "H":
                        for q in _neighbours2d(old_s):
                            oj = occupied.get(q)
                            if oj is not None and oj != j and oj not in range(i + 1, i + 1 + k) and sequence[oj] == "H":
                                delta += 1
                        for q in _neighbours2d(new_s):
                            oj = occupied.get(q)
                            if oj is not None and oj != j and oj not in range(i + 1, i + 1 + k) and sequence[oj] == "H":
                                delta -= 1
                        # Also check contacts between new positions of segment residues
                        for q in _neighbours2d(new_s):
                            oj = occupied.get(q)
                            if oj is not None and oj not in range(i + 1, i + 1 + k) and sequence[oj] == "H":
                                pass  # already counted above

                # Also need to check new contacts within the moved segment
                for a in range(k):
                    if sequence[i + 1 + a] == "H":
                        for b in range(a + 1, k):
                            if sequence[i + 1 + b] == "H":
                                if abs(new_seg[a][0] - new_seg[b][0]) + abs(new_seg[a][1] - new_seg[b][1]) == 1:
                                    if abs(new_seg[a][0] - seg[b][0]) + abs(new_seg[a][1] - seg[b][1]) != 1:
                                        delta -= 1

                if delta < 0 or random.random() < math.exp(-delta / T):
                    for idx in range(k):
                        del occupied[seg[idx]]
                    for idx in range(k):
                        occupied[new_seg[idx]] = i + 1 + idx
                    for idx in range(k):
                        chain[i + 1 + idx] = new_seg[idx]
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_occupied = dict(occupied)
                    break

        T = max(T_min, T * cooling)

    return best_chain, best_occupied, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    restarts = 0
    no_improve_count = 0
    while time.time() < _DEADLINE:
        restarts += 1
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        occupied = {p: i for i, p in enumerate(chain)}
        chain, occupied, value = _local_optimize(sequence, chain, occupied)
        if value < best_energy:
            best, best_energy = chain, value
            no_improve_count = 0
        else:
            no_improve_count += 1
        if restarts > 120 or no_improve_count > 30:
            break
    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
