"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 170.0


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


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


def _random_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(1, n):
        last = chain[-1]
        random.shuffle(dirs)
        for dx, dy in dirs:
            new_site = (last[0] + dx, last[1] + dy)
            if new_site not in occupied:
                chain.append(new_site)
                occupied[new_site] = i
                break
        else:
            # Dead end: try to backtrack one step and try other directions
            if len(chain) > 1:
                chain.pop()
                del occupied[chain[-1]]
                last = chain[-1]
                found = False
                for dx, dy in dirs:
                    new_site = (last[0] + dx, last[1] + dy)
                    if new_site not in occupied:
                        chain.append(new_site)
                        occupied[new_site] = len(chain) - 1
                        found = True
                        break
                if not found:
                    return None
            else:
                return None
    return chain


def _sa_fold(sequence, n, max_time, rng):
    """Simulated annealing folding from a random SAW."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []

    # Generate initial random SAW
    chain = _random_saw(n, rng)
    if chain is None:
        # Fallback: try multiple times
        for _ in range(50):
            chain = _random_saw(n, rng)
            if chain is not None:
                break
        if chain is None:
            return [(i, 0) for i in range(n)]

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)

    T_start = max(2.0, n * 0.15)
    T_end = 0.01
    cooling = 0.995
    T = T_start

    start_time = time.time()
    while time.time() - start_time < max_time:
        move_type = rng.random()

        if move_type < 0.35:
            # Single-residue move
            i = rng.randint(1, n - 2)
            x, y = chain[i]
            candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            rng.shuffle(candidates)
            for new_site in candidates:
                if new_site in occupied and occupied[new_site] != i:
                    continue
                if i > 0:
                    px, py = chain[i-1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue
                if i < n - 1:
                    px, py = chain[i+1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue

                old_site = chain[i]
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

                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    del occupied[old_site]
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break

        elif move_type < 0.65:
            # Kink move: translate pair (i, i+1) perpendicular to their bond
            i = rng.randint(1, n - 3)
            p1, p2 = chain[i], chain[i+1]
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1

            if dx == 1 or dx == -1:
                for sign in (1, -1):
                    new_p1 = (x1, y1 + sign)
                    new_p2 = (x2, y2 + sign)
                    if new_p1 in occupied and occupied[new_p1] not in (i, i+1):
                        continue
                    if new_p2 in occupied and occupied[new_p2] not in (i, i+1):
                        continue
                    if i > 0:
                        px, py = chain[i-1]
                        if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                            continue
                    if i + 2 < n:
                        px, py = chain[i+2]
                        if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                            continue
                    if abs(new_p1[0] - new_p2[0]) + abs(new_p1[1] - new_p2[1]) != 1:
                        continue

                    delta = 0
                    for idx, old_s, new_s in ((i, p1, new_p1), (i+1, p2, new_p2)):
                        if sequence[idx] == "H":
                            for q in _neighbours2d(old_s):
                                j = occupied.get(q)
                                if j is not None and j not in (i, i+1) and sequence[j] == "H":
                                    delta += 1
                            for q in _neighbours2d(new_s):
                                j = occupied.get(q)
                                if j is not None and j not in (i, i+1) and sequence[j] == "H":
                                    delta -= 1

                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        del occupied[p1]
                        del occupied[p2]
                        occupied[new_p1] = i
                        occupied[new_p2] = i + 1
                        chain[i] = new_p1
                        chain[i+1] = new_p2
                        current_energy += delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                        break

            elif dy == 1 or dy == -1:
                for sign in (1, -1):
                    new_p1 = (x1 + sign, y1)
                    new_p2 = (x2 + sign, y2)
                    if new_p1 in occupied and occupied[new_p1] not in (i, i+1):
                        continue
                    if new_p2 in occupied and occupied[new_p2] not in (i, i+1):
                        continue
                    if i > 0:
                        px, py = chain[i-1]
                        if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                            continue
                    if i + 2 < n:
                        px, py = chain[i+2]
                        if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                            continue
                    if abs(new_p1[0] - new_p2[0]) + abs(new_p1[1] - new_p2[1]) != 1:
                        continue

                    delta = 0
                    for idx, old_s, new_s in ((i, p1, new_p1), (i+1, p2, new_p2)):
                        if sequence[idx] == "H":
                            for q in _neighbours2d(old_s):
                                j = occupied.get(q)
                                if j is not None and j not in (i, i+1) and sequence[j] == "H":
                                    delta += 1
                            for q in _neighbours2d(new_s):
                                j = occupied.get(q)
                                if j is not None and j not in (i, i+1) and sequence[j] == "H":
                                    delta -= 1

                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        del occupied[p1]
                        del occupied[p2]
                        occupied[new_p1] = i
                        occupied[new_p2] = i + 1
                        chain[i] = new_p1
                        chain[i+1] = new_p2
                        current_energy += delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                        break

        else:
            # Pivot move: rotate segment around a pivot point
            i = rng.randint(1, n - 4)
            max_k = min(8, n - i - 1)
            k = rng.randint(2, max_k)
            pivot = chain[i]
            seg = chain[i+1:i+1+k]

            for angle in (1, 3):
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
                if i + 1 + k < n:
                    next_pt = chain[i+1+k]
                    if abs(new_seg[-1][0] - next_pt[0]) + abs(new_seg[-1][1] - next_pt[1]) != 1:
                        continue

                seg_indices = set(range(i+1, i+1+k))
                for ns in new_seg:
                    if ns in occupied and occupied[ns] not in seg_indices:
                        valid = False
                        break
                if not valid:
                    continue

                delta = 0
                for idx, old_s, new_s in zip(range(i+1, i+1+k), seg, new_seg):
                    if sequence[idx] == "H":
                        for q in _neighbours2d(old_s):
                            j = occupied.get(q)
                            if j is not None and j not in seg_indices and sequence[j] == "H":
                                delta += 1
                        for q in _neighbours2d(new_s):
                            j = occupied.get(q)
                            if j is not None and j not in seg_indices and sequence[j] == "H":
                                delta -= 1

                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    for old_s in seg:
                        del occupied[old_s]
                    for idx, new_s in zip(range(i+1, i+1+k), seg, new_seg):
                        occupied[new_s] = idx
                        chain[idx] = new_s
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break

        T = max(T_end, T * cooling)

    return best_chain


def _local_polish(sequence, chain, max_iters=2000):
    """Greedy local polish: repeatedly try all single-residue moves until no improvement."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < _DEADLINE:
        improved = False
        iters += 1
        for i in range(1, n - 1):
            x, y = chain[i]
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                new_site = (x+dx, y+dy)
                if new_site in occupied and occupied[new_site] != i:
                    continue
                if i > 0:
                    px, py = chain[i-1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue
                if i < n - 1:
                    px, py = chain[i+1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue

                old_site = chain[i]
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

                if delta < 0:
                    del occupied[old_site]
                    occupied[new_site] = i
                    chain[i] = new_site
                    energy += delta
                    improved = True
                    break
            if improved:
                break
    return chain, energy


def fold(sequence, dimension):
    """Best conformation using simulated annealing from random SAWs."""
    n = len(sequence)
    if n <= 1:
        return [[0, 0]] if n == 1 else []

    best_chain = None
    best_energy = 0  # We want most negative

    # Allocate time per SA run
    remaining = _DEADLINE - time.time()
    num_runs = max(1, int(remaining / 15.0))
    per_run_time = remaining / (num_runs + 1)

    for run in range(num_runs):
        if time.time() >= _DEADLINE - 5:
            break
        rng = random.Random(1000 + run * 7919 + len(sequence))
        sa_time = min(per_run_time, _DEADLINE - time.time() - 5)
        if sa_time < 1.0:
            break
        chain = _sa_fold(sequence, n, sa_time, rng)
        if chain is not None:
            # Polish
            polished, pol_energy = _local_polish(sequence, chain)
            if best_chain is None or pol_energy < best_energy:
                best_chain = polished
                best_energy = pol_energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
