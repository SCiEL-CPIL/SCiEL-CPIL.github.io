"""HP lattice protein folding using re-insertion and 2-opt with simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = None
_BUDGET_FRACTION = 0.85

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    n = len(point)
    for axis in range(n):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 + free * 0.05 + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        options.sort(reverse=True)
        k = min(3, len(options))
        chosen = rng.choice(options[:k])
        site = chosen[1]
        chain.append(site)
        occupied[site] = i
    return chain

def _local_search(sequence, chain, dimension, rng, max_iters=3000):
    n = len(sequence)
    if n <= 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    pos_map = {p: i for i, p in enumerate(current_chain)}

    T_start = 2.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start

    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        T *= cooling

        move_type = rng.random()
        accepted = False

        if move_type < 0.4:
            # Single-residue re-insertion: remove residue i, insert at position j
            i = rng.randint(0, n - 1)
            p_i = current_chain[i]
            del current_chain[i]
            del pos_map[p_i]
            # Try inserting at a random position
            j = rng.randint(0, len(current_chain))
            # Find valid sites
            candidates = set()
            if j > 0:
                candidates.update(_neighbours(current_chain[j - 1]))
            if j < len(current_chain):
                candidates.update(_neighbours(current_chain[j]))

            valid_site = None
            for site in candidates:
                if site in pos_map:
                    continue
                valid = True
                if j > 0:
                    prev = current_chain[j - 1]
                    if abs(site[0] - prev[0]) + abs(site[1] - prev[1]) + abs(site[2] - prev[2]) != 1:
                        valid = False
                if valid and j < len(current_chain):
                    nxt = current_chain[j]
                    if abs(site[0] - nxt[0]) + abs(site[1] - nxt[1]) + abs(site[2] - nxt[2]) != 1:
                        valid = False
                if valid:
                    valid_site = site
                    break

            if valid_site is not None:
                current_chain.insert(j, valid_site)
                pos_map = {p: idx for idx, p in enumerate(current_chain)}
                new_energy = _energy(sequence, current_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                    current_energy = new_energy
                    accepted = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                else:
                    # Revert
                    current_chain.pop(j)
                    del pos_map[valid_site]
                    pos_map = {p: idx for idx, p in enumerate(current_chain)}
            else:
                # Revert
                current_chain.insert(i, p_i)
                pos_map = {p: idx for idx, p in enumerate(current_chain)}

        elif move_type < 0.7:
            # 2-opt: reverse a segment [i+1, j]
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            old_seg = current_chain[i + 1:j + 1]
            new_seg = old_seg[::-1]
            current_chain[i + 1:j + 1] = new_seg
            pos_map = {p: idx for idx, p in enumerate(current_chain)}
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_energy = new_energy
                accepted = True
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
            else:
                current_chain[i + 1:j + 1] = old_seg
                pos_map = {p: idx for idx, p in enumerate(current_chain)}

        else:
            # Single move: move residue idx to a neighboring site
            idx = rng.randint(1, n - 2)
            p = current_chain[idx]
            moved = False
            for site in _neighbours(p):
                if site == current_chain[idx - 1] or site == current_chain[idx + 1]:
                    continue
                if site in pos_map:
                    continue
                prev = current_chain[idx - 1]
                nxt = current_chain[idx + 1]
                if abs(site[0] - prev[0]) + abs(site[1] - prev[1]) + abs(site[2] - prev[2]) != 1:
                    continue
                if abs(site[0] - nxt[0]) + abs(site[1] - nxt[1]) + abs(site[2] - nxt[2]) != 1:
                    continue
                # Try move
                del pos_map[p]
                current_chain[idx] = site
                pos_map[site] = idx
                new_energy = _energy(sequence, current_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                    current_energy = new_energy
                    moved = True
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                    break
                else:
                    current_chain[idx] = p
                    del pos_map[site]
                    pos_map[p] = idx

    return best_chain

def fold(sequence, dimension):
    rng = random.Random(42 + len(sequence))
    best, best_energy = None, 0

    num_restarts = 25
    for restart in range(num_restarts):
        if _is_deadline_exceeded():
            break

        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue

        chain = _local_search(sequence, chain, dimension, rng, max_iters=2000)

        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
