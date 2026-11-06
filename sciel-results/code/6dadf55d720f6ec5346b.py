"""HP lattice protein folding: beam search growth + pivot local refinement."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 315.0
_T0 = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    pos_map = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_map.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_grow(sequence, dimension, rng, beam_width=40):
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]

    start = (0,) * dimension
    occupied0 = {start}
    beam = [(0.0, [start], occupied0)]

    for i in range(1, n):
        if _time_left() < 2.0:
            break
        new_beam = []
        for score, chain, occ in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occ:
                            for j in range(len(chain)):
                                if chain[j] == q:
                                    if j < i - 1 and sequence[j] == 'H':
                                        gain += 1
                                    break
                free = 0
                for q in _neighbours(site):
                    if q not in occ:
                        free += 1
                new_score = -gain + 0.05 * free + rng.random() * 0.3
                new_beam.append((new_score, chain + [site], occ | {site}))

        if not new_beam:
            return None

        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]

    if not beam:
        return None
    return max(beam, key=lambda x: x[0])[1]


def _pivot_move(chain, i, axis, direction, dim):
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None

    old_pos = chain[i]
    new_pos = list(old_pos)
    new_pos[axis] += direction
    new_pos = tuple(new_pos)

    occupied_set = set(chain)
    if new_pos in occupied_set:
        return None

    delta = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1], new_pos[2] - old_pos[2])

    new_chain = list(chain)
    for j in range(i, n):
        p = chain[j]
        new_chain[j] = (p[0] + delta[0], p[1] + delta[1], p[2] + delta[2])

    new_occ = set()
    for j in range(i, n):
        if new_chain[j] in new_occ:
            return None
        new_occ.add(new_chain[j])

    for j in range(i):
        if new_chain[j] in new_occ:
            return None

    prev = chain[i - 1]
    cur = new_chain[i]
    dist = abs(prev[0] - cur[0]) + abs(prev[1] - cur[1]) + abs(prev[2] - cur[2])
    if dist != 1:
        return None

    return new_chain


def _local_refine(sequence, chain, dimension, rng, max_iter=200):
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, chain)

    for _ in range(max_iter):
        if _time_left() < 2.0:
            break

        i = rng.randint(1, n - 2)
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])

        result = _pivot_move(chain, i, axis, direction, dimension)
        if result is None:
            continue

        new_energy = _energy(sequence, result)
        delta_e = new_energy - best_energy

        if delta_e < 0:
            chain = result
            best_chain = list(chain)
            best_energy = new_energy
        elif delta_e == 0 and rng.random() < 0.05:
            chain = result

    return best_chain


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]

    rng = random.Random(42 + len(sequence))

    if n <= 48:
        beam_width = 60
        n_restarts = 8
        refine_iter = 300
    elif n <= 80:
        beam_width = 40
        n_restarts = 5
        refine_iter = 200
    else:
        beam_width = 30
        n_restarts = 3
        refine_iter = 150

    best_chain = None
    best_energy = 0

    for restart in range(n_restarts):
        if _time_left() < 5.0:
            break

        chain = _beam_grow(sequence, dimension, rng, beam_width=beam_width)
        if chain is None or len(chain) != n:
            continue

        refined = _local_refine(sequence, chain, dimension, rng, max_iter=refine_iter)
        if refined is not None and len(refined) == n:
            chain = refined

        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
