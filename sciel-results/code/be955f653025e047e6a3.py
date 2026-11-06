"""Combined program: algebraic seeds + efficient incremental local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _build_index(n):
    triples = list(combinations(range(n), 3))
    idx = {t: i for i, t in enumerate(triples)}
    return triples, idx

def _fast_is_valid(colors, n):
    triples, idx = _build_index(n)
    c = colors if isinstance(colors, np.ndarray) else np.frombuffer(bytes(colors), dtype=np.uint8)
    for a, b, cc, d in combinations(range(n), 4):
        if (c[idx[(a,b,cc)]] == 0 and c[idx[(a,b,d)]] == 0 and
            c[idx[(a,cc,d)]] == 0 and c[idx[(b,cc,d)]] == 0):
            return False
    for q in combinations(range(n), 5):
        for tri in combinations(q, 3):
            if c[idx[tri]] != 1:
                break
        else:
            return False
    return True

def _local_search(n, seed, time_limit, init=None):
    start = time.time()
    deadline = start + time_limit

    triples, idx = _build_index(n)
    m = len(triples)

    quads = list(combinations(range(n), 4))
    num_quads = len(quads)
    quad_triples = np.array([[idx[(a,b,cc)], idx[(a,b,d)], idx[(a,cc,d)], idx[(b,cc,d)]]
                             for a, b, cc, d in quads], dtype=np.int32)

    quints = list(combinations(range(n), 5))
    num_quints = len(quints)
    quint_triples = np.array([[idx[tri] for tri in combinations(q, 3)] for q in quints], dtype=np.int32)

    tri_quint = [[] for _ in range(m)]
    for qi in range(num_quints):
        for ti in quint_triples[qi]:
            tri_quint[ti].append(qi)
    tri_quad = [[] for _ in range(m)]
    for qi in range(num_quads):
        for ti in quad_triples[qi]:
            tri_quad[ti].append(qi)

    rng = random.Random(seed)
    if init is not None:
        colors = np.array(init, dtype=np.uint8).copy()
    else:
        colors = np.array([rng.choice([0, 1]) for _ in range(m)], dtype=np.uint8)

    quad_red = np.zeros(num_quads, dtype=np.int32)
    for qi in range(num_quads):
        quad_red[qi] = int(np.sum(colors[quad_triples[qi]] == 0))
    red_viol = int(np.sum(quad_red == 4))

    quint_blue = np.zeros(num_quints, dtype=np.int32)
    for qi in range(num_quints):
        quint_blue[qi] = int(np.sum(colors[quint_triples[qi]] == 1))
    blue_viol = int(np.sum(quint_blue == 10))

    total = red_viol + blue_viol
    best_total = total
    best_colors = colors.copy()

    while time.time() < deadline:
        if total == 0:
            break
        if red_viol > 0 and (blue_viol == 0 or rng.random() < red_viol / total):
            bad_quads = np.where(quad_red == 4)[0]
            qi = int(bad_quads[rng.randrange(len(bad_quads))])
            tl = quad_triples[qi]
            red_pos = [p for p in range(4) if colors[tl[p]] == 0]
            ti = int(tl[rng.choice(red_pos)])
        else:
            bad_quints = np.where(quint_blue == 10)[0]
            qi = int(bad_quints[rng.randrange(len(bad_quints))])
            tl = quint_triples[qi]
            blue_pos = [p for p in range(10) if colors[tl[p]] == 1]
            ti = int(tl[rng.choice(blue_pos)])

        new = 1 - colors[ti]
        colors[ti] = new

        for qi in tri_quint[ti]:
            quint_blue[qi] += (1 if new == 1 else -1)
            if quint_blue[qi] == 10:
                blue_viol += 1
            elif quint_blue[qi] == 9:
                blue_viol -= 1

        for qi in tri_quad[ti]:
            quad_red[qi] += (-1 if new == 0 else 1)
            if quad_red[qi] == 4:
                red_viol += 1
            elif quad_red[qi] == 3:
                red_viol -= 1

        total = red_viol + blue_viol
        if total < best_total:
            best_total = total
            best_colors = colors.copy()

    if best_total == 0:
        return best_colors
    return None

def _algebraic_seed(n):
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = np.zeros(len(triples), dtype=np.uint8)
    for a, b, c in triples:
        s = (a % 17 + b % 17 + c % 17) % 17
        colors[idx[(a,b,c)]] = 0 if s == 0 else 1
    return colors

def construct():
    start = time.time()
    deadline = start + 135

    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        seeds = []
        if n == 34:
            alg = _algebraic_seed(n)
            if alg is not None:
                if _fast_is_valid(alg, n):
                    return {'n': n, 'colors': list(alg)}
                seeds.append(('alg', alg))
        for s in range(8):
            seeds.append((s, None))

        remaining = deadline - time.time()
        per = max(0.5, remaining / (len(seeds) + 1))
        for seed_id, init in seeds:
            if time.time() > deadline:
                break
            sid = seed_id if isinstance(seed_id, int) else 0
            res = _local_search(n, sid, per, init=init)
            if res is not None:
                return {'n': n, 'colors': list(res)}

    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
