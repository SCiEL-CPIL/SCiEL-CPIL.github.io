"""Hypergraph Ramsey R(4,5;3): deterministic algebraic seeds + greedy
constraint-propagation / minimum-violation repair (no stochastic hill-climb)."""
import time
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0


def _t_idx(a, b, c, n):
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _build_data(n):
    triples = list(combinations(range(n), 3))
    m = len(triples)

    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    n_quads = len(quads)

    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    n_pents = len(pents)

    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)

    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    return triples, m, quads, pents, tri_quads, tri_pents


def _gf2k_coloring(k, mode="trace"):
    """Deterministic algebraic colouring on n = 2^k vertices using GF(2^k)
    structure. mode: 'trace' (red iff Tr(a+b+c)==0) or 'parity' (red iff
    constant bit of a+b+c is 0)."""
    n = 1 << k
    polys = {
        1: 0b11,
        2: 0b111,
        3: 0b10011,
        4: 0b100011,
        5: 0b100010011,
        6: 0b100001001,
    }
    p = polys.get(k, 0b10011)

    def mul2(x):
        y = x << 1
        if y & (1 << k):
            y ^= p
        return y & ((1 << k) - 1)

    def trace(x):
        t = 0
        cur = x
        for _ in range(k):
            t ^= (cur & 1)
            cur = mul2(cur)
        return t

    colors = []
    for a, b, c in combinations(range(n), 3):
        s = a ^ b ^ c
        if mode == "trace":
            red = (trace(s) == 0)
        else:
            red = ((s & 1) == 0)
        colors.append(0 if red else 1)
    return n, colors


def _mod_coloring(n, mod=3):
    """Deterministic modular colouring: red iff (a+b+c) % mod == 0."""
    colors = []
    for a, b, c in combinations(range(n), 3):
        colors.append(0 if (a + b + c) % mod == 0 else 1)
    return n, colors


def _greedy_repair(colors, quads, pents, tri_quads, tri_pents, time_budget, seed=0):
    """Deterministic greedy repair: repeatedly flip the triple that removes the
    most complete red K4 / blue K5, tie-breaking by minimal induced new
    violations. Random restarts only when fully stuck."""
    rng = np.random.RandomState(seed)
    colors = np.array(colors, dtype=np.int8)
    m = len(colors)
    t_end = time.time() + time_budget

    quad_red = np.zeros(len(quads), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1

    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    energy = red_k4 + blue_k5

    best_energy = energy
    best_colors = colors.copy()
    stall = 0
    MAX_STALL = 300

    while time.time() < t_end:
        if red_k4 == 0 and blue_k5 == 0:
            return colors.tolist(), 0, 0

        # Identify violating structures
        violating_quads = np.where(quad_red == 4)[0]
        violating_pents = np.where(pent_blue == 10)[0]

        if len(violating_quads) == 0 and len(violating_pents) == 0:
            # Energy mismatch; recompute
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            if energy == 0:
                return colors.tolist(), 0, 0

        # Build candidate set from violating structures
        cand_set = set()
        for qi in violating_quads:
            cand_set.update(quads[qi])
        for pi in violating_pents:
            cand_set.update(pents[pi])

        if not cand_set:
            t = rng.randint(m)
        else:
            cand_list = list(cand_set)

            # Greedy: pick triple with best (removals, -induced) score
            best_score = None
            best_t = cand_list[0]
            for t in cand_list:
                cur = colors[t]
                new = 1 - cur
                delta_red = 1 if new == 0 else -1
                delta_blue = 1 if new == 1 else -1

                aq = tri_quads[t]
                ap = tri_pents[t]

                # removals: complete structures that this flip breaks
                removals = 0
                induced = 0
                for qi in aq:
                    old = quad_red[qi]
                    newv = old + delta_red
                    if old == 4 and newv == 3:
                        removals += 1
                    elif old == 3 and newv == 4:
                        induced += 1
                for pi in ap:
                    old = pent_blue[pi]
                    newv = old + delta_blue
                    if old == 10 and newv == 9:
                        removals += 1
                    elif old == 9 and newv == 10:
                        induced += 1

                score = (removals, -induced)
                if best_score is None or score > best_score:
                    best_score = score
                    best_t = t
            t = best_t

        cur = colors[t]
        new = 1 - cur
        delta_red = 1 if new == 0 else -1
        delta_blue = 1 if new == 1 else -1

        aq = tri_quads[t]
        ap = tri_pents[t]

        dq = 0
        if aq:
            old_q = quad_red[aq]
            new_q = old_q + delta_red
            dq = int(np.sum(np.maximum(new_q - 3, 0) - np.maximum(old_q - 3, 0)))
        dp = 0
        if ap:
            old_p = pent_blue[ap]
            new_p = old_p + delta_blue
            dp = int(np.sum(np.maximum(new_p - 9, 0) - np.maximum(old_p - 9, 0)))

        delta_E = dq + dp

        colors[t] = new
        if aq:
            quad_red[aq] += delta_red
        if ap:
            pent_blue[ap] += delta_blue
        energy += delta_E

        if energy < best_energy:
            best_energy = energy
            best_colors = colors.copy()
            stall = 0
        else:
            stall += 1

        # Periodic exact recompute
        if stall % 200 == 0:
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5

        # Stuck: random perturbation (only escape mechanism)
        if stall > MAX_STALL:
            n_flip = rng.randint(3, 8)
            for _ in range(n_flip):
                t2 = rng.randint(m)
                colors[t2] = 1 - colors[t2]
                delta_red = 1 if colors[t2] == 0 else -1
                delta_blue = 1 if colors[t2] == 1 else -1
                aq2 = tri_quads[t2]
                ap2 = tri_pents[t2]
                if aq2:
                    quad_red[aq2] += delta_red
                if ap2:
                    pent_blue[ap2] += delta_blue
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            stall = 0

    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    if red_k4 + blue_k5 < best_energy:
        return colors.tolist(), red_k4, blue_k5
    return best_colors.tolist(), best_energy, best_energy


def _find_valid(n, time_budget, seed=0):
    triples, m, quads, pents, tri_quads, tri_pents = _build_data(n)
    t_end = _START + time_budget

    # Deterministic seeds in order of structural quality
    seeds = []
    k = None
    if n > 1:
        logn = n.bit_length() - 1
        if (1 << logn) == n:
            k = logn
    if k is not None:
        _, c1 = _gf2k_coloring(k, "trace")
        seeds.append(c1)
        _, c2 = _gf2k_coloring(k, "parity")
        seeds.append(c2)
    for mod in (3, 4, 5):
        _, cm = _mod_coloring(n, mod)
        seeds.append(cm)

    # Try each deterministic seed with greedy repair
    for si, s_colors in enumerate(seeds):
        if time.time() >= t_end:
            break
        remaining = t_end - time.time()
        if remaining < 1.5:
            break
        sa_time = min(remaining * 0.35, 8.0)
        colors, r4, b5 = _greedy_repair(
            s_colors, quads, pents, tri_quads, tri_pents,
            sa_time, seed=seed + si)
        if r4 == 0 and b5 == 0:
            return colors

    # Fallback: random init + greedy repair
    while time.time() < t_end:
        remaining = t_end - time.time()
        if remaining < 1.5:
            break
        sa_time = min(remaining * 0.3, 6.0)
        p_red = 0.25 + 0.25 * (seed % 7) / 7.0
        colors = (np.random.rand(m) < p_red).astype(np.int8).tolist()
        colors, r4, b5 = _greedy_repair(
            colors, quads, pents, tri_quads, tri_pents,
            sa_time, seed=seed + 1000)
        if r4 == 0 and b5 == 0:
            return colors
        seed += 1
    return None


def construct():
    deadline = _START + _DEADLINE
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Deterministic algebraic seeds for powers of 2
    for k in range(3, 7):
        n = 1 << k
        if n > 34:
            break
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        colors = _find_valid(n, min(remaining, 25.0), seed=100 + k)
        if colors is not None:
            best_n = n
            best_colors = colors

    # Extend beyond best, up to 34
    for n in range(best_n + 1, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 4.0:
            break
        colors = _find_valid(n, min(remaining, 35.0), seed=200 + n)
        if colors is not None:
            best_n = n
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
