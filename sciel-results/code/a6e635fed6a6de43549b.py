"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0  # seconds; return with margin


def _triples(n):
    """List of triples (a,b,c), a<b<c, in lexicographic order."""
    return list(combinations(range(n), 3))


def _triples_to_idx(n):
    """Map (a,b,c) -> index in lexicographic triple list."""
    triples = _triples(n)
    m = len(triples)
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx, m


def _build_quads(n, triples):
    """For each 4-set, list of its 4 triple indices. Returns list of lists."""
    m = len(triples)
    # Precompute triple index via formula for speed
    def t_idx(a, b, c):
        # lexicographic index of (a,b,c) among triples of n
        # count triples (x,y,z) with x<a, or x==a and y<b, or x==a,y==b,z<c
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
        quads.append(idxs)
    return quads


def _build_pents(n, triples):
    """For each 5-set, list of its 10 triple indices."""
    def t_idx(a, b, c):
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt
    pents = []
    for p in combinations(range(n), 5):
        a, b, c, d, e = p
        idxs = []
        for t in combinations(p, 3):
            idxs.append(t_idx(*t))
        pents.append(idxs)
    return pents


def _count_violations(colors, quads, pents):
    """Count red K4 (all 4 red) and blue K5 (all 10 blue)."""
    red_k4 = 0
    for idxs in quads:
        s = 0
        for i in idxs:
            s += colors[i] == 0
        if s == 4:
            red_k4 += 1
    blue_k5 = 0
    for idxs in pents:
        s = 0
        for i in idxs:
            s += colors[i] == 1
        if s == 10:
            blue_k5 += 1
    return red_k4, blue_k5


def _local_search(colors, quads, pents, time_budget):
    """Flip triples to reduce violations. Returns (colors, red_k4, blue_k5)."""
    colors = np.array(colors, dtype=np.int8)
    m = len(colors)
    n_quads = len(quads)
    n_pents = len(pents)
    
    # Build incidence matrices for fast updates
    # quad_incidence[qi, t] = 1 if triple t is in quad qi
    quad_incidence = np.zeros((n_quads, m), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            quad_incidence[qi, i] = 1
    
    pent_incidence = np.zeros((n_pents, m), dtype=np.int8)
    for pi, idxs in enumerate(pents):
        for i in idxs:
            pent_incidence[pi, i] = 1
    
    # Initial counts
    quad_red = np.sum(quad_incidence * (colors == 0), axis=1).astype(np.int8)
    pent_blue = np.sum(pent_incidence * (colors == 1), axis=1).astype(np.int8)
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    total_v = red_k4 + blue_k5
    if total_v == 0:
        return colors.tolist(), red_k4, blue_k5

    t_end = time.time() + time_budget
    order = list(range(m))
    random.shuffle(order)
    improved = True
    while improved and time.time() < t_end:
        improved = False
        for t in order:
            if time.time() >= t_end:
                break
            cur = colors[t]
            new = 1 - cur
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            
            # Get affected quads and pents
            affected_quads = np.where(quad_incidence[:, t] == 1)[0]
            affected_pents = np.where(pent_incidence[:, t] == 1)[0]
            
            dq = 0
            for qi in affected_quads:
                old = quad_red[qi]
                new_q = old + delta_red
                if old == 4:
                    dq -= 1
                if new_q == 4:
                    dq += 1
            
            dp = 0
            for pi in affected_pents:
                old = pent_blue[pi]
                new_p = old + delta_blue
                if old == 10:
                    dp -= 1
                if new_p == 10:
                    dp += 1
            
            if dq + dp < 0:
                colors[t] = new
                for qi in affected_quads:
                    quad_red[qi] += delta_red
                for pi in affected_pents:
                    pent_blue[pi] += delta_blue
                red_k4 += dq
                blue_k5 += dp
                improved = True
        random.shuffle(order)
    return colors.tolist(), red_k4, blue_k5


def _random_valid(n, time_budget):
    """Try to find a valid colouring on n vertices via random init + local search."""
    triples, idx, m = _triples_to_idx(n)
    quads = _build_quads(n, triples)
    pents = _build_pents(n, triples)
    
    t_end = _START + time_budget
    attempts = 0
    while time.time() < t_end:
        attempts += 1
        # Adaptive budget per attempt
        remaining = t_end - time.time()
        if remaining < 0.5:
            break
        # Scale local search time with remaining budget
        ls_time = min(remaining * 0.3, 3.0)
        
        # Random init with varying bias
        p_red = 0.3 + 0.15 * random.random()
        colors = (np.random.rand(m) < p_red).astype(np.int8).tolist()
        colors, r4, b5 = _local_search(colors, quads, pents, ls_time)
        if r4 == 0 and b5 == 0:
            return colors
    return None


def construct():
    deadline = _START + _DEADLINE - 5.0  # safety margin
    best_n = 9
    best_colors = None
    # Seed with the known small valid construction
    from itertools import combinations as _c
    n0 = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in _c(range(n0), 3)]
    best_n = n0

    # Try to grow n from 10 upward, using time budget
    for n in range(10, 41):
        if time.time() >= deadline:
            break
        # Estimate time per n - be more conservative for larger n
        remaining = deadline - time.time()
        if n <= 15:
            budget = min(remaining, 10.0)
        elif n <= 20:
            budget = min(remaining, 15.0)
        elif n <= 25:
            budget = min(remaining, 20.0)
        elif n <= 30:
            budget = min(remaining, 25.0)
        else:
            budget = min(remaining, 30.0)
        if budget < 2.0:
            break
        colors = _random_valid(n, budget)
        if colors is not None:
            best_n = n
            best_colors = colors
        if time.time() >= deadline:
            break

    return {'n': best_n, 'colors': best_colors}
