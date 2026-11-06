"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0

def _t_idx(a, b, c, n):
    """Lexicographic index of triple (a,b,c) among triples of [0..n-1]."""
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt

def _build_quads(n):
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        quads.append([_t_idx(a, b, c, n), _t_idx(a, b, d, n), _t_idx(a, c, d, n), _t_idx(b, c, d, n)])
    return quads

def _build_pents(n):
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    return pents

def _count_violations(colors, quads, pents):
    red_k4 = 0
    for idxs in quads:
        if all(colors[i] == 0 for i in idxs):
            red_k4 += 1
    blue_k5 = 0
    for idxs in pents:
        if all(colors[i] == 1 for i in idxs):
            blue_k5 += 1
    return red_k4, blue_k5

def _local_search(colors, quads, pents, time_budget):
    """Local search to reduce violations. Returns (colors, red_k4, blue_k5)."""
    colors = colors.copy()
    m = len(colors)
    
    # Precompute counts
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
    
    if red_k4 == 0 and blue_k5 == 0:
        return colors, 0, 0
    
    # Precompute adjacency
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)
    
    t_end = time.time() + time_budget
    order = list(range(m))
    
    while time.time() < t_end:
        improved = False
        random.shuffle(order)
        for t in order:
            if time.time() >= t_end:
                break
            cur = colors[t]
            new = 1 - cur
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            
            dq = 0
            for qi in tri_quads[t]:
                old = quad_red[qi]
                new_q = old + delta_red
                if old == 4:
                    dq -= 1
                if new_q == 4:
                    dq += 1
            
            dp = 0
            for pi in tri_pents[t]:
                old = pent_blue[pi]
                new_p = old + delta_blue
                if old == 10:
                    dp -= 1
                if new_p == 10:
                    dp += 1
            
            if dq + dp < 0:
                colors[t] = new
                for qi in tri_quads[t]:
                    quad_red[qi] += delta_red
                for pi in tri_pents[t]:
                    pent_blue[pi] += delta_blue
                red_k4 += dq
                blue_k5 += dp
                improved = True
        
        if not improved:
            # Try random perturbations if stuck
            if time.time() < t_end - 0.5:
                for _ in range(20):
                    t = random.randrange(m)
                    colors[t] = 1 - colors[t]
                    cur = colors[t]
                    new = 1 - cur
                    delta_red = 1 if new == 0 else -1
                    delta_blue = 1 if new == 1 else -1
                    dq = 0
                    for qi in tri_quads[t]:
                        old = quad_red[qi]
                        new_q = old + delta_red
                        if old == 4:
                            dq -= 1
                        if new_q == 4:
                            dq += 1
                    dp = 0
                    for pi in tri_pents[t]:
                        old = pent_blue[pi]
                        new_p = old + delta_blue
                        if old == 10:
                            dp -= 1
                        if new_p == 10:
                            dp += 1
                    for qi in tri_quads[t]:
                        quad_red[qi] += delta_red
                    for pi in tri_pents[t]:
                        pent_blue[pi] += delta_blue
                    red_k4 += dq
                    blue_k5 += dp
                break
    
    return colors, red_k4, blue_k5

def construct():
    """
    Returns a valid coloring for n vertices.
    Strategy:
    1. Try a known small valid coloring for n=9 as baseline.
    2. Incrementally grow from n=9 to n=40, using local search to fix violations.
    3. For each n, try multiple random restarts.
    """
    
    # Start with a valid coloring for n=9
    # A simple valid coloring: color (a,b,c) red if (a+b+c) % 3 == 0, else blue.
    # Let's verify this is valid for n=9.
    def _initial_color(n):
        colors = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if (a + b + c) % 3 == 0:
                        colors.append(0)
                    else:
                        colors.append(1)
        return colors
    
    def _verify(n, colors):
        quads = _build_quads(n)
        pents = _build_pents(n)
        r4, b5 = _count_violations(colors, quads, pents)
        return r4 == 0 and b5 == 0
    
    # Verify n=9 baseline
    colors9 = _initial_color(9)
    assert _verify(9, colors9), "n=9 baseline should be valid"
    
    best_n = 9
    best_colors = colors9[:]
    
    # Try to grow from 9 to 40
    for n in range(10, 41):
        deadline = _START + _DEADLINE - 5.0
        if time.time() >= deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # For each n, try several random initializations
        found = False
        max_attempts = max(3, int(remaining / 2))
        
        for attempt in range(max_attempts):
            if time.time() >= deadline:
                break
            
            if attempt == 0:
                # Use the best coloring from n-1 extended randomly
                if n - 1 >= 10:
                    # Extend best_colors from n-1 to n
                    # This is complex; just use random for simplicity
                    m = len(list(combinations(range(n), 3)))
                    colors = (np.random.rand(m) < 0.4).astype(np.int8).tolist()
                else:
                    colors = _initial_color(n)
            else:
                m = len(list(combinations(range(n), 3)))
                # Vary the red probability
                p_red = 0.3 + 0.1 * (attempt % 4)
                colors = (np.random.rand(m) < p_red).astype(np.int8).tolist()
            
            quads = _build_quads(n)
            pents = _build_pents(n)
            
            # Local search
            time_budget = min(remaining - 2.0, 20.0)
            if time_budget < 2.0:
                break
                
            colors, r4, b5 = _local_search(colors, quads, pents, time_budget)
            
            if r4 == 0 and b5 == 0:
                best_n = n
                best_colors = colors
                found = True
                break
            else:
                # Keep trying with different random seeds
                pass
        
        if not found:
            # Try one more time with a longer local search
            remaining = deadline - time.time()
            if remaining > 10.0:
                m = len(list(combinations(range(n), 3)))
                colors = (np.random.rand(m) < 0.4).astype(np.int8).tolist()
                quads = _build_quads(n)
                pents = _build_pents(n)
                colors, r4, b5 = _local_search(colors, quads, pents, remaining - 3.0)
                if r4 == 0 and b5 == 0:
                    best_n = n
                    best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}
