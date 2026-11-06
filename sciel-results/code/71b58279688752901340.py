import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def _triples(n):
    return list(combinations(range(n), 3))

def _build_structures(n):
    """
    Precompute adjacency structures for fast local search.
    Returns:
    - m: number of triples
    - tri_quads: list of lists, tri_quads[i] = list of quad indices containing triple i
    - tri_pents: list of lists, tri_pents[i] = list of pent indices containing triple i
    - quad_red: numpy array, count of red triples in each quad
    - pent_blue: numpy array, count of blue triples in each pent
    """
    triples = _triples(n)
    m = len(triples)
    triples_set = {t: i for i, t in enumerate(triples)}
    
    # Build quads
    quads = []
    for q in combinations(range(n), 4):
        a,b,c,d = q
        idxs = [
            triples_set[(a,b,c)],
            triples_set[(a,b,d)],
            triples_set[(a,c,d)],
            triples_set[(b,c,d)]
        ]
        quads.append(idxs)
        
    # Build pents
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(triples_set[t])
        pents.append(idxs)
        
    # Build reverse maps
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
            
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)
            
    return m, tri_quads, tri_pents, quads, pents

def _sa_local_search(colors, m, tri_quads, tri_pents, quads, pents, time_budget, temp_start=1.0):
    """
    Simulated Annealing local search.
    """
    colors = colors.copy()
    quad_red = np.zeros(len(quads), dtype=np.int16)
    pent_blue = np.zeros(len(pents), dtype=np.int16)
    
    # Initialize counts
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1
                
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    total_v = red_k4 + blue_k5
    
    if total_v == 0:
        return colors, 0, 0

    t_end = time.time() + time_budget
    temp = temp_start
    cooling = 0.99
    
    # Use a list of indices for random selection
    order = list(range(m))
    
    while time.time() < t_end:
        random.shuffle(order)
        for t in order:
            if time.time() >= t_end:
                break
            if temp < 0.01:
                break
                
            cur = colors[t]
            new = 1 - cur
            
            # Calculate delta in violations
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            
            dq = 0
            for qi in tri_quads[t]:
                old = quad_red[qi]
                new_q = old + delta_red
                if old == 4: dq -= 1
                if new_q == 4: dq += 1
                # Also check if it creates a 4? 
                # old=3, new=4 -> +1
                # old=4, new=3 -> -1
                # old=2, new=3 -> 0
                # old=5? impossible.
            dp = 0
            for pi in tri_pents[t]:
                old = pent_blue[pi]
                new_p = old + delta_blue
                if old == 10: dp -= 1
                if new_p == 10: dp += 1
            
            delta_v = dq + dp
            
            if delta_v < 0:
                # Accept
                colors[t] = new
                for qi in tri_quads[t]:
                    quad_red[qi] += delta_red
                for pi in tri_pents[t]:
                    pent_blue[pi] += delta_blue
                red_k4 += dq
                blue_k5 += dp
                total_v += delta_v
            else:
                # Accept with probability exp(-delta_v / temp)
                if delta_v > 0:
                    prob = np.exp(-delta_v / temp)
                    if random.random() < prob:
                        colors[t] = new
                        for qi in tri_quads[t]:
                            quad_red[qi] += delta_red
                        for pi in tri_pents[t]:
                            pent_blue[pi] += delta_blue
                        red_k4 += dq
                        blue_k5 += dp
                        total_v += delta_v
            
        temp *= cooling
        if total_v == 0:
            return colors, 0, 0
            
    return colors, red_k4, blue_k5

def _try_n(n, time_budget):
    """
    Try to find valid coloring for n.
    """
    m, tri_quads, tri_pents, quads, pents = _build_structures(n)
    
    # Multiple restarts
    attempts = 0
    while time.time() < _START + time_budget:
        attempts += 1
        # Random init
        p_red = 0.35
        colors = (np.random.rand(m) < p_red).astype(np.int8)
        
        colors, r4, b5 = _sa_local_search(colors, m, tri_quads, tri_pents, quads, pents, 1.5)
        if r4 == 0 and b5 == 0:
            return colors.tolist()
            
    return None

def construct():
    deadline = _START + _DEADLINE - 2.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to find larger n
    # We know n=34 is possible. 
    # Start from high n and go down? Or low to high?
    # Low to high is safer to ensure validity.
    # But building structures for n=34 is expensive.
    # C(34,3) = 5984. C(34,4) = 46376. C(34,5) = 278256.
    # This is manageable in Python if optimized.
    
    # Try n=34 first with a large budget?
    # If we fail, try 33, 32...
    
    # Actually, let's try n=34 with 60s, n=33 with 30s, etc.
    # But if 34 fails, we might not have time for 33.
    # Better: Try n=34. If found, return.
    # If not, try n=33.
    
    # To maximize score, we want the largest n.
    # Let's try 34.
    
    # Budget allocation
    # Total 145s.
    # n=34: 80s
    # n=33: 30s
    # n=32: 20s
    # n=31: 10s
    # n=30: 5s
    
    targets = [34, 33, 32, 31, 30]
    budgets = [80, 30, 20, 10, 5]
    
    for n, budget in zip(targets, budgets):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < budget:
            budget = remaining - 1.0
        if budget < 2.0:
            break
            
        colors = _try_n(n, budget)
        if colors is not None:
            best_n = n
            best_colors = colors
            break # Found valid, stop.
            
    return {'n': best_n, 'colors': best_colors}
