"""Hypergraph Ramsey R(4,5;3) - find coloring of 3-uniform hypergraph on n vertices
with no red K4 and no blue K5. Maximize n."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _triple_index(a, b, c):
    """Compute the lexicographic index of triple (a,b,c) with a<b<c in n vertices."""
    # We need n for this; we'll use a closure or pass n
    pass

def _build_triple_list(n):
    """Build list of all triples in lexicographic order."""
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    return triples

def _build_quad_triples(n, triple_to_idx):
    """For each 4-set, return list of 4 triple indices."""
    quads = list(combinations(range(n), 4))
    quad_triples = []
    for quad in quads:
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quad_triples.append([triple_to_idx[t] for t in ts])
    return quads, quad_triples

def _build_quint_triples(n, triple_to_idx):
    """For each 5-set, return list of 10 triple indices."""
    quints = list(combinations(range(n), 5))
    quint_triples = []
    for quint in quints:
        ts = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quint_triples.append([triple_to_idx[t] for t in ts])
    return quints, quint_triples

def _count_violations(colors, quad_triples, quint_triples):
    """Count total violations: red K4s + blue K5s."""
    red_viol = 0
    for qt in quad_triples:
        if all(colors[i] == 0 for i in qt):
            red_viol += 1
    blue_viol = 0
    for qt in quint_triples:
        if all(colors[i] == 1 for i in qt):
            blue_viol += 1
    return red_viol + blue_viol

def _is_valid(colors, quad_triples, quint_triples):
    """Check if coloring is valid (no red K4, no blue K5)."""
    for qt in quad_triples:
        if all(colors[i] == 0 for i in qt):
            return False
    for qt in quint_triples:
        if all(colors[i] == 1 for i in qt):
            return False
    return True

def _incremental_flip_impact(idx, colors, triple_to_quads, triple_to_quints, quad_triples, quint_triples):
    """
    Compute the change in violation count when flipping triple idx.
    Returns delta = new_viol - old_viol.
    """
    old_color = colors[idx]
    new_color = 1 - old_color
    
    delta = 0
    
    # Check quads containing this triple
    for qi in triple_to_quads[idx]:
        qt = quad_triples[qi]
        # Count how many of the 4 triples in this quad are red (color 0)
        red_count = sum(1 for i in qt if colors[i] == 0)
        
        if old_color == 0:
            # Was red, now blue: red_count decreases by 1
            if red_count == 4:
                # Was a red K4 violation, now it's not
                delta -= 1
            # If red_count was 4, after flip it's 3, so no new violation
        else:
            # Was blue, now red: red_count increases by 1
            if red_count == 3:
                # After flip, red_count becomes 4: new red K4 violation
                delta += 1
    
    # Check quints containing this triple
    for qi in triple_to_quints[idx]:
        qt = quint_triples[qi]
        # Count how many of the 10 triples in this quint are blue (color 1)
        blue_count = sum(1 for i in qt if colors[i] == 1)
        
        if old_color == 1:
            # Was blue, now red: blue_count decreases by 1
            if blue_count == 10:
                # Was a blue K5 violation, now it's not
                delta -= 1
        else:
            # Was red, now blue: blue_count increases by 1
            if blue_count == 9:
                # After flip, blue_count becomes 10: new blue K5 violation
                delta += 1
    
    return delta

def _local_search(n, deadline, seed=42, init_colors=None):
    """
    Local search with incremental violation counting.
    Returns (colors, violations) or (None, None) if deadline exceeded.
    """
    random.seed(seed)
    
    triples = _build_triple_list(n)
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    quads, quad_triples = _build_quad_triples(n, triple_to_idx)
    quints, quint_triples = _build_quint_triples(n, triple_to_idx)
    
    # Build reverse mappings: triple index -> list of quad indices / quint indices
    triple_to_quads = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quad_triples):
        for ti in qt:
            triple_to_quads[ti].append(qi)
    
    triple_to_quints = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quint_triples):
        for ti in qt:
            triple_to_quints[ti].append(qi)
    
    # Initialize colors
    if init_colors is not None:
        colors = init_colors[:]
    else:
        colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Count initial violations
    cur_viol = _count_violations(colors, quad_triples, quint_triples)
    best_colors = colors[:]
    best_viol = cur_viol
    
    if cur_viol == 0:
        return colors, 0
    
    # Simulated annealing parameters
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        if cur_viol == 0:
            break
        
        # Pick a random triple to flip
        idx = random.randint(0, num_triples - 1)
        old_color = colors[idx]
        
        # Compute delta using incremental method
        delta = _incremental_flip_impact(idx, colors, triple_to_quads, triple_to_quints, quad_triples, quint_triples)
        
        new_viol = cur_viol + delta
        if new_viol < 0:
            new_viol = 0
        
        # Accept/reject
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            colors[idx] = 1 - old_color
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors[:]
        # else: reject, colors unchanged
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_colors, best_viol

def _algebraic_init(n, seed=0):
    """
    Generate an initial coloring using algebraic structure.
    Try various group-based colorings.
    """
    triples = _build_triple_list(n)
    colors = []
    
    # Try: color based on sum mod 3
    # For triple (a,b,c): color 0 (red) if (a+b+c) % 3 == 0, else 1 (blue)
    # This is a known heuristic but may not be optimal
    
    # Alternative: use a more sophisticated approach
    # Color based on whether the triple forms a "structured" pattern
    
    # Let's try multiple strategies and pick the best
    best_colors = None
    best_viol = float('inf')
    
    # Strategy 1: sum mod 3
    c1 = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    # Strategy 2: sum mod 3 with offset
    c2 = [0 if (a + b + c + 1) % 3 == 0 else 1 for a, b, c in triples]
    
    # Strategy 3: product-based
    c3 = [0 if ((a * (b + 1) + c) % 3) == 0 else 1 for a, b, c in triples]
    
    # Strategy 4: random with structure
    random.seed(seed)
    c4 = [random.randint(0, 1) for _ in triples]
    
    # Strategy 5: based on differences
    c5 = [0 if ((a - b) * (b - c) + (c - a)) % 2 == 0 else 1 for a, b, c in triples]
    
    candidates = [c1, c2, c3, c4, c5]
    
    # Evaluate each candidate
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    quads, quad_triples = _build_quad_triples(n, triple_to_idx)
    quints, quint_triples = _build_quint_triples(n, triple_to_idx)
    
    for c in candidates:
        v = _count_violations(c, quad_triples, quint_triples)
        if v < best_viol:
            best_viol = v
            best_colors = c
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Start with a small known-good coloring
    # For n=9, the algebraic coloring works
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to find valid colorings for increasing n
    # Use local search with incremental updates
    
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try multiple seeds and initializations
        for seed in [42, 123, 999, 777, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if time.time() > deadline - 2:
                break
            
            # Try algebraic init first
            init = _algebraic_init(n, seed=seed)
            colors, viol = _local_search(n, deadline - 2, seed=seed, init_colors=init)
            
            if colors is not None and viol == 0:
                # Verify
                triples = _build_triple_list(n)
                triple_to_idx = {t: i for i, t in enumerate(triples)}
                quads, quad_triples = _build_quad_triples(n, triple_to_idx)
                quints, quint_triples = _build_quint_triples(n, triple_to_idx)
                if _is_valid(colors, quad_triples, quint_triples):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
        
        if not found and n > 15:
            # If we failed for several consecutive n, stop early
            # But don't stop too early; the known answer is 34
            if n > 20:
                break
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
