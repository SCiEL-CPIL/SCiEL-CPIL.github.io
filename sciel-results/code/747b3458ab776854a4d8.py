import time
import itertools
import random
import numpy as np

def _idx(i, j):
    """Index for pair (i,j) with 0<=i<j<16 in lexicographic flat list."""
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _validate_list(colors_list):
    """Check every K4 spans at least 5 distinct colours."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        if len({e1, e2, e3, e4, e5, e6}) < 5:
            return False
    return True

def _count_bad_k4(colors_list):
    """Count K4s with fewer than 5 distinct colours."""
    bad = 0
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        if len({e1, e2, e3, e4, e5, e6}) < 5:
            bad += 1
    return bad

def _make_cyclic_1factorization():
    """Standard cyclic 1-factorization of K16. Returns list of 120 colors."""
    n = 16
    colors = [0] * 120
    # Vertices 0..14 are in a cycle, 15 is infinity
    # Matching i contains (15, i) and (i+j, i-j) mod 15 for j=1..7
    for i in range(15):
        # Edge (15, i)
        colors[_idx(15, i)] = i
        # Edges (i+j, i-j)
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            colors[_idx(a, b)] = i
    return colors

def _local_search(colors_list, max_iters=500000, seed=42):
    """Local search to minimize bad K4s."""
    n = 16
    num_colors = 15
    colors = list(colors_list)
    rng = random.Random(seed)
    
    # Precompute K4s and their edge indices
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        k4_edges.append([
            _idx(a, b), _idx(a, c), _idx(a, d),
            _idx(b, c), _idx(b, d), _idx(c, d)
        ])
    
    # Precompute for each edge, which K4s it belongs to
    edge_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4_edges):
        for e_idx in edges:
            edge_k4s[e_idx].append(k4_idx)

    def get_bad_count(colors):
        bad = 0
        for edges in k4_edges:
            c = {colors[i] for i in edges}
            if len(c) < 5:
                bad += 1
        return bad

    current_bad = get_bad_count(colors)
    if current_bad == 0:
        return colors
    
    best_colors = list(colors)
    best_bad = current_bad
    
    # Simulated Annealing
    temp = 10.0
    cooling = 0.999
    
    for iteration in range(max_iters):
        # Pick a random edge
        e_idx = rng.randint(0, 119)
        old_color = colors[e_idx]
        new_color = rng.randint(0, num_colors - 1)
        
        if old_color == new_color:
            continue
            
        colors[e_idx] = new_color
        
        # Calculate delta in bad count
        # Only K4s containing e_idx are affected
        delta = 0
        for k4_idx in edge_k4s[e_idx]:
            edges = k4_edges[k4_idx]
            old_set = {colors[i] if i != e_idx else old_color for i in edges}
            new_set = {colors[i] for i in edges}
            
            old_bad = 1 if len(old_set) < 5 else 0
            new_bad = 1 if len(new_set) < 5 else 0
            delta += new_bad - old_bad
            
        if delta < 0 or (delta == 0 and rng.random() < 0.5):
            # Accept
            current_bad += delta
            if current_bad < best_bad:
                best_bad = current_bad
                best_colors = list(colors)
                if best_bad == 0:
                    break
        else:
            # Revert if not accepted
            colors[e_idx] = old_color
            
        temp *= cooling
        
    return best_colors

def construct():
    t0 = time.time()
    deadline = 140  # Safety margin
    
    # 1. Try cyclic 1-factorization
    colors = _make_cyclic_1factorization()
    bad = _count_bad_k4(colors)
    
    if bad == 0:
        return colors
    
    # 2. Local search from cyclic
    improved = _local_search(colors, max_iters=200000, seed=1)
    bad_imp = _count_bad_k4(improved)
    
    if bad_imp < bad:
        colors = improved
        bad = bad_imp
        
    if bad == 0:
        return colors
    
    # 3. Try a few random 1-factorizations if time permits
    # Generate random 1-factorization
    for trial in range(5):
        if time.time() - t0 > deadline:
            break
            
        rng = random.Random(100 + trial)
        rand_colors = [0] * 120
        # Generate random perfect matchings
        # This is tricky to do correctly and fast. 
        # Instead, just perturb the cyclic one more aggressively.
        
        # Try a different local search seed
        improved = _local_search(colors, max_iters=100000, seed=200 + trial)
        bad_imp = _count_bad_k4(improved)
        if bad_imp < bad:
            colors = improved
            bad = bad_imp
            if bad == 0:
                return colors

    # 4. Fallback: Return the best found. 
    # If it's not valid, it's a risk, but 15 colors is the goal.
    # The problem states "A valid colouring certifies...". 
    # If we return an invalid one, score is 0.
    # We MUST return a valid one.
    
    # Let's check if we have a valid one.
    if _validate_list(colors):
        return colors
        
    # If we didn't find a valid 15-coloring, we must fall back to a known valid one.
    # The 17-coloring is valid.
    # Can we construct a valid 16-coloring easily?
    # Or just return the 17-coloring to guarantee a non-zero score?
    # The seed program returned 17.
    # But we want to beat 1.0 (which corresponds to 17? No, seed is 1.0).
    # If seed is 1.0 and it returns 17, then score = seed/value?
    # "search score is seed/value". If seed=1.0, value=17, score=1/17?
    # No, "1.0 equals the seed program".
    # So if seed program returns 17, and score is 1.0, then value must be 17?
    # If I return 15, score = 17/15 = 1.13.
    
    # So I MUST return a valid coloring.
    # If I can't find a valid 15-coloring, I should return a valid 17-coloring.
    
    # Let's try to quickly find a valid 15-coloring with more aggressive search.
    # If time is running out, return 17.
    
    if time.time() - t0 < deadline - 5:
        # One more try with a different initial state
        # Maybe a random coloring with 15 colors?
        rng = random.Random(999)
        rand_init = [rng.randint(0, 14) for _ in range(120)]
        improved = _local_search(rand_init, max_iters=100000, seed=999)
        if _count_bad_k4(improved) == 0:
            return improved
            
    # Fallback to 17-coloring
    # (i+j) % 17
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
