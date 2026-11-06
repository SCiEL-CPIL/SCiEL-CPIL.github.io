"""Rewritten program for hypergraph Ramsey R(4,5;3) construction."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct():
    """
    Construct a 2-coloring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Uses a combination of algebraic initialization and local search with incremental violation counting.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline (10s safety margin)
    
    best_n = 0
    best_colors = None
    
    # Try to find a valid coloring for n=34 first (the target)
    for target_n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        colors = _find_coloring(target_n, deadline)
        if colors is not None:
            if target_n > best_n:
                best_n = target_n
                best_colors = colors
            break
    
    if best_colors is None:
        # Fallback: simple valid construction for small n
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
    
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    """Find a valid 2-coloring of C(n,3) triples with no red K4^3 and no blue K5^3."""
    triples = list(itertools.combinations(range(n), 3))
    m = len(triples)
    
    # Build adjacency structures for fast violation counting
    # For each triple, find all K4^3 (quads) containing it and all K5^3 (quints) containing it
    # A K4^3 is a set of 4 vertices; it contains 4 triples
    # A K5^3 is a set of 5 vertices; it contains 10 triples
    
    # Map each triple to its index
    triple_idx = {}
    for i, t in enumerate(triples):
        triple_idx[t] = i
    
    # For each triple, list of (quad, triple_index_in_quad) - but we need to track violations differently
    # A red K4^3 violation: all 4 triples in a quad are red
    # A blue K5^3 violation: all 10 triples in a quint are blue
    
    # Precompute for each triple which quads contain it
    # And for each triple which quints contain it
    
    # For efficiency, let's use a different approach:
    # Count violations incrementally when flipping a triple
    
    # Precompute: for each triple t, the list of quads that contain t
    # A quad contains t if t is a 3-subset of the quad
    # For triple (a,b,c), the quads containing it are {a,b,c,x} for x not in {a,b,c}
    
    # Similarly for quints: (a,b,c) is in quints {a,b,c,x,y} for x,y not in {a,b,c}, x<y
    
    # This is expensive to precompute fully, so let's use a smarter approach
    
    # Initialize with an algebraic heuristic
    colors = _algebraic_init(n, triples)
    
    # Count initial violations
    red_violations = _count_red_k4(n, colors, triple_idx)
    blue_violations = _count_blue_k5(n, colors, triple_idx)
    
    if red_violations == 0 and blue_violations == 0:
        return colors
    
    # Local search: flip triples to reduce violations
    # Use simulated annealing or greedy approach
    
    # Greedy phase: repeatedly flip the triple that reduces violations the most
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        if red_violations == 0 and blue_violations == 0:
            break
        
        best_flip = -1
        best_delta = 0
        
        # Sample a subset of triples to consider (for speed)
        sample_size = min(m, 200)
        candidates = random.sample(range(m), sample_size)
        
        for i in candidates:
            # Try flipping triple i
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Compute change in violations
            delta_red, delta_blue = _flip_delta(n, i, old_color, colors, triple_idx)
            
            total_delta = delta_red + delta_blue
            if total_delta < best_delta:
                best_delta = total_delta
                best_flip = i
        
        if best_flip >= 0 and best_delta < 0:
            old_color = colors[best_flip]
            colors[best_flip] = 1 - old_color
            delta_red, delta_blue = _flip_delta(n, best_flip, old_color, colors, triple_idx)
            red_violations += delta_red
            blue_violations += delta_blue
        else:
            # No improvement found; try random flips with simulated annealing
            if red_violations + blue_violations > 100:
                # Accept worse moves occasionally
                i = random.randint(0, m - 1)
                old_color = colors[i]
                colors[i] = 1 - old_color
                delta_red, delta_blue = _flip_delta(n, i, old_color, colors, triple_idx)
                red_violations += delta_red
                blue_violations += delta_blue
    
    # Check final validity
    if _is_valid(n, colors, triple_idx):
        return colors
    
    # If local search didn't work, try a different initialization
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        colors = _random_init(n, triples)
        red_violations = _count_red_k4(n, colors, triple_idx)
        blue_violations = _count_blue_k5(n, colors, triple_idx)
        
        if red_violations == 0 and blue_violations == 0:
            return colors
        
        # Local search
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            if red_violations == 0 and blue_violations == 0:
                break
            
            best_flip = -1
            best_delta = 0
            
            sample_size = min(m, 200)
            candidates = random.sample(range(m), sample_size)
            
            for i in candidates:
                old_color = colors[i]
                delta_red, delta_blue = _flip_delta(n, i, old_color, colors, triple_idx)
                total_delta = delta_red + delta_blue
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_flip = i
            
            if best_flip >= 0 and best_delta < 0:
                old_color = colors[best_flip]
                colors[best_flip] = 1 - old_color
                delta_red, delta_blue = _flip_delta(n, best_flip, old_color, colors, triple_idx)
                red_violations += delta_red
                blue_violations += delta_blue
            else:
                i = random.randint(0, m - 1)
                old_color = colors[i]
                colors[i] = 1 - old_color
                delta_red, delta_blue = _flip_delta(n, i, old_color, colors, triple_idx)
                red_violations += delta_red
                blue_violations += delta_blue
        
        if _is_valid(n, colors, triple_idx):
            return colors
    
    return None

def _algebraic_init(n, triples):
    """Initialize colors using an algebraic heuristic."""
    colors = []
    for a, b, c in triples:
        # Use a combination of modular conditions
        s = a + b + c
        p = a * b + b * c + c * a
        if s % 2 == 0 and p % 3 == 0:
            colors.append(0)
        else:
            colors.append(1)
    return colors

def _random_init(n, triples):
    """Initialize colors randomly."""
    return [random.randint(0, 1) for _ in range(len(triples))]

def _count_red_k4(n, colors, triple_idx):
    """Count number of red K4^3 violations."""
    count = 0
    for quad in itertools.combinations(range(n), 4):
        all_red = True
        for triple in itertools.combinations(quad, 3):
            idx = triple_idx[triple]
            if colors[idx] != 0:
                all_red = False
                break
        if all_red:
            count += 1
    return count

def _count_blue_k5(n, colors, triple_idx):
    """Count number of blue K5^3 violations."""
    count = 0
    for quint in itertools.combinations(range(n), 5):
        all_blue = True
        for triple in itertools.combinations(quint, 3):
            idx = triple_idx[triple]
            if colors[idx] != 1:
                all_blue = False
                break
        if all_blue:
            count += 1
    return count

def _flip_delta(n, triple_i, old_color, colors, triple_idx):
    """
    Compute the change in violation counts when flipping triple_i.
    Returns (delta_red_violations, delta_blue_violations).
    """
    # Get the triple
    triple = None
    for i, t in enumerate(itertools.combinations(range(n), 3)):
        if i == triple_i:
            triple = t
            break
    
    a, b, c = triple
    new_color = 1 - old_color
    
    delta_red = 0
    delta_blue = 0
    
    # Check K4^3 violations: quads containing this triple
    # A quad {a,b,c,x} for x not in {a,b,c}
    for x in range(n):
        if x == a or x == b or x == c:
            continue
        quad = tuple(sorted([a, b, c, x]))
        
        # Count how many of the 4 triples in this quad are red/blue
        # The 4 triples are: (a,b,c), (a,b,x), (a,c,x), (b,c,x)
        other_triples = []
        for t in itertools.combinations(quad, 3):
            if t != triple:
                other_triples.append(t)
        
        # Current state before flip
        if old_color == 0:  # was red
            # If all other 3 are red, this quad is a red violation
            all_other_red = all(colors[triple_idx[t]] == 0 for t in other_triples)
            if all_other_red:
                delta_red -= 1  # removing a violation
            
            # If new_color is blue, no new red violation
        else:  # was blue
            # If all other 3 are red, flipping to red creates a red violation
            all_other_red = all(colors[triple_idx[t]] == 0 for t in other_triples)
            if all_other_red:
                delta_red += 1  # creating a violation
    
    # Check K5^3 violations: quints containing this triple
    # A quint {a,b,c,x,y} for x,y not in {a,b,c}, x<y
    for x in range(n):
        if x == a or x == b or x == c:
            continue
        for y in range(x + 1, n):
            if y == a or y == b or y == c:
                continue
            quint = tuple(sorted([a, b, c, x, y]))
            
            # The 10 triples in this quint
            other_triples = []
            for t in itertools.combinations(quint, 3):
                if t != triple:
                    other_triples.append(t)
            
            if old_color == 1:  # was blue
                # If all other 9 are blue, this quint is a blue violation
                all_other_blue = all(colors[triple_idx[t]] == 1 for t in other_triples)
                if all_other_blue:
                    delta_blue -= 1  # removing a violation
            else:  # was red
                # If all other 9 are blue, flipping to blue creates a blue violation
                all_other_blue = all(colors[triple_idx[t]] == 1 for t in other_triples)
                if all_other_blue:
                    delta_blue += 1  # creating a violation
    
    return delta_red, delta_blue

def _is_valid(n, colors, triple_idx):
    """Check if coloring has no red K4^3 and no blue K5^3."""
    # Check red K4^3
    for quad in itertools.combinations(range(n), 4):
        all_red = True
        for triple in itertools.combinations(quad, 3):
            if colors[triple_idx[triple]] != 0:
                all_red = False
                break
        if all_red:
            return False
    
    # Check blue K5^3
    for quint in itertools.combinations(range(n), 5):
        all_blue = True
        for triple in itertools.combinations(quint, 3):
            if colors[triple_idx[triple]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True
# EVOLVE-BLOCK-END
