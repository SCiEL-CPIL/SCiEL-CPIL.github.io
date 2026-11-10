"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _triple_indices(n):
    """Return list of (i,j,k) for all triples a<b<c in lex order."""
    return list(combinations(range(n), 3))

def _build_adjacency(n, colors, triples):
    """
    For red K4^(3): a red K4 is 4 vertices where all 4 triples are red.
    For blue K5^(3): a blue K5 is 5 vertices where all 10 triples are blue.
    
    We'll track for each vertex pair, which vertices complete a red K4 or blue K5.
    """
    # For red K4: for each pair (a,b), track which c>a,b make (a,b,c) red.
    # A red K4 on {a,b,c,d} requires all 4 triples red.
    # Strategy: for each 4-set, check if all 4 triples are red.
    # But that's O(C(n,4)*4) which is fine for n up to 40.
    
    # For efficiency, let's precompute:
    # red_triples: set of frozensets or a 2D/3D array
    # Actually, let's use a numpy array for colors indexed by triple index.
    
    return None

def _count_red_k4(n, colors, triples):
    """Count number of red K4^(3) subgraphs."""
    # Build a lookup: for triple (a,b,c), get color
    # Map (a,b,c) -> index in triples list
    triple_set = set()
    for idx, (a,b,c) in enumerate(triples):
        triple_set.add((a,b,c))
    
    count = 0
    for quad in combinations(range(n), 4):
        a,b,c,d = quad
        if colors[triple_set[(a,b,c)]] == 0 and \
           colors[triple_set[(a,b,d)]] == 0 and \
           colors[triple_set[(a,c,d)]] == 0 and \
           colors[triple_set[(b,c,d)]] == 0:
            count += 1
    return count

def _count_blue_k5(n, colors, triples):
    """Count number of blue K5^(3) subgraphs."""
    triple_set = set()
    for idx, (a,b,c) in enumerate(triples):
        triple_set.add((a,b,c))
    
    count = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[triple_set[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            count += 1
    return count

def _validate(n, colors):
    """Check if coloring is valid (no red K4, no blue K5)."""
    triples = _triple_indices(n)
    triple_map = {}
    for idx, (a,b,c) in enumerate(triples):
        triple_map[(a,b,c)] = idx
    
    # Check red K4
    for quad in combinations(range(n), 4):
        a,b,c,d = quad
        if (colors[triple_map[(a,b,c)]] == 0 and
            colors[triple_map[(a,b,d)]] == 0 and
            colors[triple_map[(a,c,d)]] == 0 and
            colors[triple_map[(b,c,d)]] == 0):
            return False
    
    # Check blue K5
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[triple_map[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True

def _objective(n, colors):
    """
    Objective to minimize: number of red K4 + number of blue K5.
    Valid coloring has objective 0.
    """
    triples = _triple_indices(n)
    triple_map = {}
    for idx, (a,b,c) in enumerate(triples):
        triple_map[(a,b,c)] = idx
    
    red_k4 = 0
    for quad in combinations(range(n), 4):
        a,b,c,d = quad
        if (colors[triple_map[(a,b,c)]] == 0 and
            colors[triple_map[(a,b,d)]] == 0 and
            colors[triple_map[(a,c,d)]] == 0 and
            colors[triple_map[(b,c,d)]] == 0):
            red_k4 += 1
    
    blue_k5 = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[triple_map[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            blue_k5 += 1
    
    return red_k4 + blue_k5

def _local_search(n, colors, triples, deadline, max_iters=10000):
    """
    Local search: flip colors to reduce objective.
    Returns (best_n, best_colors, best_obj).
    """
    triple_map = {}
    for idx, (a,b,c) in enumerate(triples):
        triple_map[(a,b,c)] = idx
    
    colors = list(colors)
    best_obj = _objective(n, colors)
    best_colors = list(colors)
    
    if best_obj == 0:
        return n, best_colors, 0
    
    # Precompute for each triple, which K4s and K5s it belongs to
    # For a triple (a,b,c), it's in K4s with d where d not in {a,b,c}
    # and in K5s with d,e where d,e not in {a,b,c}
    
    # For efficiency, let's compute the impact of flipping a triple
    # Flipping triple t from red to blue:
    # - Removes red K4s that include t and have other 3 triples red
    # - Adds blue K5s that include t and have other 9 triples blue
    
    # This is complex. Let's just do simple hill climbing with random restarts.
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random triple to flip
        idx = random.randint(0, len(triples) - 1)
        a,b,c = triples[idx]
        
        # Try flipping
        old_color = colors[idx]
        colors[idx] = 1 - old_color
        
        new_obj = _objective(n, colors)
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_colors = list(colors)
            if best_obj == 0:
                return n, best_colors, 0
        else:
            # Revert
            colors[idx] = old_color
    
    return n, best_colors, best_obj

def _grow(n, colors, triples, deadline):
    """
    Try to grow the coloring by adding vertices.
    This is complex; for now, we'll just try different n values.
    """
    return n, colors, _objective(n, colors)

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s margin
    
    # Strategy: try to find valid colorings for increasing n
    # Start with small n and use local search
    
    # Known: n=9 works with modular coloring
    # Try to find larger n
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Verify n=9
    if _validate(9, best_colors):
        pass
    else:
        # Should not happen
        best_n = 1
        best_colors = [0]
    
    # Try to find colorings for larger n using local search
    # We'll try n from 10 upward, but this is computationally expensive
    
    # For n=10, C(10,3)=120 triples, C(10,4)=210 quads, C(10,5)=252 quints
    # For n=20, C(20,3)=1140, C(20,4)=4845, C(20,5)=15504
    # For n=30, C(30,3)=4060, C(30,4)=27405, C(30,5)=142506
    # For n=34, C(34,3)=5984, C(34,4)=46376, C(34,5)=278256
    
    # The objective computation is O(C(n,4)*4 + C(n,5)*10)
    # For n=34, that's about 46376*4 + 278256*10 = 185504 + 2782560 = ~3M operations per objective eval
    # That's too slow for local search with many iterations.
    
    # Better approach: use a smarter objective or precomputation.
    
    # Let's try a different approach: start with random colorings and use local search
    # with a faster objective.
    
    # Actually, let's just try to find valid colorings for small n first,
    # then attempt to grow.
    
    # For the time being, let's return the n=9 solution if we can't do better.
    # But let's try a bit of local search on n=9 to see if we can find a valid one
    # (it should already be valid).
    
    # Let's try n=10 with local search
    if time.time() < deadline - 60:
        for n_try in [10, 11, 12, 13, 14, 15]:
            if time.time() > deadline - 30:
                break
            
            triples = _triple_indices(n_try)
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(len(triples))]
            
            # Local search
            n_found, colors_found, obj = _local_search(n_try, colors, triples, deadline - 30, max_iters=5000)
            
            if obj == 0 and n_found > best_n:
                best_n = n_found
                best_colors = colors_found
                break
            elif obj == 0:
                best_n = n_found
                best_colors = colors_found
    
    # If we found a valid coloring, return it
    if _validate(best_n, best_colors):
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return the n=9 modular coloring
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

# EVOLVE-BLOCK-END
