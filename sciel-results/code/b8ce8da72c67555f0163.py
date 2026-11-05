"""
Stronger algorithm for the K16 edge-coloring task:
- Precompute all 4-cycles (as sets of 4 edge indices) and, for each edge,
  the list of 4-cycles containing it.
- Start from a small number of colors (e.g., 3 colors in a structured way)
  and perform local search: repeatedly try to recolor an edge to a new color
  (preferably an unused color) while maintaining the constraint that no 4-cycle
  has four distinct colors.
- Use a deadline guard to return the best valid coloring found within ~85s.
"""
import time
import random
import itertools
from collections import Counter, defaultdict

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    """Map pair (i,j) to a unique index in [0, NUM_EDGES)."""
    if i > j:
        i, j = j, i
    # Count edges (i, k) for k < j, k != i, plus edges (k, j) for k < i
    # Standard formula: index = i*(2*N - i - 1)//2 + (j - i - 1)
    return i * (2 * N - i - 1) // 2 + (j - i - 1)

def _precompute_cycles():
    """
    Return:
      cycles: list of frozensets of 4 edge indices (all 4-cycles in K16)
      edge_cycles: list of lists, edge_cycles[e] = list of cycle indices that contain edge e
    """
    cycles = []
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Cycle 1: a-b-c-d-a
        e1 = _edge_idx(a, b)
        e2 = _edge_idx(b, c)
        e3 = _edge_idx(c, d)
        e4 = _edge_idx(d, a)
        cyc = frozenset([e1, e2, e3, e4])
        cycles.append(cyc)
        for e in [e1, e2, e3, e4]:
            edge_cycles[e].append(len(cycles) - 1)
        
        # Cycle 2: a-b-d-c-a
        e1 = _edge_idx(a, b)
        e2 = _edge_idx(b, d)
        e3 = _edge_idx(d, c)
        e4 = _edge_idx(c, a)
        cyc = frozenset([e1, e2, e3, e4])
        cycles.append(cyc)
        for e in [e1, e2, e3, e4]:
            edge_cycles[e].append(len(cycles) - 1)
        
        # Cycle 3: a-c-b-d-a
        e1 = _edge_idx(a, c)
        e2 = _edge_idx(c, b)
        e3 = _edge_idx(b, d)
        e4 = _edge_idx(d, a)
        cyc = frozenset([e1, e2, e3, e4])
        cycles.append(cyc)
        for e in [e1, e2, e3, e4]:
            edge_cycles[e].append(len(cycles) - 1)
    
    return cycles, edge_cycles

def _is_valid_recolor(colors, edge, new_color, cycles, edge_cycles):
    """
    Check if recoloring 'edge' to 'new_color' keeps all 4-cycles valid.
    Only need to check cycles containing 'edge'.
    """
    old_color = colors[edge]
    if old_color == new_color:
        return True
    
    for cyc_idx in edge_cycles[edge]:
        cyc = cycles[cyc_idx]
        # Count distinct colors in this cycle after recoloring
        distinct = set()
        for e in cyc:
            if e == edge:
                distinct.add(new_color)
            else:
                distinct.add(colors[e])
        if len(distinct) == 4:
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _initial_coloring(n_colors=3):
    """
    Create an initial valid coloring using n_colors colors.
    Use a simple structured approach: color edges based on (i+j) mod n_colors.
    This should be valid for small n_colors.
    """
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            e = _edge_idx(i, j)
            colors[e] = (i + j) % n_colors
    return colors

def _local_search(colors, cycles, edge_cycles, deadline, seed=42):
    """
    Local search: repeatedly try to recolor edges to increase distinct colors.
    Prefer recoloring to unused colors first.
    """
    rng = random.Random(seed)
    colors = colors[:]
    current_count = _count_distinct(colors)
    best_colors = colors[:]
    best_count = current_count
    
    used_colors = set(colors)
    unused_colors = [c for c in range(NUM_COLORS) if c not in used_colors]
    
    edge_order = list(range(NUM_EDGES))
    
    while time.time() < deadline:
        # Shuffle edge order for diversity
        rng.shuffle(edge_order)
        
        improved = False
        for edge in edge_order:
            if time.time() >= deadline:
                break
            
            old_color = colors[edge]
            
            # Try unused colors first (to increase distinct count)
            candidates = unused_colors[:]
            # Also try some random used colors for diversification
            used_list = list(used_colors)
            rng.shuffle(used_list)
            candidates.extend(used_list[:5])
            
            rng.shuffle(candidates)
            
            for new_color in candidates:
                if _is_valid_recolor(colors, edge, new_color, cycles, edge_cycles):
                    colors[edge] = new_color
                    # Update used/unused
                    if old_color not in used_colors - {old_color}:
                        # Check if old_color is still used
                        pass
                    used_colors = set(colors)
                    unused_colors = [c for c in range(NUM_COLORS) if c not in used_colors]
                    
                    new_count = _count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                    improved = True
                    break
        
        if not improved:
            # Try a few random restarts with slight perturbations
            for _ in range(3):
                if time.time() >= deadline:
                    break
                # Perturb: randomly recolor a few edges
                perturbed = best_colors[:]
                for _ in range(5):
                    e = rng.randint(0, NUM_EDGES - 1)
                    c = rng.randint(0, NUM_COLORS - 1)
                    if _is_valid_recolor(perturbed, e, c, cycles, edge_cycles):
                        perturbed[e] = c
                # Continue search from perturbed
                colors = perturbed
                used_colors = set(colors)
                unused_colors = [c for c in range(NUM_COLORS) if c not in used_colors]
                improved = False
                break
    
    return best_colors, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    cycles, edge_cycles = _precompute_cycles()
    
    best_colors = None
    best_count = 0
    
    # Try different initial colorings
    for n_init in [2, 3, 4]:
        if time.time() >= deadline:
            break
        init = _initial_coloring(n_init)
        # Verify initial validity
        valid = True
        for cyc in cycles:
            if len(set(init[e] for e in cyc)) == 4:
                valid = False
                break
        if not valid:
            continue
        
        colors, count = _local_search(init, cycles, edge_cycles, deadline, seed=n_init)
        if count > best_count:
            best_count = count
            best_colors = colors
    
    if best_colors is None:
        # Fallback: all zeros
        return [0] * NUM_EDGES
    
    return best_colors

# EVOLVE-BLOCK-END
