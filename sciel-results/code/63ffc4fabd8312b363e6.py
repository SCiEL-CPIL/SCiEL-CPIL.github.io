"""
Dense 20-vertex graph with 20 edge-colours such that every 4-cycle is rainbow.
Strategy: start from a random dense graph with a valid rainbow-4-cycle colouring,
then try to add edges (greedy + local search) while maintaining the rainbow constraint.
"""
import random
import itertools
import time
from collections import defaultdict

N = 20
C = 20

def get_all_4cycles(n):
    """Precompute all 4-cycles (as tuples of 4 distinct vertices in cycle order)."""
    cycles = []
    vertices = list(range(n))
    # A 4-cycle is (a,b,c,d) with edges a-b, b-c, c-d, d-a
    # To avoid duplicates, we can use a < b and a < d, and b < c or something.
    # Actually, let's just generate all combinations of 4 vertices and all 3 ways to arrange them in a cycle.
    # For 4 vertices {a,b,c,d}, the cycles are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a (but these are the same cycle in reverse or different pairing)
    # Actually the distinct 4-cycles on 4 vertices are:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) - but (a,b,c,d) and (a,d,c,b) are the same.
    # Let's just enumerate all 4-cycles properly.
    # A 4-cycle is determined by choosing 4 vertices and one of the 3 perfect matchings of the remaining edges... 
    # No, a 4-cycle on vertices {a,b,c,d} has edges {ab, bc, cd, da} or {ab, bd, dc, ca} or {ac, cb, bd, da}.
    # These correspond to the 3 ways to partition the 4 vertices into 2 pairs for the "diagonals".
    # Let's just generate them: for each 4-subset, there are 3 distinct 4-cycles.
    
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        cycles.append((a, b, c, d))
        # Cycle 2: a-b-d-c-a
        cycles.append((a, b, d, c))
        # Cycle 3: a-c-b-d-a
        cycles.append((a, c, b, d))
    return cycles

# Precompute all 4-cycles
ALL_CYCLES = get_all_4cycles(N)

def is_rainbow_4cycle(cycle, edge_colors):
    """Check if a 4-cycle (a,b,c,d) has 4 distinct colours on edges ab, bc, cd, da."""
    a, b, c, d = cycle
    edges = [(a,b), (b,c), (c,d), (d,a)]
    colors = []
    for u, v in edges:
        key = (min(u,v), max(u,v))
        if key not in edge_colors:
            return False  # edge missing, not a valid 4-cycle in the graph
        colors.append(edge_colors[key])
    return len(set(colors)) == 4

def check_all_cycles(edge_colors):
    """Check all 4-cycles in the graph defined by edge_colors."""
    for cycle in ALL_CYCLES:
        if not is_rainbow_4cycle(cycle, edge_colors):
            # Check if all 4 edges exist
            a, b, c, d = cycle
            edges = [(min(a,b),max(a,b)), (min(b,c),max(b,c)), (min(c,d),max(c,d)), (min(d,a),max(d,a))]
            if all(e in edge_colors for e in edges):
                return False
    return True

def try_add_edge(edge_colors, u, v, color, all_cycles=None):
    """Try to add edge (u,v) with color. Returns True if valid."""
    if all_cycles is None:
        all_cycles = ALL_CYCLES
    key = (min(u,v), max(u,v))
    if key in edge_colors:
        return False
    
    # Check all 4-cycles that include this edge
    # A 4-cycle contains edge (u,v) if the cycle has u and v adjacent.
    # For efficiency, let's just check all cycles that contain both u and v.
    # Actually, let's just check all cycles and see if (u,v) is one of the 4 edges.
    # This is O(#cycles) per add, which is ~ 1050*3 = 3150 cycles. That's fine.
    
    for cycle in all_cycles:
        a, b, c, d = cycle
        edges = [(min(a,b),max(a,b)), (min(b,c),max(b,c)), (min(c,d),max(c,d)), (min(d,a),max(d,a))]
        if key not in edges:
            continue
        # This cycle involves the new edge. Check if all 4 edges exist.
        if not all(e in edge_colors or e == key for e in edges):
            continue
        # All 4 edges exist (or will exist after adding key). Check rainbow.
        colors = []
        for e in edges:
            if e == key:
                colors.append(color)
            else:
                colors.append(edge_colors[e])
        if len(set(colors)) != 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    random.seed(42)
    
    # Start with an empty graph
    edge_colors = {}
    
    # First, build a valid base graph. Let's try to add edges greedily.
    # Order edges by some heuristic. Let's try random order with multiple restarts.
    
    best_edges = 0
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(1000 + restart)
        edge_colors = {}
        
        # Generate all possible edges
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            key = (min(u,v), max(u,v))
            if key in edge_colors:
                continue
            
            # Try each color
            colors_to_try = list(range(C))
            random.shuffle(colors_to_try)
            for color in colors_to_try:
                if try_add_edge(edge_colors, u, v, color):
                    edge_colors[key] = color
                    break
        
        if len(edge_colors) > best_edges:
            best_edges = len(edge_colors)
            best_solution = dict(edge_colors)
        
        # Local search: try to improve by swapping colors or adding more edges
        # This is complex, so let's just do multiple random restarts for now.
    
    # Convert to required format
    if best_solution is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, N)]
    
    result = []
    for (u, v), c in best_solution.items():
        result.append([u, v, c])
    
    return result
