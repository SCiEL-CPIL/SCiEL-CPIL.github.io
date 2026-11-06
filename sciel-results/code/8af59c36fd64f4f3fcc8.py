"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices."""
import time
import random
from itertools import combinations

N = 20
DEADLINE_MARGIN = 5.0  # seconds to spare before the 150s wall-clock kill

def _all_4_cycles():
    """Generate all 4-cycles (sets of 4 edges) in K_n.
    A 4-cycle is determined by 4 vertices; there are 3 ways to pair them into a cycle.
    For vertices a<b<c<d, the three 4-cycles are:
      (a,b,c,d), (a,b,d,c), (a,c,b,d)
    Each cycle is a frozenset of 4 (u,v) edges with u<v.
    """
    cycles = []
    for a, b, c, d in combinations(range(N), 4):
        # Three ways to form a 4-cycle from 4 vertices
        for perm in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
            x, y, z, w = perm
            edges = set()
            for u, v in [(x, y), (y, z), (z, w), (w, x)]:
                edges.add((min(u, v), max(u, v)))
            cycles.append(frozenset(edges))
    return cycles

def _is_valid(edges_set, colors, cycles):
    """Check if the current edge set + coloring is valid (all 4-cycles are rainbow)."""
    for cyc in cycles:
        # Check all 4 edges exist
        if not all(e in edges_set for e in cyc):
            continue
        cols = [colors[e] for e in cyc]
        if len(set(cols)) < 4:
            return False
    return True

def _verify_result(result, cycles):
    """Verify a result is valid. Returns True if valid."""
    if not result:
        return False
    edges_set = set()
    colors = {}
    for u, v, c in result:
        e = (min(u, v), max(u, v))
        if e in edges_set:
            return False
        edges_set.add(e)
        colors[e] = c
    return _is_valid(edges_set, colors, cycles)

def _try_group_coloring(cycles, deadline):
    """
    Structured approach: Use a group-based edge coloring.
    
    Key insight: If we color edge (u,v) with color (v-u) mod N (or similar),
    then in any 4-cycle a-b-c-d-a, the colors are:
      (b-a), (c-b), (d-c), (a-d) mod N
    For this to be rainbow, we need all four differences to be distinct.
    
    However, (b-a) + (c-b) + (d-c) + (a-d) = 0, so they can't all be distinct
    unless we use a different structure.
    
    Better: Use a proper edge-coloring of K_N when N is odd (N-1 colors suffice for
    proper edge-coloring). But we need rainbow 4-cycles, which is stronger.
    
    Alternative structured approach: Use a tournament-based or bipartite construction.
    
    Actually, let's think about this differently. The constraint is that every 4-cycle
    must have 4 distinct colors. This is equivalent to saying that no two edges in any
    4-cycle share a color. 
    
    One clean construction: color edge (u,v) with (u XOR v) if we work in a field,
    but 20 isn't a prime power.
    
    Let me try: partition vertices into groups and use structured coloring within/between groups.
    
    Actually, a simpler approach that might work well:
    - Use a proper edge-coloring (no two adjacent edges share a color)
    - This doesn't guarantee rainbow 4-cycles, but it's a necessary condition
    - Then add constraints for 4-cycles
    
    Let me try a different structured approach:
    - Assign each vertex a "type" based on some partition
    - Color edges based on types in a way that forces rainbow 4-cycles
    """
    t_start = time.time()
    all_edges = list(combinations(range(N), 2))
    
    # Approach: Try multiple structured colorings and keep the best valid subset
    
    best_edges = 0
    best_result = []
    
    # Structure 1: Color (u,v) = (u + v) % N
    # Structure 2: Color (u,v) = (u - v) % N  
    # Structure 3: Color (u,v) = (u * v) % N
    # Structure 4: Use a Latin square based approach
    
    # For each structured coloring, greedily remove edges that violate the constraint
    # but be smarter: remove the edge that appears in the most violating cycles
    
    def greedy_remove_for_coloring(color_func):
        colors = {}
        for u, v in all_edges:
            colors[(u, v)] = color_func(u, v) % N
        
        edges_set = set(all_edges)
        
        # Precompute which cycles contain which edges for fast lookup
        # Actually, let's just iterate and check
        
        while True:
            if time.time() - t_start > deadline - 2.0:
                break
            
            # Find all bad cycles
            bad_edge_count = {}
            for cyc in cycles:
                if not all(e in edges_set for e in cyc):
                    continue
                cols = [colors[e] for e in cyc]
                if len(set(cols)) < 4:
                    for e in cyc:
                        bad_edge_count[e] = bad_edge_count.get(e, 0) + 1
            
            if not bad_edge_count:
                break
            
            # Remove the edge in the most bad cycles
            worst = max(bad_edge_count, key=lambda e: bad_edge_count[e])
            edges_set.remove(worst)
            del colors[worst]
        
        return [[u, v, colors[(u, v)]] for (u, v) in sorted(edges_set)]
    
    # Try several structured colorings
    color_funcs = [
        lambda u, v: (u + v) % N,
        lambda u, v: (u - v) % N,
        lambda u, v: (u + 2 * v) % N,
        lambda u, v: (2 * u + v) % N,
        lambda u, v: (u * v) % N,
        lambda u, v: (u * 3 + v * 7) % N,
        lambda u, v: (u * 5 + v * 3) % N,
        lambda u, v: (u * 7 + v * 11) % N,
    ]
    
    for cf in color_funcs:
        if time.time() - t_start > deadline - 3.0:
            break
        result = greedy_remove_for_coloring(cf)
        if len(result) > best_edges:
            best_edges = len(result)
            best_result = result
    
    # Also try random colorings with the same greedy removal
    rng = random.Random(42)
    for trial in range(10):
        if time.time() - t_start > deadline - 2.0:
            break
        def rand_color(u, v):
            return rng.randint(0, N - 1)
        # Need a different random coloring each time
        colors_dict = {}
        for u, v in all_edges:
            colors_dict[(u, v)] = rng.randint(0, N - 1)
        
        # Use the precomputed colors
        def precomputed_color(u, v):
            return colors_dict[(u, v)]
        
        result = greedy_remove_for_coloring(precomputed_color)
        if len(result) > best_edges:
            best_edges = len(result)
            best_result = result
    
    return best_result

def _try_bipartite_construction(cycles, deadline):
    """
    Try a bipartite-based construction.
    
    Split vertices into two sets A and B of size 10.
    Color edges within A, within B, and between A and B differently.
    
    For edges between A and B: color (a,b) = (a_index + b_index) mod 10, mapped to 0..19
    For edges within A: use another scheme
    For edges within B: use another scheme
    
    The key is to ensure 4-cycles are rainbow.
    """
    t_start = time.time()
    
    # Try different partitions
    best_edges = 0
    best_result = []
    
    for split_seed in range(5):
        if time.time() - t_start > deadline - 2.0:
            break
        
        rng = random.Random(split_seed)
        # Random partition into two sets of 10
        vertices = list(range(N))
        rng.shuffle(vertices)
        A = set(vertices[:10])
        B = set(vertices[10:])
        
        # Build coloring
        colors = {}
        all_edges = list(combinations(range(N), 2))
        
        for u, v in all_edges:
            if u in A and v in A:
                # Within A: use index-based coloring
                au = list(A).index(u)
                av = list(A).index(v)
                colors[(u, v)] = (au + av) % 10
            elif u in B and v in B:
                # Within B: use index-based coloring
                bu = list(B).index(u)
                bv = list(B).index(v)
                colors[(u, v)] = 10 + (bu + bv) % 10
            else:
                # Between A and B
                a_vert = u if u in A else v
                b_vert = v if v in A else u
                ai = list(A).index(a_vert)
                bi = list(B).index(b_vert)
                colors[(u, v)] = (ai + 2 * bi) % 20
        
        edges_set = set(all_edges)
        
        # Greedy removal
        while True:
            if time.time() - t_start > deadline - 2.0:
                break
            
            bad_edge_count = {}
            for cyc in cycles:
                if not all(e in edges_set for e in cyc):
                    continue
                cols = [colors[e] for e in cyc]
                if len(set(cols)) < 4:
                    for e in cyc:
                        bad_edge_count[e] = bad_edge_count.get(e, 0) + 1
            
            if not bad_edge_count:
                break
            
            worst = max(bad_edge_count, key=lambda e: bad_edge_count[e])
            edges_set.remove(worst)
            del colors[worst]
        
        result = [[u, v, colors[(u, v)]] for (u, v) in sorted(edges_set)]
        if len(result) > best_edges:
            best_edges = len(result)
            best_result = result
    
    return best_result

def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    cycles = _all_4_cycles()
    
    best_edges = 0
    best_result = []
    
    # Try structured group coloring
    if deadline - time.time() > 10.0:
        result = _try_group_coloring(cycles, deadline)
        if len(result) > best_edges:
            best_edges = len(result)
            best_result = result
    
    # Try bipartite construction
    if deadline - time.time() > 10.0:
        result = _try_bipartite_construction(cycles, deadline)
        if len(result) > best_edges:
            best_edges = len(result)
            best_result = result
    
    # Final verification and return
    if best_result and _verify_result(best_result, cycles):
        return best_result
    
    # Fallback: return a small valid graph
    # Just a star: all edges from 0 to others, all different colors
    fallback = [[0, v, v] for v in range(1, N)]
    return fallback
