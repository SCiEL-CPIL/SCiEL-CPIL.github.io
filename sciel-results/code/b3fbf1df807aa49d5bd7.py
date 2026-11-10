import time
import itertools

N = 16

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # Generate 1-factorization of K_16
    # Vertices 0..15
    # Factors 0..14
    # Each factor is a list of 8 edges (i, j) with i < j
    
    factors = []
    for k in range(15):
        factor_edges = []
        # Edge (15, k)
        factor_edges.append((15, k))
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u > v:
                u, v = v, u
            factor_edges.append((u, v))
        factors.append(factor_edges)
        
    # Map edge to color index (0..14)
    # Initially, color of edge in factor k is k
    edge_color = {}
    for k in range(15):
        for u, v in factors[k]:
            edge_color[(u, v)] = k
            edge_color[(v, u)] = k
            
    # List of all edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def get_color(u, v):
        if u > v:
            u, v = v, u
        return edge_color.get((u, v), 0)
        
    def set_color(u, v, c):
        if u > v:
            u, v = v, u
        edge_color[(u, v)] = c
        edge_color[(v, u)] = c
        
    def is_valid():
        # Check all 4-subsets
        for quad in itertools.combinations(range(N), 4):
            a, b, c, d = quad
            # Cycle 1: a-b-c-d
            c1 = get_color(a, b)
            c2 = get_color(b, c)
            c3 = get_color(c, d)
            c4 = get_color(d, a)
            if len({c1, c2, c3, c4}) == 4:
                return False
            # Cycle 2: a-b-d-c
            c1 = get_color(a, b)
            c2 = get_color(b, d)
            c3 = get_color(d, c)
            c4 = get_color(c, a)
            if len({c1, c2, c3, c4}) == 4:
                return False
            # Cycle 3: a-c-b-d
            c1 = get_color(a, c)
            c2 = get_color(c, b)
            c3 = get_color(b, d)
            c4 = get_color(d, a)
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True
        
    def find_rainbow_cycle():
        for quad in itertools.combinations(range(N), 4):
            a, b, c, d = quad
            # Cycle 1: a-b-c-d
            c1 = get_color(a, b)
            c2 = get_color(b, c)
            c3 = get_color(c, d)
            c4 = get_color(d, a)
            if len({c1, c2, c3, c4}) == 4:
                return [(a,b), (b,c), (c,d), (d,a)]
            # Cycle 2: a-b-d-c
            c1 = get_color(a, b)
            c2 = get_color(b, d)
            c3 = get_color(d, c)
            c4 = get_color(c, a)
            if len({c1, c2, c3, c4}) == 4:
                return [(a,b), (b,d), (d,c), (c,a)]
            # Cycle 3: a-c-b-d
            c1 = get_color(a, c)
            c2 = get_color(c, b)
            c3 = get_color(b, d)
            c4 = get_color(d, a)
            if len({c1, c2, c3, c4}) == 4:
                return [(a,c), (c,b), (b,d), (d,a)]
        return None
        
    # Check initial validity
    if not is_valid():
        while time.time() < deadline:
            cycle = find_rainbow_cycle()
            if cycle is None:
                break
            # Pick a random edge in the cycle and change its color to match a neighbor
            # To maximize colors, we should try to change a color that is unique in the cycle
            # to a color already present in the cycle.
            # e.g., if colors are 1,2,3,4, change 4 to 1.
            edges_in_cycle = cycle
            colors_in_cycle = [get_color(u, v) for u, v in edges_in_cycle]
            
            # Try to fix by changing one edge's color to another edge's color in the cycle
            # We want to minimize the reduction in total distinct colors used globally.
            # Simple heuristic: change the color of edge 3 to edge 0's color.
            u3, v3 = edges_in_cycle[2]
            c0 = get_color(edges_in_cycle[0][0], edges_in_cycle[0][1])
            set_color(u3, v3, c0)
            
    # Final check
    if not is_valid():
        # Fallback to 5-coloring if local search fails
        # c(i,j) = (i+j) % 5
        for i in range(N):
            for j in range(i+1, N):
                set_color(i, j, (i+j) % 5)
                
    # Count distinct colors
    used_colors = set()
    for i in range(N):
        for j in range(i+1, N):
            used_colors.add(get_color(i, j))
            
    # Map to 0..119 range (already 0..14 or 0..4)
    # The task requires integers in 0..119.
    # Our colors are small integers, so they are valid.
    
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(get_color(i, j))
            
    return result

