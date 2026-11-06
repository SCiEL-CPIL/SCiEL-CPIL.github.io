import itertools
import time

def construct():
    """
    Constructs a dense graph on 20 vertices with rainbow 4-cycles.
    Uses a 1-factorization of K_20 and greedily removes edges 
    involved in non-rainbow 4-cycles.
    """
    n = 20
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit
    
    # 1. Generate 1-factorization of K_20
    # Vertices 0..18 on circle, 19 is infinity
    matchings = []
    for k in range(19):
        m = []
        # Edge to infinity
        m.append((19, k))
        # Edges on the circle
        for j in range(1, 10):
            u = (k + j) % 19
            v = (k - j) % 19
            # Ensure u < v for consistent storage
            if u > v:
                u, v = v, u
            m.append((u, v))
        matchings.append(m)
        
    # 2. Assign colors and create edge list
    # edge_to_color: dict (u, v) -> color
    # edge_list: list of [u, v, c]
    edge_to_color = {}
    edge_list = []
    
    for c, m in enumerate(matchings):
        for u, v in m:
            edge_to_color[(u, v)] = c
            edge_list.append([u, v, c])
            
    # 3. Identify non-rainbow 4-cycles and remove edges
    # We store edges in a set for fast lookup
    # We will mark edges for removal
    to_remove = set()
    
    # Helper to get color of edge (u, v)
    def get_color(u, v):
        if u > v:
            u, v = v, u
        return edge_to_color.get((u, v), None)
        
    # Iterate over all sets of 4 vertices
    # There are 3 ways to form a 4-cycle from 4 vertices
    # v1-v2-v3-v4-v1
    # v1-v2-v4-v3-v1
    # v1-v3-v2-v4-v1
    
    vertices = range(n)
    
    # To be efficient, we can just iterate and check.
    # If time is running out, break.
    
    count = 0
    for v1, v2, v3, v4 in itertools.combinations(vertices, 4):
        if time.time() > deadline:
            break
            
        # Cycle 1: v1-v2-v3-v4-v1
        c1 = get_color(v1, v2)
        c2 = get_color(v2, v3)
        c3 = get_color(v3, v4)
        c4 = get_color(v4, v1)
        
        if c1 is not None and c2 is not None and c3 is not None and c4 is not None:
            if len({c1, c2, c3, c4}) < 4:
                # Remove one edge. Let's pick the one with the highest color index or arbitrary.
                # Removing (v3, v4)
                u, v = v3, v4
                if u > v: u, v = v, u
                to_remove.add((u, v))
                
        # Cycle 2: v1-v2-v4-v3-v1
        c1 = get_color(v1, v2)
        c2 = get_color(v2, v4)
        c3 = get_color(v4, v3)
        c4 = get_color(v3, v1)
        
        if c1 is not None and c2 is not None and c3 is not None and c4 is not None:
            if len({c1, c2, c3, c4}) < 4:
                u, v = v4, v3
                if u > v: u, v = v, u
                to_remove.add((u, v))
                
        # Cycle 3: v1-v3-v2-v4-v1
        c1 = get_color(v1, v3)
        c2 = get_color(v3, v2)
        c3 = get_color(v2, v4)
        c4 = get_color(v4, v1)
        
        if c1 is not None and c2 is not None and c3 is not None and c4 is not None:
            if len({c1, c2, c3, c4}) < 4:
                u, v = v2, v4
                if u > v: u, v = v, u
                to_remove.add((u, v))
                
    # 4. Filter edge list
    result = []
    for e in edge_list:
        u, v, c = e
        key = (u, v) if u < v else (v, u)
        if key not in to_remove:
            result.append(e)
            
    return result

# EVOLVE-BLOCK-END
