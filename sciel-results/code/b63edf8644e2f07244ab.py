import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    n = 20
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Try to build a dense graph with rainbow 4-cycles
    # Strategy: start with complete graph, try to assign colours, remove edges that can't be coloured
    
    # First, let's try a structured approach using modular arithmetic
    # Use the fact that if we assign colour based on (u+v) mod 20 or similar, we might work
    
    # Let's try: for edge (u,v), assign colour = (u + v) % 20
    # Check if this works for 4-cycles
    # In a 4-cycle (a,b,c,d,a): colours are (a+b)%20, (b+c)%20, (c+d)%20, (d+a)%20
    # These need to be all distinct
    
    # This might not work in general. Let's use a different approach.
    
    # Greedy: try to add edges one by one, assign colours, check constraints
    # For efficiency, we'll use a heuristic ordering
    
    # Actually, let's try a specific known-good structure first
    # Use a complete bipartite graph K_{10,10} which has 100 edges
    # In K_{10,10}, any 4-cycle is of the form (a1,b1,a2,b2,a1)
    # We can assign colours to avoid conflicts
    
    # Let's try: left vertices 0-9, right vertices 10-19
    # For edge (i, 10+j), assign colour = (i + j) % 20
    # In a 4-cycle (i1, 10+j1, i2, 10+j2, i1):
    # colours: (i1+j1)%20, (i2+j1)%20, (i2+j2)%20, (i1+j2)%20
    # These are distinct if the 2x2 matrix [[i1+j1, i1+j2], [i2+j1, i2+j2]] has all distinct entries mod 20
    # This is true if (i1-i2) and (j1-j2) are such that no two sums collide
    # (i1+j1) = (i2+j2) iff i1-i2 = j2-j1
    # (i1+j1) = (i2+j1) iff i1=i2 (not possible since distinct vertices)
    # So we need: (i1+j1) != (i2+j2), (i1+j2) != (i2+j1)
    # i1+j1 = i2+j2 => i1-i2 = j2-j1
    # i1+j2 = i2+j1 => i1-i2 = j1-j2
    # These are different conditions unless j1=j2 or i1=i2
    
    # For distinct i1,i2 and j1,j2, we need both (i1-i2) != (j2-j1) and (i1-i2) != (j1-j2)
    # i.e., i1-i2 != ±(j2-j1)
    
    # This doesn't always hold. So K_{10,10} with this colouring may have non-rainbow 4-cycles.
    
    # Let's just do a greedy search with time limit
    
    best_edges = []
    best_count = 0
    
    # Try multiple random seeds
    for seed in range(50):
        if time.time() - start_time > deadline - 5:
            break
        random.seed(seed)
        
        # Start with a random subset of edges
        # Try to build up from a base structure
        edges = []
        edge_colors = {}
        
        # Try to add all edges, assigning colours greedily
        # Order edges by some heuristic
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # For each edge, try to assign a colour that doesn't create a non-rainbow 4-cycle
        for (u, v) in shuffled_edges:
            if time.time() - start_time > deadline - 2:
                break
            found_color = False
            for c in range(20):
                # Check if adding (u,v) with colour c creates a non-rainbow 4-cycle
                valid = True
                # A 4-cycle containing (u,v) has form (u, x, v, y, u) or (u, x, y, v, u) etc.
                # More precisely, a 4-cycle using edge (u,v) must have two other edges: (u,w) and (v,z) with (w,z) also present, forming u-w-z-v-u
                # So we need: for any w,z such that (u,w), (v,z), (w,z) are in the graph,
                # the colours of (u,v), (u,w), (w,z), (v,z) must be all distinct
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (u, w) not in edge_colors and (w, u) not in edge_colors:
                        continue
                    cuw = edge_colors.get((u, w), edge_colors.get((w, u)))
                    if cuw is None:
                        continue
                    for z in range(n):
                        if z == u or z == v or z == w:
                            continue
                        if (v, z) not in edge_colors and (z, v) not in edge_colors:
                            continue
                        cvz = edge_colors.get((v, z), edge_colors.get((z, v)))
                        if cvz is None:
                            continue
                        if (w, z) not in edge_colors and (z, w) not in edge_colors:
                            continue
                        cwz = edge_colors.get((w, z), edge_colors.get((z, w)))
                        if cwz is None:
                            continue
                        # Check 4-cycle u-v-z-w-u: edges (u,v)=c, (v,z)=cvz, (z,w)=cwz, (w,u)=cuw
                        colors = [c, cvz, cwz, cuw]
                        if len(set(colors)) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    edge_colors[(u, v)] = c
                    edges.append([u, v, c])
                    found_color = True
                    break
            
            if not found_color:
                # Try other colours already used or skip
                pass
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        if best_count >= 100:  # good enough, stop
            break
    
    # Verify the best solution
    if best_edges:
        # Build adjacency for verification
        adj = {}
        for u, v, c in best_edges:
            adj.setdefault((u, v), c)
            adj.setdefault((v, u), c)
        
        # Check all 4-cycles
        valid = True
        vertices = set(range(n))
        # For each pair of vertices, check 4-cycles
        # A 4-cycle is (a,b,c,d) with a<b<c<d not necessarily, but as a subgraph
        # Enumerate all 4-cycles: choose 4 vertices, check if they form a cycle
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Possible 4-cycles on {a,b,c,d}: 
            # (a,b,c,d,a), (a,b,d,c,a), (a,c,b,d,a)
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)],
            ]
            for cyc in cycles:
                # Check if all edges exist
                edge_list = []
                ok = True
                for (x, y) in cyc:
                    if (x, y) in adj:
                        edge_list.append(adj[(x, y)])
                    elif (y, x) in adj:
                        edge_list.append(adj[(y, x)])
                    else:
                        ok = False
                        break
                if ok and len(edge_list) == 4:
                    if len(set(edge_list)) != 4:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            return best_edges
    
    # Fallback: return a star if nothing worked
    return [[0, v, 0] for v in range(1, 20)]
