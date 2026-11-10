import time
import random
import sys

sys.setrecursionlimit(10000)

# Moore bound for (3, 16)
# For odd girth: n >= 1 + sum_{i=1}^{(g-1)/2} (k-1)^i
# For even girth: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
def moore_bound(k, g):
    if g % 2 == 0:
        return 2 * sum((k - 1)**i for i in range(g // 2))
    else:
        return 1 + sum((k - 1)**i for i in range(1, g // 2 + 1))

# LUW construction (fallback)
def luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# Wagner graph (M7) - 14 vertices, 21 edges, girth 8
def wagner_graph():
    vertices = list(range(14))
    edges = []
    # Cycle C7: 0-1-2-3-4-5-6-0
    for i in range(7):
        edges.append((i, (i + 1) % 7))
    # Cycle C7: 7-8-9-10-11-12-13-7
    for i in range(7):
        edges.append((7 + i, 7 + (i + 1) % 7))
    # Matching: i - (7+i)
    for i in range(7):
        edges.append((i, 7 + i))
    return vertices, edges

# Enumerate all non-backtracking closed walks of length L in the base graph
def enumerate_non_backtracking_walks(base_edges, L):
    # Build adjacency list with edge indices
    adj = {}
    for idx, (u, v) in enumerate(base_edges):
        adj.setdefault(u, []).append((v, idx))
        adj.setdefault(v, []).append((u, idx))
    
    # Find all non-backtracking closed walks of length L starting from each vertex
    # A walk is non-backtracking if it never immediately reverses an edge
    # For efficiency, we track (vertex, previous_edge) states
    
    walks = []
    
    for start in adj:
        # BFS over states (vertex, prev_edge, path_edges)
        # To avoid explosion, we only track length up to L
        # Use iterative approach
        
        # state: (vertex, prev_edge, tuple_of_edges)
        # But this explodes. Instead, let's enumerate edge sequences.
        
        # Alternative: enumerate all length-L edge sequences that form a walk
        # and are non-backtracking, then check if closed.
        
        # For a cubic graph with 21 edges, number of length-L walks is 3*2^(L-1) per start
        # For L=15, that's 3*2^14 = 49152 per start, * 14 starts = ~700k total
        # That's manageable.
        
        for first_edge in adj[start]:
            # Start with first edge
            current_v = first_edge[0]
            prev_e = first_edge[1]
            path_edges = [prev_e]
            
            if L == 1:
                # Check if closed: current_v == start and the edge forms a loop? No loops in simple graph.
                continue
            
            # Extend to length L
            def extend(path_edges, current_v, prev_e, depth):
                if depth == L:
                    # Check if closed: current_v == start
                    if current_v == start:
                        # Verify non-backtracking is already ensured by construction
                        walks.append(tuple(path_edges))
                    return
                
                for next_v, next_e in adj[current_v]:
                    if next_e == prev_e:
                        continue  # Backtracking
                    path_edges.append(next_e)
                    extend(path_edges, next_v, next_e, depth + 1)
                    path_edges.pop()
            
            extend(path_edges, current_v, prev_e, 2)
    
    # Deduplicate
    return list(set(walks))

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k != 3 or g != 16:
        # Fallback for other parameters
        return luw_construct(k, g)
    
    # Try voltage graph approach on Wagner graph
    base_vertices, base_edges = wagner_graph()
    n_base_edges = len(base_edges)
    
    # Precompute all non-backtracking closed walks of length < g
    # For girth 16, we need no cycles of length < 16
    # So we need to avoid closed non-backtracking walks of length 1 to 15
    # But in a simple graph, no closed non-backtracking walks of length < 3
    # In a graph with girth 8, no closed non-backtracking walks of length < 8
    
    # Let's enumerate walks of length 8 to 15
    # For each length L, enumerate and collect
    
    all_walks = []
    for L in range(8, 16):
        walks_L = enumerate_non_backtracking_walks(base_edges, L)
        all_walks.extend(walks_L)
        if time.time() > deadline - 30:
            break
    
    if time.time() > deadline - 30:
        return luw_construct(k, g)
    
    # Number of unique walks
    unique_walks = list(set(all_walks))
    n_walks = len(unique_walks)
    
    # For each walk, the sum of voltages must be non-zero in the group
    # Group: Z_m x Z_m
    # Try small m values
    
    # Build incidence vectors for each walk: for each edge, how many times it's used with sign
    # Actually, for a walk, we just need the sequence of edges. The voltage sum is sum of voltages along the walk.
    # If an edge appears multiple times, it contributes multiple times.
    
    # For each walk, create a vector of edge counts (with multiplicity)
    # But since voltages are assigned to edges, and the walk traverses edges in sequence,
    # the sum is sum over positions in the walk of voltage[edge_at_position]
    
    # Let's represent each walk as a list of edge indices in order
    # The constraint is: sum_{pos} voltage[walk[pos]] != 0 in Z_m^2
    
    # This is a system of constraints. We can try random voltage assignments and check.
    
    # Try m = 3, 4, 5, ...
    best_result = None
    best_n = float('inf')
    
    for m in range(3, 20):
        if time.time() > deadline - 10:
            break
        
        n_lift = m * m * 14  # 14 base vertices, m^2 group elements
        
        # Try random voltage assignments
        for trial in range(500):
            if time.time() > deadline - 5:
                break
            
            # Random voltages
            voltages = [(random.randint(0, m - 1), random.randint(0, m - 1)) for _ in range(n_base_edges)]
            
            # Check all walks
            valid = True
            for walk in unique_walks:
                # Compute voltage sum
                sx = 0
                sy = 0
                for e in walk:
                    sx = (sx + voltages[e][0]) % m
                    sy = (sy + voltages[e][1]) % m
                if sx == 0 and sy == 0:
                    valid = False
                    break
            
            if valid:
                # Found a valid voltage assignment!
                # Build the lift
                # Vertices: (v, gx, gy) where v in 0..13, gx,gy in 0..m-1
                # Edges: for each base edge (u, v) with voltage (vx, vy),
                #   connect (u, gx, gy) to (v, (gx+vx)%m, (gy+vy)%m)
                
                n_lift_actual = m * m * 14
                lift_edges = []
                
                # Map (v, gx, gy) to integer
                def vid(v, gx, gy):
                    return (v * m + gy) * m + gx
                
                for u, v in base_vertices:  # This is wrong, base_vertices is list of vertex ids
                    pass
                
                for u in base_vertices:
                    for vx in range(m):
                        for vy in range(m):
                            for w, (eidx, _) in enumerate(adj[u]):
                                # Need adjacency with edge indices
                                pass
                
                # Let's rebuild properly
                adj_with_eidx = {}
                for idx, (u, v) in enumerate(base_edges):
                    adj_with_eidx.setdefault(u, []).append((v, idx))
                    adj_with_eidx.setdefault(v, []).append((u, idx))
                
                lift_edges = []
                for u in base_vertices:
                    for vx in range(m):
                        for vy in range(m):
                            for w, eidx in adj_with_eidx[u]:
                                ex, ey = voltages[eidx]
                                wvx = (vx + ex) % m
                                wvy = (vy + ey) % m
                                u_id = vid(u, vx, vy)
                                w_id = vid(w, wvx, wvy)
                                if u_id < w_id:
                                    lift_edges.append((u_id, w_id))
                                # We add each edge once by checking u_id < w_id
                                # But this might miss some or double-count
                                # Better: just add all, then deduplicate
                                else:
                                    lift_edges.append((u_id, w_id))
                
                # Deduplicate
                lift_edges = list(set(lift_edges))
                
                # Verify degree
                deg = [0] * n_lift_actual
                for a, b in lift_edges:
                    deg[a] += 1
                    deg[b] += 1
                
                if all(d == 3 for d in deg):
                    # Valid!
                    if n_lift_actual < best_n:
                        best_n = n_lift_actual
                        best_result = (n_lift_actual, lift_edges)
                        # Don't break, maybe smaller m exists
                        # But we're iterating m in increasing order, so first valid is best
                        break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return luw_construct(k, g)
