import time
import random
import itertools

def construct(k, g):
    # Fallback: LUW graph
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

    if k != 3 or g != 16:
        return luw_construct(k, g)

    deadline = time.time() + 140
    
    # Petersen graph edges
    # Vertices 0..9
    # 0:{0,1}, 1:{0,2}, 2:{0,3}, 3:{0,4}, 4:{1,2}, 5:{1,3}, 6:{1,4}, 7:{2,3}, 8:{2,4}, 9:{3,4}
    # Adjacency:
    # 0: 7,8,9
    # 1: 5,6,9
    # 2: 4,6,8
    # 3: 4,5,7
    # 4: 2,3,9
    # 5: 1,3,8
    # 6: 1,2,7
    # 7: 0,3,6
    # 8: 0,2,5
    # 9: 0,1,4
    
    base_edges = [
        (0, 7), (0, 8), (0, 9),
        (1, 5), (1, 6), (1, 9),
        (2, 4), (2, 6), (2, 8),
        (3, 4), (3, 5), (3, 7),
        (4, 9), # 4-2, 4-3, 4-9
        (5, 8), # 5-1, 5-3, 5-8
        (6, 7)  # 6-1, 6-2, 6-7
    ]
    # Note: undirected edges, so (u,v) is same as (v,u).
    # We need to assign voltage v_e for each edge.
    # Let's index edges 0..14.
    edge_index = {e: i for i, e in enumerate(base_edges)}
    # Also map (v,u) to same index
    for u, v in base_edges:
        edge_index[(v, u)] = edge_index[(u, v)]

    # Generate all non-backtracking closed walks of length L < 16
    # Walk is a sequence of vertices v0, v1, ..., vL = v0
    # Non-backtracking: v_{i-1} != v_{i+1} for i=1..L-1
    # We represent a walk by the sequence of edge indices.
    # Actually, we just need the linear combination of voltages.
    # A walk is defined by start vertex and sequence of neighbors.
    
    constraints = []
    
    # Precompute neighbors
    neighbors = [[] for _ in range(10)]
    for u, v in base_edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
        
    # To avoid duplicates, we can just generate all walks.
    # Number of walks is small.
    # Length 5: 10*3*2*2*2*2? No.
    # From a vertex, 3 choices. Then 2 choices (non-backtracking).
    # So 10 * 3 * 2^4 = 480 walks of length 5.
    # Many are not closed.
    
    for L in range(5, 16):
        if time.time() > deadline:
            break
        # Generate all non-backtracking walks of length L starting from each vertex
        # We can use DFS or just enumerate paths.
        # Since L is small, recursion is fine.
        
        def gen_walks(start, current, prev, L_remaining, path):
            if L_remaining == 0:
                if current == start:
                    # Check if it's a closed walk
                    # It is non-backtracking by construction
                    # Calculate voltage sum
                    # path is list of (u, v) edges
                    # We need to sum voltages.
                    # But we don't have voltages yet.
                    # We store the edge indices in the walk.
                    # The walk is a sequence of edges e1, ..., eL.
                    # The voltage sum is sum(v_ei).
                    # Note: if the walk traverses an edge in reverse, the voltage is -v_e.
                    # So we need to track direction.
                    # Let's store (u, v) for each step.
                    # If step is (u, v), and edge_index[(u,v)] = idx, then voltage is +v_idx if we defined v_uv = v_idx.
                    # But v_vu = -v_idx.
                    # So we just sum the voltages of the directed edges.
                    # We can represent the constraint as a vector of coefficients for v_0..v_14.
                    # coeff[idx] += 1 if direction matches base edge orientation?
                    # No, let's just say v_e is the voltage from u to v.
                    # Then v_vu = -v_e.
                    # So if we traverse (u, v), we add v_e.
                    # If we traverse (v, u), we add -v_e.
                    # We can define the base orientation as (min(u,v), max(u,v)).
                    # Let base_dir = (min(u,v), max(u,v)).
                    # If walk step is (u, v):
                    #   if (u, v) == base_dir: coeff[idx] += 1
                    #   else: coeff[idx] -= 1
                    pass
            else:
                for next_v in neighbors[current]:
                    if next_v != prev:
                        path.append((current, next_v))
                        gen_walks(start, next_v, current, L_remaining - 1, path)
                        path.pop()

        # This recursion might be slow if not optimized, but L=15 is small.
        # Let's collect constraints.
        # We need to avoid duplicate constraints?
        # If two walks have the same voltage sum expression, they are the same constraint.
        # We can use a set of tuples of coefficients.
        
        seen_constraints = set()
        
        for start in range(10):
            gen_walks(start, start, -1, L, [])
            
        # The above function doesn't return constraints.
        # Let's rewrite to collect them.
        
    # The recursion above is a bit messy. Let's do it more cleanly.
    
    all_constraints = []
    
    def get_walks(start, current, prev, L_remaining, path):
        if L_remaining == 0:
            if current == start:
                # Build constraint vector
                coeff = [0] * 15
                for u, v in path:
                    idx = edge_index[(u, v)]
                    # Determine direction
                    base_u, base_v = base_edges[idx]
                    if (u, v) == (base_u, base_v):
                        coeff[idx] += 1
                    else:
                        coeff[idx] -= 1
                # Convert to tuple for hashing
                all_constraints.append(tuple(coeff))
        else:
            for next_v in neighbors[current]:
                if next_v != prev:
                    path.append((current, next_v))
                    get_walks(start, next_v, current, L_remaining - 1, path)
                    path.pop()

    for L in range(5, 16):
        for start in range(10):
            get_walks(start, start, -1, L, [])
            
    # Remove duplicates
    unique_constraints = list(set(all_constraints))
    
    # Now search for voltage assignments
    # Try primes
    primes = [79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    for p in primes:
        if time.time() > deadline:
            break
        # Try random assignments
        # Number of edges is 15.
        # We can try to assign voltages one by one?
        # Or just random full assignment.
        # Probability of success is (1 - 1/p)^M.
        # M is len(unique_constraints).
        
        M = len(unique_constraints)
        # If M is large, random is slow.
        # But let's try.
        
        # To speed up, we can fix some voltages?
        # No, let's just try random.
        
        # We can also try to solve a subset of constraints first.
        
        # Let's try 1000 random assignments per prime.
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Random voltages in 1..p-1
            # Ensure not all zero?
            # If all zero, girth is 5.
            # So at least one non-zero.
            voltages = [random.randint(1, p - 1) for _ in range(15)]
            
            # Check constraints
            valid = True
            for c in unique_constraints:
                s = 0
                for i in range(15):
                    if c[i] != 0:
                        s += c[i] * voltages[i]
                if s % p == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the graph
                # Vertices: (v, g) for v in 0..9, g in 0..p-1
                # Total 10*p vertices.
                n = 10 * p
                edges = []
                # Map (v, g) to index
                # index = v * p + g
                
                for u in range(10):
                    for v in neighbors[u]:
                        idx = edge_index[(u, v)]
                        vol = voltages[idx]
                        # Edge from (u, g) to (v, g+vol)
                        # We need to add each undirected edge once.
                        # Iterate over all u, v in base_edges
                        # For each g, add edge
                        # But we must ensure we don't add duplicates.
                        # Base edges are undirected.
                        # Let's iterate over base_edges
                        pass
                        
                # Better: iterate over base_edges
                for idx, (u, v) in enumerate(base_edges):
                    vol = voltages[idx]
                    for g in range(p):
                        # Edge from (u, g) to (v, g+vol)
                        # But this adds each edge twice if we do it for all u,v?
                        # No, base_edges contains each undirected edge once.
                        # So for each base edge (u, v), we add edges in the lift.
                        # The lift has edges between (u, g) and (v, g+vol).
                        # Is this edge the same as (v, g+vol) and (u, g)?
                        # Yes.
                        # Do we add it again when we process (v, u)?
                        # No, because (v, u) is not in base_edges.
                        # So we add each edge in the lift exactly once?
                        # Wait.
                        # In the lift, the edge connects (u, g) and (v, g+vol).
                        # If we process (u, v), we add this edge.
                        # If we process (v, u), we would add edge between (v, g') and (u, g'+vol_vu).
                        # vol_vu = -vol_uv.
                        # So (v, g') and (u, g'-vol).
                        # This is the same set of edges.
                        # So we just iterate over base_edges.
                        
                        u_id = u * p + g
                        v_id = v * p + (g + vol) % p
                        edges.append((u_id, v_id))
                        
                # Check if graph is connected?
                # If voltages generate Z_p, it is connected.
                # Since we picked random non-zero voltages, it's likely connected.
                # But we should verify.
                # If not connected, we return the largest component?
                # No, the task requires a single graph.
                # If the lift is disconnected, the girth is the min girth of components.
                # But if it's disconnected, the size is smaller?
                # No, the size is still 10*p.
                # But the graph is not connected.
                # The task says "Construct a 3-regular graph". It implies connected?
                # Usually cages are connected.
                # Let's assume connected.
                
                return n, edges

    # Fallback
    return luw_construct(k, g)
