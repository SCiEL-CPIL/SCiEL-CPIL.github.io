import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 110.0

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

    if k == 4 and g == 10:
        # Try to construct 320-vertex graph via voltage lift of Q4
        n_target = 320
        base_n = 16
        group_order = 20
        
        # Q4 edges
        q4_edges = []
        for u in range(16):
            for bit in range(4):
                v = u ^ (1 << bit)
                if u < v:
                    q4_edges.append((u, v))
        
        # Find all closed walks of length L < g in Q4
        # We need to check lengths 4, 6, 8
        # A closed walk is a sequence of vertices v0..vL where v0=vL and (vi, vi+1) is an edge
        # We represent walks by their edge sequences or vertex sequences
        # To avoid duplicates, we can just generate all walks and check sums
        # Number of walks of length L starting at v is 4^L
        # Total walks L=8: 16 * 4^8 = 1M. This is fine.
        
        # Precompute edge indices for fast voltage lookup
        # Map (u,v) to index
        edge_map = {}
        for i, (u, v) in enumerate(q4_edges):
            edge_map[(u, v)] = i
            edge_map[(v, u)] = i
            
        # Generate all closed walks of length 4, 6, 8
        closed_walks = []
        for L in [4, 6, 8]:
            for start_v in range(16):
                # DFS/BFS to find all walks of length L returning to start_v
                # State: (current, length_so_far, path_edges)
                # Path edges is list of edge indices
                stack = [(start_v, 0, [])]
                while stack:
                    curr, l, path = stack.pop()
                    if l == L:
                        if curr == start_v:
                            closed_walks.append(path)
                        continue
                    for nb in range(16):
                        # Check if edge exists
                        # Q4 neighbors are u ^ (1<<bit)
                        # We can just iterate neighbors
                        pass # Placeholder, will use neighbor list
                    
                    # Better: Precompute neighbors
        # Precompute neighbors for Q4
        q4_neighbors = [[] for _ in range(16)]
        for u in range(16):
            for bit in range(4):
                v = u ^ (1 << bit)
                q4_neighbors[u].append(v)
                
        closed_walks = []
        for L in [4, 6, 8]:
            for start_v in range(16):
                stack = [(start_v, 0, [])]
                while stack:
                    curr, l, path = stack.pop()
                    if l == L:
                        if curr == start_v:
                            closed_walks.append(path)
                        continue
                    for nb in q4_neighbors[curr]:
                        e_idx = edge_map[(curr, nb)]
                        stack.append((nb, l + 1, path + [e_idx]))
                        
        # Random search for voltage assignment
        best_edges = None
        for trial in range(1000):
            if time.time() > start_time + deadline:
                break
            voltages = [random.randint(0, group_order - 1) for _ in q4_edges]
            valid = True
            for path in closed_walks:
                s = sum(voltages[e] for e in path) % group_order
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                lift_edges = []
                # Vertices are (base_v, group_elem) -> index = base_v * group_order + group_elem
                for u, v in q4_edges:
                    e_idx = edge_map[(u, v)]
                    vol = voltages[e_idx]
                    for w in range(group_order):
                        u_lift = u * group_order + w
                        v_lift = v * group_order + (w + vol) % group_order
                        lift_edges.append((u_lift, v_lift))
                # Also need edges for reverse direction? 
                # The graph is undirected. The lift of an undirected edge (u,v) with voltage v_e
                # connects (u,w) to (v, w+v_e).
                # Since we iterate over all undirected edges in q4_edges, and for each w,
                # we add one edge. This creates the full graph.
                # Is it simple? Yes.
                # Is it 4-regular? Yes.
                best_edges = lift_edges
                break
        
        if best_edges is not None:
            return n_target, best_edges
        else:
            # Fallback to LUW
            return luw_construct(k, g)
    else:
        return luw_construct(k, g)
