import time
import random
import sys

# Helper to generate PG(2,3) incidence graph
def get_pg23_graph():
    # Points are 0..12, Lines are 13..25
    # Line i contains points based on projective plane structure
    # PG(2,3) has 13 points, 13 lines.
    # We can define lines by their 4 points.
    # Standard representation:
    # Points: (x,y,z) in GF(4) non-zero, up to scale.
    # But easier to hardcode the incidence matrix or generate via combinatorics.
    # For PG(2,3), the lines can be indexed 0..12.
    # Incidence:
    # Line 0: {0, 1, 2, 3}
    # Line 1: {0, 4, 5, 6}
    # ...
    # It's easier to use the known structure or generate it.
    # Let's generate it using the property that it's a Steiner system S(2,4,13).
    
    points = list(range(13))
    lines = []
    # Generate all blocks of size 4 from 13 points
    # such that every pair of points is in exactly one block.
    # This is the unique Steiner system S(2,4,13).
    # We can hardcode one or generate.
    # Hardcoding is safer for speed.
    
    # One standard list of lines for S(2,4,13):
    line_points = [
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [0, 7, 8, 9],
        [0, 10, 11, 12],
        [1, 4, 7, 10],
        [1, 5, 8, 11],
        [1, 6, 9, 12],
        [2, 4, 8, 12],
        [2, 5, 7, 11],
        [2, 6, 10, 9],
        [3, 4, 9, 11],
        [3, 5, 10, 8],
        [3, 6, 7, 12]
    ]
    
    edges = []
    for i, pts in enumerate(line_points):
        for p in pts:
            # Point p is 0..12, Line i is 13+i
            edges.append((p, 13 + i))
            
    return 26, edges

def find_cycles(nodes, edges, length):
    # Find all cycles of given length
    # Simple DFS for small graphs
    adj = {i: [] for i in range(nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    cycles = []
    # To avoid duplicates, only consider cycles starting at the smallest node
    # and traversing in a fixed direction (e.g., increasing indices)
    
    def dfs(current, start, path, visited):
        if len(path) == length:
            if current == start:
                # Check if it's a simple cycle (no repeated nodes except start/end)
                # Since we track visited, it is simple.
                # Normalize: rotate so min node is first
                cycle = path[1:] # path[0] is start
                min_idx = cycle.index(min(cycle))
                norm = tuple(cycle[min_idx:] + cycle[:min_idx])
                # Also check reverse?
                # For undirected graph, cycle and reverse are same.
                # We can just add the normalized tuple.
                cycles.append(norm)
            return
        
        for neighbor in adj[current]:
            if neighbor == start and len(path) < length:
                continue # Don't close early
            if neighbor in visited:
                continue
            if neighbor < start:
                continue # Only allow nodes > start to avoid duplicates?
                # This restricts to cycles where all nodes > start, except start.
                # This works if we iterate start from 0 to N-1.
            
            visited.add(neighbor)
            path.append(neighbor)
            dfs(neighbor, start, path, visited)
            path.pop()
            visited.remove(neighbor)
            
    for start in range(nodes):
        dfs(start, start, [start], {start})
        
    # Remove duplicates
    return list(set(cycles))

def solve_csp(edges, cycles, m, time_limit=10):
    # Assign voltages 0..m-1 to edges
    # Constraint: sum(v[e] for e in cycle) != 0 mod m
    # Note: The cycle edges need to be mapped to edge indices.
    # We need a mapping from (u,v) to edge_index.
    
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    cycle_edges = []
    for cycle in cycles:
        ce = []
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1)%len(cycle)]
            ce.append(edge_index[(u, v)])
        cycle_edges.append(ce)
        
    num_edges = len(edges)
    voltages = [random.randint(0, m-1) for _ in range(num_edges)]
    
    start_time = time.time()
    
    # Simple Local Search
    # Objective: Minimize number of violated constraints
    def count_violations():
        viol = 0
        for ce in cycle_edges:
            s = sum(voltages[e] for e in ce) % m
            if s == 0:
                viol += 1
        return viol
        
    current_viol = count_violations()
    
    while time.time() - start_time < time_limit:
        if current_viol == 0:
            return voltages
            
        # Pick a random violated constraint
        violated = []
        for i, ce in enumerate(cycle_edges):
            s = sum(voltages[e] for e in ce) % m
            if s == 0:
                violated.append(i)
                
        if not violated:
            return voltages
            
        ci = random.choice(violated)
        ce = cycle_edges[ci]
        
        # Try changing one edge in this cycle
        # Pick a random edge in the cycle
        ei = random.choice(ce)
        
        # Try all possible values for this edge?
        # Or just random?
        # Random is faster but might get stuck.
        # Let's try random first.
        old_val = voltages[ei]
        new_val = random.randint(0, m-1)
        if new_val == old_val:
            continue
            
        voltages[ei] = new_val
        new_viol = count_violations()
        
        if new_viol < current_viol:
            current_viol = new_viol
        else:
            # Accept with some probability? Or just reject?
            # Simple hill climbing rejects.
            voltages[ei] = old_val
            
    return None

def construct_lift(base_nodes, base_edges, voltages, m):
    # Build the lifted graph
    # Nodes: (base_node, voltage)
    # Edges: ((u, v), (v, v + volt[edge]))
    
    lifted_nodes = base_nodes * m
    lifted_edges = []
    
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    for (u, v) in base_edges:
        idx = edge_index[(u, v)]
        vol = voltages[idx]
        for i in range(m):
            # u at layer i connects to v at layer (i + vol) % m
            n_u = u * m + i
            n_v = v * m + ((i + vol) % m)
            lifted_edges.append((n_u, n_v))
            
    return lifted_nodes, lifted_edges

# Fallback: LUW Construction from seed
def construct_luw(k, g):
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    if k == 4 and g == 10:
        # Try Voltage Lift
        base_nodes, base_edges = get_pg23_graph()
        # Find cycles
        # 6-cycles and 8-cycles
        cycles_6 = find_cycles(base_nodes, base_edges, 6)
        cycles_8 = find_cycles(base_nodes, base_edges, 8)
        all_cycles = cycles_6 + cycles_8
        
        # Try increasing m
        # Moore bound suggests m >= 27
        for m in range(27, 60):
            if time.time() > deadline:
                break
            # Solve CSP
            # Use a tighter time limit per m
            time_per_m = max(1, (deadline - time.time()) / (60 - m))
            voltages = solve_csp(base_edges, all_cycles, m, time_limit=time_per_m)
            if voltages:
                n, edges = construct_lift(base_nodes, base_edges, voltages, m)
                return n, edges
                
    # Fallback to LUW
    return construct_luw(k, g)
