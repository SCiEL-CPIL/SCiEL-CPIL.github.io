"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for k=4, g=10: 161. Best known is 320.
    
    # Try voltage lifts with various base graphs and m values
    base_graphs = []
    
    # Base graph 1: 8-cycle with opposite chords (4-regular, 8 vertices)
    base1_edges = [(i, (i+1)%8) for i in range(8)] + [(i, i+4) for i in range(4)]
    base_graphs.append((8, base1_edges))
    
    # Base graph 2: Complete bipartite K(4,4) minus perfect matching (4-regular, 8 vertices)
    base3_edges = []
    for i in range(4):
        for j in range(4):
            if i != j:
                base3_edges.append((i, 4+j))
    base_graphs.append((8, base3_edges))
    
    # Base graph 4: 12-vertex 4-regular graph (dodecahedron-like structure)
    base4_edges = []
    # Two 6-cycles with connections
    for i in range(6):
        base4_edges.append((i, (i+1)%6))
        base4_edges.append((6+i, (6+(i+1))%6))
    # Connect the two cycles
    for i in range(6):
        base4_edges.append((i, 6+i))
    # Add extra edges to make 4-regular (need 2 more per vertex)
    base4_edges.append((0, 3))
    base4_edges.append((1, 4))
    base4_edges.append((2, 5))
    base4_edges.append((6, 9))
    base4_edges.append((7, 10))
    base4_edges.append((8, 11))
    base_graphs.append((12, base4_edges))
    
    # Base graph 5: Another 8-vertex 4-regular graph
    base5_edges = [(i, (i+1)%8) for i in range(8)] + [(i, (i+2)%8) for i in range(4)]
    base_graphs.append((8, base5_edges))
    
    # Base graph 6: 10-vertex 4-regular graph
    base6_edges = []
    for i in range(10):
        base6_edges.append((i, (i+1)%10))
        base6_edges.append((i, (i+2)%10))
    # Remove duplicates and ensure 4-regular
    seen = set()
    unique_base6 = []
    for e in base6_edges:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key)
            unique_base6.append(key)
    base_graphs.append((10, unique_base6))
    
    # Try different m values (voltage group size)
    m_candidates = []
    for base_n, base_edges in base_graphs:
        min_m = max(2, (161 + base_n - 1) // base_n)
        for m in range(min_m, min_m + 15):
            m_candidates.append((base_n, base_edges, m))
    
    # Sort by resulting n = m * base_n
    m_candidates.sort(key=lambda x: x[2] * x[0])
    
    # For each candidate, try to find valid voltage assignments
    # Voltage assignment: each edge gets a voltage in Z_m
    # The lift has vertex (v, t) where v in base, t in Z_m
    # Edge (u,v) with voltage a connects (u,t) to (v, t+a)
    
    def check_lift_girth_fast(base_n, base_edges, m, voltages, g, deadline):
        """Check if the lift has girth >= g using non-backtracking walk enumeration with early termination."""
        n_lift = base_n * m
        
        # Build adjacency list for lift
        adj = [[] for _ in range(n_lift)]
        for (u, v), vol in zip(base_edges, voltages):
            for t in range(m):
                adj[u * m + t].append(v * m + (t + vol) % m)
                adj[v * m + t].append(u * m + (t - vol) % m)
        
        min_cycle = float('inf')
        
        # Check from all base vertices
        sample_vertices = list(range(base_n))
        
        for sv in sample_vertices:
            if time.time() > deadline:
                return min_cycle
            
            visited = {}  # (current, prev) -> distance
            q = deque()
            
            sv_lift = sv * m
            for nb in adj[sv_lift]:
                key = (nb, sv_lift)
                visited[key] = 1
                q.append(key)
            
            while q:
                if time.time() > deadline:
                    return min_cycle
                cur, prev = q.popleft()
                dist = visited[(cur, prev)]
                
                if dist >= g:
                    continue
                
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    
                    if (nb, cur) in visited:
                        cycle_len = dist + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return min_cycle
                    else:
                        visited[(nb, cur)] = dist + 1
                        q.append((nb, cur))
        
        return min_cycle
    
    def random_voltages(num_edges, m):
        return [random.randint(0, m - 1) for _ in range(num_edges)]
    
    def build_lift_edges(base_n, base_edges, m, voltages, k):
        """Build the lift edge list and verify it's valid."""
        n_lift = base_n * m
        edges = []
        for (u, v), vol in zip(base_edges, voltages):
            for t in range(m):
                edges.append((u * m + t, v * m + (t + vol) % m))
        
        edge_set = set()
        unique_edges = []
        for e in edges:
            key = (min(e), max(e))
            if key not in edge_set:
                edge_set.add(key)
                unique_edges.append((key[0], key[1]))
        
        if len(unique_edges) == n_lift * k // 2:
            return unique_edges
        return None
    
    # Simulated annealing for each candidate
    for base_n, base_edges, m in m_candidates:
        if time.time() > deadline - 30:
            break
        
        n_lift = base_n * m
        if n_lift >= best_n:
            continue
        
        num_edges = len(base_edges)
        
        # Try multiple random restarts with limited SA iterations
        max_restarts = 4
        max_iterations = 3000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 25:
                break
            
            current_voltages = random_voltages(num_edges, m)
            
            girth = check_lift_girth_fast(base_n, base_edges, m, current_voltages, g, deadline - 20)
            
            if girth >= g:
                unique_edges = build_lift_edges(base_n, base_edges, m, current_voltages, k)
                if unique_edges is not None:
                    best_n = n_lift
                    best_edges = unique_edges
                    break
            
            # Simulated annealing with better cooling schedule
            temperature = 1.0
            cooling = 0.999
            for it in range(max_iterations):
                if time.time() > deadline - 20:
                    break
                
                edge_idx = random.randint(0, num_edges - 1)
                old_vol = current_voltages[edge_idx]
                new_vol = random.randint(0, m - 1)
                if new_vol == old_vol:
                    continue
                
                current_voltages[edge_idx] = new_vol
                
                girth_new = check_lift_girth_fast(base_n, base_edges, m, current_voltages, g, deadline - 15)
                
                if girth_new >= g:
                    unique_edges = build_lift_edges(base_n, base_edges, m, current_voltages, k)
                    if unique_edges is not None and n_lift < best_n:
                        best_n = n_lift
                        best_edges = unique_edges
                        temperature *= cooling
                else:
                    # Accept with probability based on temperature
                    if random.random() < temperature:
                        temperature *= cooling
                    else:
                        current_voltages[edge_idx] = old_vol
                        temperature *= cooling
            
            if best_edges is not None and best_n == n_lift:
                break
    
    # If voltage lift didn't work, fall back to LUW construction
    if best_edges is None:
        # LUW construction
        # For k=4, g=10: use d=5 (odd, >= g-5=5)
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(d, 3)
        q = k
        
        if q == 4:
            # Use Z_2 x Z_2 arithmetic
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        # LUW rules for degree q=4, girth g
        # rule_a[c], rule_b[c] for c in 2..d
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        
        # Standard LUW construction rules
        for c in range(2, d + 1):
            if c == 2:
                rule_a[c] = 1
                rule_b[c] = 1
            elif c == 3:
                rule_a[c] = 2
                rule_b[c] = 1
            elif c == 4:
                rule_a[c] = 1
                rule_b[c] = 2
            else:
                # For c >= 5, use periodic pattern
                r = (c - 5) % 4
                c0 = c - r
                if r == 0:
                    rule_a[c] = 1
                    rule_b[c] = c0 - 2
                elif r == 1:
                    rule_a[c] = c0 - 1
                    rule_b[c] = 1
                elif r == 2:
                    rule_a[c] = c0
                    rule_b[c] = 1
                else:
                    rule_a[c] = 1
                    rule_b[c] = c0 + 1
        
        def compute_neighbours_fast(vertex_key):
            side, x = vertex_key
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            return result
        
        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        head = 0
        
        while head < len(order):
            if time.time() > deadline:
                break
            vertex = order[head]
            head += 1
            for other in compute_neighbours_fast(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
