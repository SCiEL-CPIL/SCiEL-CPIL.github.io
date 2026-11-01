import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # safety margin
    
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

    if k == 3 and g == 16:
        # Petersen graph edges (10 vertices, 15 edges)
        # Outer pentagon: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
        ]
        petersen_adj = [[] for _ in range(10)]
        for u, v in petersen_edges:
            petersen_adj[u].append(v)
            petersen_adj[v].append(u)
        
        def check_girth_lift(m, voltages):
            """Check if the voltage lift has girth >= g using BFS with voltage tracking.
            voltages: list of 15 integers mod m, one per Petersen edge.
            Returns True if girth >= g."""
            # For each base vertex, do BFS tracking (vertex, voltage)
            # A cycle of length L < g exists iff there's a path from (v,0) back to (v,0) with length < g
            # Actually, we need to check: is there any cycle of length < g?
            # A cycle in the lift corresponds to a closed walk in the base with total voltage = 0 mod m
            # We check all closed walks of length < g in the Petersen graph and see if any has voltage sum = 0
            
            # Better: BFS from each (start_vertex, 0) and check if we can return to (start_vertex, 0) in < g steps
            # But that's expensive. Instead, enumerate all closed walks of length up to g-1.
            
            # For g=16, we need to check lengths 3 to 15 (Petersen has no 4-cycles, min cycle is 5)
            # Actually Petersen has girth 5, so we check lengths 5 to 15
            
            # Enumerate closed walks: start at each vertex, walk up to g-1 steps, check if we return
            # with total voltage 0 mod m
            
            # This is expensive but m is small (52-100) and we can prune
            # For each start vertex s, do DFS/BFS tracking (current_vertex, current_voltage, length)
            # We want to find if (s, 0) is reachable from (s, 0) in exactly L steps for some 5 <= L <= 15
            
            # More efficient: for each pair (u, v) and length L, compute the set of possible voltages
            # of walks from u to v of length L. If 0 is in the set for u=v and L < g, fail.
            
            # Let's do it step by step.
            # reachable[L][u][v] = set of voltages achievable by a walk of length L from u to v
            
            # Initialize: reachable[0][u][u] = {0} for all u
            # reachable[L+1][u][w] = union over v adjacent to w of { (vol + volt(edge(v,w))) % m for vol in reachable[L][u][v] }
            
            # For g=16, we need L up to 15. 10 vertices, 15 steps.
            # reachable[L] is a 10x10 matrix of sets.
            
            # This could be memory intensive but let's try for small m.
            
            # Actually, for checking girth, we only need to check if 0 is in reachable[L][s][s] for any s and L < g.
            # We can do this iteratively.
            
            # Let's use a different approach: for each start vertex s, do BFS in the product space
            # (vertex, voltage) and check if we can return to (s, 0) in < g steps.
            # The state space is 10 * m. For m=100, that's 1000 states. BFS for 15 steps is fine.
            
            for s in range(10):
                # BFS from (s, 0)
                # visited: dict (vertex, voltage) -> min length
                # We want to know if (s, 0) is reachable in L steps for 5 <= L <= 15
                # Actually, (s,0) is the start, so we need a non-trivial return.
                # We track the length and check if we hit (s, 0) again at length L where 5 <= L <= 15.
                
                # Use BFS layer by layer
                # current_layer: set of (vertex, voltage) at current distance
                # We start at distance 0: {(s, 0)}
                # At distance L, if (s, 0) is in the layer and L >= 5 and L < g, we have a cycle of length L.
                
                # But we need to avoid counting the trivial path of length 0.
                # So we check layers 1 to g-1, and if (s,0) appears at layer L where L >= 5, fail.
                
                # Actually, the first time we return to (s,0) is the shortest cycle through s.
                # If that length is < g, we fail.
                
                visited = set()
                current = {(s, 0)}
                visited.add((s, 0))
                
                for L in range(1, g):
                    next_layer = set()
                    for (v, vol) in current:
                        for nb in petersen_adj[v]:
                            # Find the voltage of edge (v, nb)
                            # Need to map edge to voltage index
                            edge_idx = None
                            for i, (a, b) in enumerate(petersen_edges):
                                if (a == v and b == nb) or (a == nb and b == v):
                                    edge_idx = i
                                    break
                            if edge_idx is None:
                                continue
                            new_vol = (vol + voltages[edge_idx]) % m
                            state = (nb, new_vol)
                            if state not in visited:
                                visited.add(state)
                                next_layer.add(state)
                    
                    # Check if (s, 0) is in next_layer
                    if (s, 0) in next_layer:
                        # We found a cycle of length L
                        if L >= 5 and L < g:
                            return False
                        # If L < 5, Petersen doesn't have such cycles, so this shouldn't happen
                        # Actually, Petersen has no 3 or 4 cycles, so L >= 5 always for a real cycle
                        # But (s,0) could be reached via a non-simple walk? No, BFS finds shortest path.
                        # The shortest return to (s,0) is the girth contribution from s.
                        # If it's < g, we fail.
                        return False
                    
                    current = next_layer
                    if not current:
                        break
            
            return True
        
        # Try systematic voltage assignments for various m
        best_n = 1024  # LUW gives 1024 for (3,16)
        best_edges = None
        
        # Moore bound for (3, 16): 2 * (1 + 2 + 4 + 8 + 16 + 32) = 2 * 63 = 126
        # Actually, Moore bound for cubic graph of girth g:
        # For even girth g = 2r: n >= 2 * (3^r - 1) / 2 = 3^r - 1... 
        # Let me recall: for k-regular graph of girth g:
        # If g is odd, g = 2r+1: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # If g is even, g = 2r: n >= 2 * sum_{i=0}^{r-1} (k-1)^i
        # For k=3, g=16=2*8: n >= 2 * sum_{i=0}^{7} 2^i = 2 * (2^8 - 1) = 2 * 255 = 510
        # So Moore bound is 510.
        
        # The Petersen lift has n = 10 * m vertices.
        # So we need 10 * m >= 510, i.e., m >= 51.
        # Try m from 51 upward.
        
        # For each m, try different voltage assignments.
        # Systematic approach: use a random seed and try many random assignments.
        
        random.seed(42)
        
        for m in range(51, 101):
            if time.time() - start_time > deadline - 10:
                break
            
            # Try multiple random voltage assignments
            max_trials = 200
            found = False
            
            for trial in range(max_trials):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Random voltages in [0, m)
                voltages = [random.randint(0, m - 1) for _ in range(15)]
                
                # Quick check: ensure no self-loop in the lift
                # A self-loop occurs if for some edge (u,v) with u != v in base,
                # the lift has an edge from (u, a) to (v, b) where u = v... no, u != v in base.
                # Self-loop in lift: (u, a) connected to (u, a). This happens if there's a base edge (u, v) with u = v, which doesn't exist.
                # Or if the voltage assignment causes (u, a) to connect to itself via some path? No, edges are direct.
                # Actually, in the lift, vertex (u, a) connects to (v, (a + vol(u,v)) % m) for each base neighbor v.
                # This is a self-loop if v = u and (a + vol) % m = a, i.e., vol = 0. But u != v for all Petersen edges.
                # So no self-loops.
                
                # Check for multi-edges: two different base edges could give the same lift edge.
                # Edge (u,v) with voltage vol1 and edge (u,w) with voltage vol2 could give the same lift edge
                # if v = w and vol1 = vol2, but v != w for different edges from u.
                # Or edge (u,v) and edge (u',v') could give the same lift edge if u = u' and v = v' (same base edge)
                # or u = v' and v = u' (reverse). Since we treat edges as undirected, (u,v) and (v,u) are the same.
                # In our lift, for base edge (u,v) with voltage vol, we add lift edges:
                # (u, a) - (v, (a + vol) % m) for all a.
                # And (v, b) - (u, (b - vol) % m) for all b, which is the same set of edges.
                # So no multi-edges from the same base edge.
                # Different base edges: (u,v) and (u,w) with v != w give different lift edges since the second coordinate differs.
                # (u,v) and (w,x) with {u,v} != {w,x}: could they give the same lift edge?
                # Lift edge is ((u,a), (v, (a+vol1)%m)) = ((w,c), (x, (c+vol2)%m))
                # This requires u = w, v = x, a = c, (a+vol1)%m = (c+vol2)%m, so vol1 = vol2.
                # Or u = x, v = w, a = (c+vol2)%m, (a+vol1)%m = c.
                # In the first case, same base edge. In the second, same base edge reversed.
                # So no multi-edges.
                
                if check_girth_lift(m, voltages):
                    # Build the lift
                    n = 10 * m
                    edges = []
                    for i, (u, v) in enumerate(petersen_edges):
                        vol = voltages[i]
                        for a in range(m):
                            b = (a + vol) % m
                            u_idx = u * m + a
                            v_idx = v * m + b
                            if u_idx < v_idx:
                                edges.append((u_idx, v_idx))
                            else:
                                edges.append((v_idx, u_idx))
                    
                    # Verify: check that each vertex has degree 3
                    # And check girth by BFS (expensive, but we already checked via voltage)
                    
                    if len(edges) < best_n or best_edges is None:
                        best_n = n
                        best_edges = edges
                        found = True
                        break
            
            if found:
                # Found a valid lift, check if we can do better with larger m? No, larger m means more vertices.
                # But maybe a different m in the range gives fewer? We're going up, so first found is smallest.
                # Actually, we should check all m and keep the best.
                pass
        
        if best_edges is not None and best_n < 1024:
            return best_n, best_edges
        else:
            return luw_construct(k, g)
    
    return luw_construct(k, g)
