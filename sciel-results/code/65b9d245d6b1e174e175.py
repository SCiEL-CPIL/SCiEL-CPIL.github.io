import time
import sys
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    # --- Helper: Check girth using BFS from every vertex (optimized) ---
    def check_girth_fast(adj, n, g):
        # For girth g, we just need to ensure no cycle of length < g.
        # A cycle of length L exists iff there is a non-backtracking walk of length L-1 
        # that returns to the start? No, simpler: 
        # If we do BFS from every node, the first time we see a visited node 
        # (other than parent) gives a cycle. 
        # However, for large n, BFS from every node is O(n^2).
        # For n ~ 500-1000, n^2 is 250k-1M operations, which is fast in Python if optimized.
        
        # Optimization: We only need to check up to depth g//2.
        # If no cycle is found within depth g//2 from any node, the girth is >= g.
        # Actually, if girth >= g, then the ball of radius g//2 around any node is a tree.
        # So we can just check if the ball of radius g//2 is a tree.
        
        half_g = g // 2
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a visited node. 
                        # If v is not the parent of u, we found a cycle.
                        # The length of the cycle is dist[u] + dist[v] + 1?
                        # No, this is not strictly true for arbitrary visited nodes in BFS.
                        # The shortest cycle involving edge (u,v) where v is visited 
                        # is found by LCA(u,v) in the BFS tree.
                        # However, a simpler check for "no cycle shorter than g":
                        # If the graph has girth >= g, then the subgraph induced by 
                        # nodes at distance <= g//2 - 1 from start must be a tree.
                        # So if we encounter a visited node v (v != parent[u]) 
                        # such that dist[u] + dist[v] + 1 < g, then we have a short cycle.
                        # Actually, if dist[u] + dist[v] + 1 < g, the cycle length is at most that.
                        # But we need to be careful. The standard BFS girth check:
                        # The girth is the minimum over all edges (u,v) not in the BFS tree 
                        # of dist[u] + dist[v] + 1 (where dist is from a specific root).
                        # This only finds cycles containing the root? No.
                        # To find global girth, we need to run BFS from every node?
                        # Or use the property: Girth >= g iff for every edge (u,v), 
                        # the distance between u and v in the graph with (u,v) removed is >= g-1.
                        # That's too expensive.
                        
                        # Let's use the non-backtracking walk method or simple BFS tree check.
                        # If we perform BFS from 'start', and we find an edge (u,v) where 
                        # parent[v] != u and parent[u] != v, then we have a cycle.
                        # The length of this cycle is dist[u] + dist[v] + 1 ONLY IF the paths 
                        # from start to u and start to v are disjoint except at start.
                        # This is true for the BFS tree. The cycle formed by tree paths + edge (u,v)
                        # has length dist[u] + dist[v] + 1.
                        # If this length < g, we fail.
                        # If dist[u] + dist[v] + 1 >= g for all such edges, does it mean girth >= g?
                        # Yes, because any cycle must have an edge (u,v) where u and v are in the BFS tree
                        # and the cycle length is exactly dist[u] + dist[v] + 1 (if the LCA is start).
                        # Wait, if the LCA is not start, the cycle length is smaller.
                        # dist[u] + dist[v] + 1 is the length of the cycle formed by the two tree paths 
                        # and the edge. This cycle contains the LCA. 
                        # If the LCA is not start, the cycle is actually a sub-cycle of this one?
                        # No, the cycle formed by tree paths to u and v and edge (u,v) is a simple cycle 
                        # ONLY if the tree paths are disjoint except at LCA.
                        # The length of this simple cycle is (dist[u] - dist[lca]) + (dist[v] - dist[lca]) + 1.
                        # This is <= dist[u] + dist[v] + 1.
                        # So if dist[u] + dist[v] + 1 >= g, it doesn't guarantee the actual cycle is >= g.
                        
                        # Correct approach for girth >= g:
                        # The ball of radius (g-1)//2 around every vertex must be a tree.
                        # So we just need to check if BFS from 'start' encounters any 
                        # "cross edge" within depth (g-1)//2.
                        # If g is even, say g=16. Radius = 7.
                        # If we find an edge (u,v) where dist[u] <= 7, dist[v] <= 7, 
                        # and v is not parent[u] and u is not parent[v], then there is a cycle 
                        # of length <= dist[u] + dist[v] + 1 <= 15.
                        # So if we find such an edge within radius (g-1)//2, girth < g.
                        
                        max_depth_allowed = (g - 1) // 2
                        if dist[u] <= max_depth_allowed and dist[v] <= max_depth_allowed:
                            # Check if it's a tree edge
                            if parent[u] == v or parent[v] == u:
                                continue
                            # It's a cross edge within the radius.
                            # The cycle length is at most dist[u] + dist[v] + 1.
                            # Since both <= max_depth_allowed, sum + 1 <= 2*max_depth_allowed + 1.
                            # If g is even, 2*(g-1)//2 + 1 = g-1 < g.
                            # If g is odd, 2*(g-1)//2 + 1 = g-2 < g.
                            # So any cross edge within this radius implies girth < g.
                            return False
        return True

    # --- Strategy: Voltage Graph Lifting ---
    # We construct a base graph with voltage labels.
    # The lift is a graph where vertices are (base_vertex, group_element).
    # Edges are lifted according to the voltage.
    # We want the lift to be 3-regular and have girth >= g.
    
    # For k=3, a simple base is a cycle of length L with voltages, 
    # or a tree-like structure.
    # A common construction for cages is using "voltage graphs" on a tree or small graph.
    # Let's try a base graph that is a single vertex with 3 self-loops? No, simple graph required.
    # Base graph must be simple.
    # Let's use a base graph that is a path or cycle.
    
    # Inspiration 2 uses `_construct_petersen_lift`. 
    # The Petersen graph is 3-regular, 10 vertices, girth 5.
    # If we lift the Petersen graph with a cyclic group of order m, 
    # the result is 3-regular, 10*m vertices.
    # The girth of the lift depends on the voltages.
    
    # Let's implement a generic voltage graph lifter.
    # Base graph: G_base = (V_base, E_base).
    # Voltage assignment: phi: E_base -> Z_m.
    # Lifted graph G: V = V_base x Z_m.
    # Edge (u, i) connected to (v, i + phi(u,v)) if (u,v) in E_base.
    
    # To ensure 3-regularity, G_base must be 3-regular.
    # To maximize girth, we want the voltages to be "large" in some sense, 
    # i.e., the sum of voltages around any short cycle in the base must be non-zero 
    # (or have large order) in Z_m.
    
    # Let's try lifting the Petersen graph.
    # Petersen graph vertices 0..9.
    # Edges:
    # Outer cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Inner star: (5,7), (7,9), (9,6), (6,8), (8,5)
    # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
    
    # We will search for a voltage assignment (phi) and group size m 
    # such that the lift has girth >= g.
    
    # For k=3, g=16.
    # If we use m=100, n=1000.
    # If we use m=50, n=500.
    # We want n as small as possible.
    
    # Heuristic: Random voltages.
    # Check girth of the lift.
    
    def get_petersen_edges():
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return edges

    def build_lift(base_edges, m, voltages):
        # base_edges: list of (u, v)
        # voltages: dict or list corresponding to edges? 
        # Let's use a list of voltages corresponding to base_edges order.
        n_base = 10
        n = n_base * m
        adj = [[] for _ in range(n)]
        
        for idx, (u, v) in enumerate(base_edges):
            vol = voltages[idx] % m
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + vol) % m
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
        
        # Check for simple graph (no multiple edges in lift)
        # Since base is simple and voltages are fixed, 
        # could we have multiple edges?
        # If there are two edges (u,v) and (u,v) in base with different voltages, 
        # we would have multiple edges. But base is simple.
        # Could (u,i) connect to (v,j) via two different base edges?
        # Only if base has multiple edges between u and v. It doesn't.
        # So the lift is simple.
        
        return adj, n

    # Fallback: LUW construction (Baseline)
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

    # Get baseline
    baseline_n, baseline_edges = construct_luw(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    # Try voltage lifting on Petersen graph
    if k == 3:
        base_edges = get_petersen_edges()
        n_base_edges = len(base_edges)
        
        # Try different group sizes m
        # n = 10 * m
        # We want n < baseline_n.
        # If baseline_n is 936, we can try m up to 93.
        # Smaller m is better.
        # Let's try m from 20 up to baseline_n // 10.
        
        # To speed up, we can use a heuristic for voltages.
        # Random voltages are a good start.
        
        max_m = baseline_n // 10
        if max_m < 20:
            max_m = 20
            
        # We'll search for a valid m and voltages.
        # Start with small m, increase if not found? 
        # Or start with large m (easier to satisfy girth) and reduce?
        # Girth constraint is harder to satisfy for small m (more collisions).
        # So let's try large m first to find *a* solution, then try to minimize m?
        # Actually, we want the smallest n. So we should try small m first.
        # But small m is harder. 
        # Let's try a few random attempts for each m.
        
        # Time budget for search
        search_deadline = deadline - 5.0
        
        # We'll iterate m from 10 to max_m
        # But 10 to 90 is 80 values.
        # For each m, we try R random voltage assignments.
        # Girth check is O(n^2) roughly? No, O(n * g).
        # n=10*m. g=16.
        # BFS from each node: n * (1 + 2 + ... + 2^7) ~ n * 256.
        # For n=100, 25600 ops. Very fast.
        # For n=900, 230400 ops. Still fast.
        
        # So we can do many checks.
        
        for m in range(10, max_m + 1):
            if time.time() > search_deadline:
                break
                
            # Try a few random voltage assignments
            # Heuristic: Use random integers 1..m-1.
            attempts = 5
            for _ in range(attempts):
                if time.time() > search_deadline:
                    break
                
                voltages = [random.randint(1, m-1) for _ in range(n_base_edges)]
                adj, n = build_lift(base_edges, m, voltages)
                
                if check_girth_fast(adj, n, g):
                    # Found a valid graph with n vertices
                    if n < best_n:
                        # Convert adj to edge list
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_n = n
                        best_edges = edges
                        # We found a smaller one, continue searching for even smaller?
                        # Yes, but we are iterating m upwards. 
                        # If we found one at m, smaller m might work too but we started at 10.
                        # Wait, we are iterating m from 10 upwards.
                        # So if we find one at m=50, we might find one at m=20 later? 
                        # No, we already passed m=20.
                        # So we should iterate m from small to large, and break if we find one?
                        # No, a smaller m gives smaller n.
                        # So we should iterate m from small to large, and for each m, 
                        # try to find a valid voltage assignment.
                        # If we find one, that's the best for that m.
                        # Since m is increasing, n is increasing.
                        # So the first valid m we find is the optimal m?
                        # Not necessarily, because for m=10, we might fail all random attempts, 
                        # but m=11 might succeed.
                        # But m=10 gives n=100, m=11 gives n=110.
                        # So we want the smallest m that *can* work.
                        # We don't know if m=10 *can* work without exhaustive search.
                        # So we try random attempts. If we fail, we move to m=11.
                        # This might miss a valid m=10 graph.
                        # To be safer, we can do more attempts for small m.
                        
                        # Since we found a valid graph, we can update best.
                        # We continue the loop to see if a smaller m was possible? 
                        # No, we are going up. 
                        # But we might have missed a smaller m.
                        # Let's just keep the best found so far.
                        
                        pass # Continue searching for potentially better (smaller n) 
                        # But since m is increasing, n is increasing.
                        # So the first success is the best?
                        # Only if we are sure we didn't miss a smaller m.
                        # Given random attempts, it's possible.
                        # But 936 is the baseline. Any m < 94 is an improvement.
                        # So we can stop at the first success?
                        # Yes, because any success with m < 94 is better than baseline.
                        # And smaller m is better.
                        # So we break out of the m loop if we find a success?
                        # No, we should break if we find a success, 
                        # because we are iterating m in increasing order.
                        # The first m that yields a valid graph gives the smallest n 
                        # among the m's we've tried.
                        # But we might have skipped a valid m=10 because of bad luck.
                        # However, trying m=10 with 5 attempts is weak.
                        # Let's do more attempts for small m.
                        
                        # For now, let's just update best and continue to see if we can find 
                        # an even smaller one? No, m is increasing.
                        # So the first valid m found is the smallest m found.
                        # We can break.
                        
                        break # Break out of attempts loop
                        
            if best_n < baseline_n:
                # We found an improvement. 
                # Since m is increasing, we can't find a smaller n with larger m.
                # So we can stop.
                break

    return best_n, best_edges
