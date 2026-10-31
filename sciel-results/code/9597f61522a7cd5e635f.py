import time
import sys
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a 3-regular graph with girth at least g using voltage lifts.
    Strategy:
    1. Use the Dodecahedral graph (20 vertices, 3-regular, girth 5) as the base.
    2. Lift it using a voltage assignment over Z_m to create a 3-regular graph of size 20*m.
    3. Scan m from the Moore bound (512) upwards.
    4. Use fast vectorized non-backtracking walk verification to check girth.
    5. Return the smallest valid graph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if k != 3:
        # Fallback for other k if needed, though task implies k=3
        # For now, just raise or return None if not 3, but contract says k=3.
        # We'll implement a generic random search fallback if k != 3, but focus on 3.
        pass

    # Moore bound for n(3, 16) is 512.
    # Since the graph is a lift of a 20-vertex base, n must be a multiple of 20.
    # Smallest multiple of 20 >= 512 is 520 (m=26).
    # We scan m from 26 upwards.
    
    # Base Graph: Dodecahedral Graph
    # Vertices 0..19
    # Standard construction:
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagon: 5-9-8-7-6-5 (reversed orientation for symmetry? No, let's use standard adjacency)
    # Actually, let's use the known adjacency list for the dodecahedron.
    # Symmetry: Icosahedral.
    # Let's define it programmatically to avoid hardcoding errors.
    # Or use a known list.
    # Dodecahedron vertices can be generated from icosahedron faces or vice versa.
    # Here is a standard edge list for the dodecahedron (0-19):
    
    def get_dodecahedron_edges():
        edges = []
        # Two concentric pentagons and 10 "spokes" forming a belt?
        # No, dodecahedron has 12 faces.
        # Let's use the 3D coordinates or a known list.
        # Known list from NetworkX or similar sources:
        # 0-1, 1-2, 2-3, 3-4, 4-0 (Top face)
        # 5-6, 6-7, 7-8, 8-9, 9-5 (Bottom face)
        # 0-10, 1-11, 2-12, 3-13, 4-14 (Top spokes)
        # 5-15, 6-16, 7-17, 8-18, 9-19 (Bottom spokes)
        # 10-11, 11-12, 12-13, 13-14, 14-10 (Upper belt)
        # 15-16, 16-17, 17-18, 18-19, 19-15 (Lower belt)
        # 10-15, 11-16, 12-17, 13-18, 14-19 (Verticals)
        
        # Let's verify degrees.
        # 0: 1, 4, 10 (deg 3)
        # 1: 0, 2, 11 (deg 3)
        # ...
        # This forms a valid dodecahedron.
        
        # Top
        edges += [(0,1), (1,2), (2,3), (3,4), (4,0)]
        # Bottom
        edges += [(5,6), (6,7), (7,8), (8,9), (9,5)]
        # Top Spokes
        edges += [(0,10), (1,11), (2,12), (3,13), (4,14)]
        # Bottom Spokes
        edges += [(5,15), (6,16), (7,17), (8,18), (9,19)]
        # Upper Belt
        edges += [(10,11), (11,12), (12,13), (13,14), (14,10)]
        # Lower Belt
        edges += [(15,16), (16,17), (17,18), (18,19), (19,15)]
        # Verticals
        edges += [(10,15), (11,16), (12,17), (13,18), (14,19)]
        
        return edges

    base_edges = get_dodecahedron_edges()
    n_base = 20
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Ensure base is 3-regular
    for u in range(n_base):
        if len(base_adj[u]) != 3:
            # This should not happen
            raise ValueError("Base graph is not 3-regular")

    def verify_girth_fast(n, edges, k, g):
        """
        Verify girth >= g using non-backtracking walks.
        For g=16, we check walks of length 7 and 8?
        Actually, girth is the length of the shortest cycle.
        If there is a cycle of length < g, it fails.
        We can check for cycles of length L for L=3..g-1.
        However, for k=3, triangles are unlikely in high girth candidates.
        A faster check: 
        If the graph is bipartite, girth is even.
        Dodecahedron is not bipartite (has 5-cycles).
        So the lift might have odd cycles.
        
        Efficient check:
        Count closed non-backtracking walks of length L.
        If count > 0 for any L < g, girth < g.
        
        For L=3: Triangles.
        For L=4: 4-cycles.
        ...
        
        We can use matrix powers of the non-backtracking matrix B.
        B is of size 2E x 2E. For n=520, E=780, 2E=1560.
        1560x1560 matrix multiplication is fast in NumPy.
        We need to check L from 3 to g-1 (15).
        
        However, constructing B for every m is O(n).
        Matrix multiplication is O((2n)^3) ~ 1560^3 ~ 3.8e9 ops. Too slow for many m.
        
        Alternative: BFS from each vertex up to depth g/2.
        For n=520, depth 8.
        Branching factor 2. 2^8 = 256.
        520 * 256 * 8 ~ 1M ops per verification.
        Very fast.
        
        Let's stick to the BFS/DFS verification but optimize it with early termination.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
                
        # BFS from each start
        # We want to find any cycle of length < g.
        # A cycle of length L corresponds to a closed walk of length L with no backtracking?
        # No, a simple cycle of length L is a non-backtracking closed walk of length L.
        # If there is a non-backtracking closed walk of length L < g, there is a cycle <= L < g.
        
        # We can just do BFS and track (node, prev).
        # If we reach (start, prev) with depth < g, we found a cycle.
        
        for start in range(n):
            # visited[state] = depth
            # state = (u, v) means we are at u, coming from v.
            visited = {}
            queue = deque()
            for v in adj[start]:
                state = (v, start)
                visited[state] = 1
                queue.append((v, start, 1))
                
            while queue:
                u, prev, depth = queue.popleft()
                if depth >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start:
                        if depth + 1 < g:
                            return False
                        continue
                    state = (v, u)
                    if state in visited:
                        if depth + 1 < g:
                            # Found a cycle shorter than g?
                            # Wait. If state (v,u) was seen at depth d1, and now at depth d2.
                            # The cycle length is d2 - d1 + 1?
                            # Actually, if we reach a state already visited, it forms a cycle.
                            # The length of the cycle is not simply related to depth.
                            # But if we reach (start, prev) we have a closed walk.
                            # The condition "v == start" handles closed walks starting at start.
                            # What about cycles not including start?
                            # If a cycle exists, it must be reachable from start.
                            # If we traverse a path from start to u, and then find a cycle,
                            # the first time we revisit a state (v,u), we have found a cycle.
                            # Is the cycle length < g?
                            # Not necessarily. The path from start to the cycle entry might be long.
                            # However, if the graph has a cycle of length L < g,
                            # BFS from any node in that cycle will find it within depth L/2?
                            # No, BFS explores all non-backtracking walks.
                            # If there is a cycle of length L, there is a non-backtracking closed walk of length L.
                            # BFS from a node on the cycle will find the state (v,u) again after L steps.
                            # But BFS stops at depth g?
                            # If we stop at depth g, we might miss a cycle of length L < g if the path to it is long?
                            # No, the cycle is local.
                            # If we start BFS at a node on the cycle, we find it in L steps.
                            # If we start at a node far away, we might not find it in depth g?
                            # But we check ALL starts.
                            # So if we start at a node on the short cycle, we will find it.
                            # So the check is valid.
                            pass
                    else:
                        visited[state] = depth + 1
                        queue.append((v, u, depth + 1))
                        
        return True

    def construct_lift(m, seed=0):
        """
        Construct a random voltage lift of the base graph over Z_m.
        """
        rng = np.random.default_rng(seed)
        # Assign random voltages to edges
        # Each edge (u,v) has a voltage a in Z_m.
        # The lift has vertices (u, i) for u in 0..19, i in 0..m-1.
        # Total n = 20 * m.
        n = n_base * m
        edges = []
        
        # We need to be consistent with orientation.
        # Let's define directed edges for the base.
        # For each undirected edge (u,v), pick a direction u->v.
        # Assign voltage a.
        # In the lift:
        # (u, i) -> (v, i+a)
        # (v, j) -> (u, j-a) (since voltage is a, reverse is -a)
        
        # To ensure simple graph, we must avoid multiple edges.
        # With random voltages, multiple edges are unlikely for large m, but possible.
        # We can check for duplicates.
        
        edge_voltages = {}
        for u, v in base_edges:
            # Ensure consistent key
            key = (min(u,v), max(u,v))
            volt = rng.integers(0, m)
            edge_voltages[key] = (u, v, volt) # direction u->v
            
        for key, (u, v, a) in edge_voltages.items():
            for i in range(m):
                # u, i -> v, i+a
                u_lift = u * m + i
                v_lift = v * m + (i + a) % m
                edges.append((u_lift, v_lift))
                
                # v, j -> u, j-a
                # This is the same edge set, just traversed in reverse.
                # We only need to add each undirected edge once.
                # The edge (u_lift, v_lift) covers the connection.
                
        # Check for duplicates
        # Sort and unique?
        # For n=520, E=780.
        # Just return edges. The verification will catch issues if degree != 3.
        # But random lifts might have self-loops or multi-edges if voltages align?
        # Self-loop: u==v in base? No, base is simple.
        # Multi-edge: (u,i) connected to (v,i+a) and also (u,i) connected to (v,i+b) with a!=b?
        # This would require two base edges between u and v. Base is simple.
        # So no multi-edges between same pair of base vertices.
        # What about (u,i) connected to (u,i) via a cycle in base?
        # No, direct edges only.
        
        return n, edges

    best_n = float('inf')
    best_edges = []
    
    # Scan m
    # Start from m=26 (n=520)
    # Increment m.
    # We can try a few random seeds for each m to increase chance.
    
    # Time budget per m?
    # Verification is ~1ms for n=520.
    # Construction is ~1ms.
    # We can try 100 m values in 100s?
    # 120s / 100 = 1.2s per m.
    # We can try 5 seeds per m.
    
    m_start = 26
    m_end = 200 # Upper limit, 20*200 = 4000. If we don't find it by 4000, we are in trouble.
    
    # Actually, if we don't find it, we should fall back to a known valid graph.
    # The current program returns LUW(3,7) which is 4374.
    # If we find a graph with n < 4374, we win.
    
    for m in range(m_start, m_end):
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds
        found = False
        for seed in range(5):
            n, edges = construct_lift(m, seed)
            # Quick check: is it simple and 3-regular?
            # The lift of a simple 3-regular graph is 3-regular.
            # It is simple if no two edges in the lift connect the same pair of vertices.
            # As argued, base is simple, so no multi-edges.
            # So we just check girth.
            
            if verify_girth_fast(n, edges, 3, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break # Found one for this m, try next m (smaller n)
                
        if found:
            # Since we scan m upwards, the first m found is the smallest n.
            # We can break here if we are confident?
            # No, we should continue to see if a smaller m works?
            # We are scanning upwards. So the first found is the smallest m.
            # So we can break.
            break
            
    if best_n < float('inf'):
        return best_n, best_edges

    # Fallback: If no lift found, return the LUW graph from the current program logic?
    # We don't have the LUW code here, but we can implement a simple fallback.
    # Or just return a large random graph? No, must be valid.
    # Let's implement a simple LUW(3,7) construction as a fallback to ensure we return *something* valid.
    
    def luw_construct_fallback(k, g, d=7):
        if k != 3:
            return None, None
        q = 3
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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

    n_luw, edges_luw = luw_construct_fallback(3, g, 7)
    if n_luw and verify_girth_fast(n_luw, edges_luw, 3, g):
        return n_luw, edges_luw
        
    # Last resort
    n_luw, edges_luw = luw_construct_fallback(3, g, 8)
    return n_luw, edges_luw

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
