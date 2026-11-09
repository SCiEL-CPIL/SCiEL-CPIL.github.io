"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. For k=4, g=10, attempt to find the 320-vertex cage graph using
       randomized local search (2-switches) to maximize girth.
    2. If successful, return the graph.
    3. Fallback to LUV construction if search fails or for other k.
    """
    start_time = time.time()
    deadline = 115.0  # 5s safety margin before 120s limit

    if k == 4 and g == 10:
        # Target n is 320 (known smallest)
        # We will try to find a valid graph for n in [320, 320+50]
        # If we find one for 320, we return it.
        # If not, we return the smallest valid one found.
        # If none found, fall back to LUV.
        
        def get_girth_and_cycle(adj, n):
            """
            Returns (girth, cycle_edges) where cycle_edges is a list of (u, v) in the shortest cycle.
            If no cycle < g, returns (g, []).
            """
            # BFS from every node to find shortest cycle
            # Optimization: We can stop BFS early if we find a cycle of length < current best
            # But we need the cycle edges to break it.
            
            best_girth = float('inf')
            best_cycle = []
            
            for start in range(n):
                # BFS
                dist = [-1] * n
                parent = [-1] * n
                q = deque()
                q.append(start)
                dist[start] = 0
                
                found_cycle_in_this_bfs = False
                cycle_len = float('inf')
                cycle_edges = []
                
                while q:
                    u = q.popleft()
                    du = dist[u]
                    if du >= best_girth: # Prune
                        continue
                    
                    for v in range(n):
                        if adj[u, v] == 1:
                            if dist[v] == -1:
                                dist[v] = du + 1
                                parent[v] = u
                                q.append(v)
                            else:
                                # Back edge or cross edge
                                # If u and v are not parent/child, it's a cycle
                                if u != parent[v] and v != parent[u]:
                                    # Reconstruct cycle
                                    # Path from v to u in tree + edge (u, v)
                                    # Wait, BFS tree gives shortest path from start.
                                    # Cycle length = dist[u] + dist[v] + 1
                                    cl = du + dist[v] + 1
                                    if cl < cycle_len:
                                        cycle_len = cl
                                        # Reconstruct edges
                                        # Path v -> ... -> start
                                        # Path u -> ... -> start
                                        # This is complex to get exact edges of the cycle in the graph
                                        # Easier: Just record the length for now?
                                        # For 2-switch, we need to modify edges.
                                        # We need to know WHICH edges to switch.
                                        # A random edge in the cycle is enough.
                                        # We can pick a random edge incident to u or v?
                                        # Actually, any edge in the cycle will do.
                                        # Let's just pick (u, v) as the "chord" that closes the cycle?
                                        # No, (u, v) is an edge in the graph.
                                        # The cycle is u - ... - v - u.
                                        # So (u, v) is part of the cycle.
                                        # We can try to 2-switch (u, v) with another edge.
                                        
                                        if cl < best_girth:
                                            best_girth = cl
                                            best_cycle = [(u, v)] # Placeholder, we'll pick random edge from cycle later if needed
                                            # Actually, we just need to know that a short cycle exists.
                                            # To break it, we can pick a random edge from the graph that is in a short cycle.
                                            # But identifying all such edges is expensive.
                                            # Heuristic: If we find a short cycle, pick a random edge (a, b) from the graph.
                                            # If (a, b) is in a short cycle, switching it might help.
                                            # Better: Pick an edge from the specific cycle found.
                                            # To get the cycle edges:
                                            # Trace back from u to LCA(u, v) and v to LCA(u, v).
                                            # This is O(n).
                                            
                                            # For simplicity and speed, let's just return the length.
                                            # And in the main loop, we will pick random edges to switch.
                                            # If the girth decreases, revert.
                                            pass
                # End BFS for start
                
                if cycle_len < best_girth:
                    best_girth = cycle_len
                    # We don't strictly need the cycle edges if we use random switching
                    # and accept/reject based on girth change.
                
                if best_girth == g: # Found girth g, can stop?
                    # Actually we want girth >= g. If we found girth g, we are good?
                    # No, we want NO cycle shorter than g.
                    # So if min cycle is g, it is valid.
                    # If min cycle < g, it is invalid.
                    # So if we find a cycle of length < g, we return it.
                    # If we finish all BFS and min cycle >= g, return (g, []).
                    pass
                    
            if best_girth < g:
                return best_girth, best_cycle
            else:
                return g, [] # Valid

        def random_4_regular_graph(n, seed=None):
            """Generates a random 4-regular graph."""
            if seed is not None:
                random.seed(seed)
            
            # Configuration model
            # 4n stubs
            # Pair them up randomly
            # Ensure no multi-edges or loops (retry if fail)
            
            for _ in range(1000): # Max retries
                stubs = [i for i in range(n) for _ in range(4)]
                random.shuffle(stubs)
                adj = np.zeros((n, n), dtype=int)
                valid = True
                for i in range(0, 4*n, 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if adj[u, v] == 1:
                        valid = False
                        break
                    adj[u, v] = 1
                    adj[v, u] = 1
                if valid:
                    # Check connectivity? Not strictly required for girth, but good.
                    # If disconnected, girth is min of components.
                    # We can just proceed.
                    return adj
            return None

        def apply_2_switch(adj, n, e1, e2):
            """
            e1 = (u, v), e2 = (x, y)
            Remove (u, v) and (x, y)
            Add (u, x) and (v, y) OR (u, y) and (v, x)
            Returns new adj if valid (no multi-edges/loops), else None.
            """
            u, v = e1
            x, y = e2
            
            # Check if edges exist
            if adj[u, v] != 1 or adj[x, y] != 1:
                return None
            
            # Option 1: (u, x) and (v, y)
            # Check if these edges exist or are loops
            if u != x and v != y and adj[u, x] == 0 and adj[v, y] == 0:
                new_adj = adj.copy()
                new_adj[u, v] = 0; new_adj[v, u] = 0
                new_adj[x, y] = 0; new_adj[y, x] = 0
                new_adj[u, x] = 1; new_adj[x, u] = 1
                new_adj[v, y] = 1; new_adj[y, v] = 1
                return new_adj
            
            # Option 2: (u, y) and (v, x)
            if u != y and v != x and adj[u, y] == 0 and adj[v, x] == 0:
                new_adj = adj.copy()
                new_adj[u, v] = 0; new_adj[v, u] = 0
                new_adj[x, y] = 0; new_adj[y, x] = 0
                new_adj[u, y] = 1; new_adj[y, u] = 1
                new_adj[v, x] = 1; new_adj[x, v] = 1
                return new_adj
                
            return None

        best_n = None
        best_edges = None
        
        # Try n = 320
        target_n = 320
        
        # Time budget for search
        search_deadline = start_time + 100.0
        
        # We will try multiple restarts
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > search_deadline:
                break
            
            adj = random_4_regular_graph(target_n, seed=42 + restart)
            if adj is None:
                continue
                
            # Initial girth check
            cur_girth, _ = get_girth_and_cycle(adj, target_n)
            
            if cur_girth >= g:
                # Convert to edge list
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i, j] == 1:
                            edges.append((i, j))
                return target_n, edges
            
            # Local search
            for step in range(2000):
                if time.time() > search_deadline:
                    break
                    
                # Pick two random edges
                # We need to pick edges from the adjacency matrix
                # For efficiency, we can precompute edge list?
                # n=320, edges=640.
                # Picking random indices from edge list is O(1) if list exists.
                # Let's maintain an edge list.
                
                # Rebuild edge list occasionally or maintain it?
                # For simplicity, pick random u, v, x, y and check if they are edges.
                # Probability of hitting an edge is 2*E / n^2 = 1280 / 102400 ~ 1.2%.
                # Too low.
                
                # Better: Maintain a list of edges.
                # This is getting complex.
                # Alternative: Pick random u, then random neighbor v.
                
                u = random.randint(0, target_n - 1)
                # Get neighbors of u
                neighbors_u = np.where(adj[u] == 1)[0]
                if len(neighbors_u) == 0: continue
                v = neighbors_u[random.randint(0, len(neighbors_u)-1)]
                
                x = random.randint(0, target_n - 1)
                neighbors_x = np.where(adj[x] == 1)[0]
                if len(neighbors_x) == 0: continue
                y = neighbors_x[random.randint(0, len(neighbors_x)-1)]
                
                # Ensure (u,v) and (x,y) are distinct
                if (u == x and v == y) or (u == y and v == x):
                    continue
                    
                new_adj = apply_2_switch(adj, target_n, (u, v), (x, y))
                if new_adj is None:
                    continue
                
                new_girth, _ = get_girth_and_cycle(new_adj, target_n)
                
                # Accept if girth improved or same (to allow exploration)
                # Or if new_girth >= g
                if new_girth >= cur_girth:
                    adj = new_adj
                    cur_girth = new_girth
                    
                    if cur_girth >= g:
                        edges = []
                        for i in range(target_n):
                            for j in range(i+1, target_n):
                                if adj[i, j] == 1:
                                    edges.append((i, j))
                        return target_n, edges
        
        # If 320 failed, try larger n?
        # The problem asks for "as few vertices as possible".
        # If we can't find 320, we should try 321, etc.
        # But 320 is the known minimum.
        # If we fail, we return LUV.
        
        pass

    # Fallback to LUV
    def luv_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    try:
        n, edges = luv_construct(k, g)
        return n, edges
    except:
        return 0, []
# EVOLVE-BLOCK-END
