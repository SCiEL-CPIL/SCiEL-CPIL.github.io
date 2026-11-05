import time
import random
from collections import defaultdict
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # The seed program uses 4 * n * d iterations
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def get_conflicts(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # Conflict condition: 
            # 1. Share vertex: x in {u,v} or y in {u,v}
            # 2. Joined by edge: (u,x) in E, (u,y) in E, (v,x) in E, (v,y) in E
            
            # Efficient enumeration:
            # Any edge (x,y) conflicts with (u,v) if:
            # - x or y is u or v (share vertex)
            # - (u,x) or (u,y) or (v,x) or (v,y) is an edge
            
            # Let's iterate neighbors of u and v.
            # If (x,y) is an edge and x is neighbor of u, then (u,x) is edge -> conflict.
            # This covers:
            # - Edges incident to u (x=u, y=neighbor): (u, neighbor) is edge. Wait, if x=u, is u neighbor of u? No.
            #   But we iterate neighbors of u. Let x be neighbor of u.
            #   If we look at edges incident to x, say (x, y).
            #   If y == u, then (x,u) is edge. Conflict with (u,v)? Yes, shares u.
            #   If y != u, then (u,x) is edge. Conflict with (u,v)? Yes, joined by (u,x).
            #   So all edges incident to neighbors of u conflict with (u,v).
            # Similarly for neighbors of v.
            
            # Does this miss anything?
            # What if an edge (x,y) conflicts because (v,x) is an edge?
            # Then x is neighbor of v. We iterate neighbors of v, find x, then find edge (x,y). Covered.
            
            # What if an edge (x,y) conflicts because it shares vertex v?
            # Then one endpoint is v. Say y=v.
            # Then x is neighbor of v. We iterate neighbors of v, find x, then find edge (x,v). Covered.
            
            # So the set of conflicting edges for (u,v) is exactly the union of:
            # { (x,y) in E | x in N(u) } U { (x,y) in E | x in N(v) }
            
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
                        
            conflicts[i].discard(i)
            
        return conflicts

    def solve_graph(edges, adj, max_time):
        m = len(edges)
        if m == 0:
            return []
        
        conflicts = get_conflicts(edges, adj)
        
        # Convert conflicts to lists for faster iteration if needed, but sets are okay for small graphs
        # DSATUR requires tracking saturation degree (number of distinct colors in neighbors)
        
        # We will use a backtracking solver with DSATUR heuristic.
        # State:
        # color[i]: color of edge i, -1 if uncolored
        # sat[i]: set of colors used by colored neighbors of i
        # domain[i]: set of available colors (0..max_c - 1) minus sat[i]
        
        # Since we don't know max_c, we will grow the number of colors as needed.
        # Or we can binary search on max_c? 
        # Given the bounds (20, 31, 45), we can try to find the minimum.
        # However, DSATUR usually finds a good solution quickly.
        # Let's try to find *a* valid coloring with K colors, then try K-1, etc?
        # Or just let DSATUR assign the next available color (like greedy) but with backtracking?
        # Standard DSATUR for minimum coloring:
        # 1. Pick uncolored vertex with max saturation degree (tie-break: max degree in conflict graph).
        # 2. Assign smallest feasible color.
        # 3. Backtrack if conflict.
        
        # To ensure we find the *minimum*, we can iterate K from lower_bound to upper_bound.
        # Lower bound: max degree in conflict graph + 1? No, chromatic number >= clique number.
        # Upper bound: greedy result.
        
        # Let's first run a fast greedy to get an upper bound.
        def greedy_coloring():
            # Order by degree descending
            order = list(range(m))
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)
            color = [-1] * m
            for i in order:
                forbidden = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        forbidden.add(color[j])
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
            return color, max(color) + 1 if color else 0

        init_color, ub = greedy_coloring()
        
        # Lower bound:
        # Find max clique? Too slow.
        # Simple LB: 1 + max_degree_in_conflict_graph? No.
        # Let's just start from ub and go down? Or start from 1 and go up?
        # Going up is safer if LB is weak. But UB is tight.
        # Let's try K from 1 to UB.
        # Actually, DSATUR is very effective. Let's just run DSATUR with a dynamic color limit?
        # No, standard DSATUR for min coloring usually fixes K and checks satisfiability.
        
        # Given time constraints, let's try K from 1 to UB.
        # For each K, run CP with DSATUR.
        # If found, return.
        
        # Optimization: Precompute conflict degrees
        conf_degrees = [len(conflicts[i]) for i in range(m)]
        
        def cp_solver(K):
            """
            Returns a valid coloring using colors 0..K-1 or None.
            Uses DSATUR heuristic.
            """
            color = [-1] * m
            sat = [set() for _ in range(m)] # Colors of neighbors
            # We need to track uncolored vertices efficiently
            
            # For speed, we can use a simple list and scan for max sat each time.
            # m is small (< 1000), so O(m^2) per step is fine?
            # Steps: m. Total O(m^3) worst case? 1000^3 = 10^9. Too slow in Python.
            # Need better selection.
            
            # Let's use a priority queue or just scan.
            # Scanning 500 vertices 500 times = 250,000 ops. Very fast.
            
            def select_vertex():
                best_idx = -1
                best_sat = -1
                best_deg = -1
                for i in range(m):
                    if color[i] == -1:
                        s = len(sat[i])
                        d = conf_degrees[i]
                        # Tie break: higher degree
                        if s > best_sat or (s == best_sat and d > best_deg):
                            best_sat = s
                            best_deg = d
                            best_idx = i
                return best_idx

            def backtrack(idx):
                if time.time() > max_time:
                    return False
                
                if idx == -1:
                    return True
                
                # Select next vertex using DSATUR
                v = select_vertex()
                if v == -1:
                    return True # All colored
                
                # Try colors 0..K-1
                for c in range(K):
                    if c in sat[v]:
                        continue
                    
                    # Assign
                    color[v] = c
                    # Update neighbors' saturation
                    # We need to revert this later.
                    # Store changes?
                    # Or just recompute?
                    # Recomputing sat sets is O(deg).
                    # Let's store the previous state of neighbors' sat sets?
                    # Or simply:
                    # Since we are backtracking, we can just record which neighbors had c added.
                    
                    affected = []
                    for u in conflicts[v]:
                        if color[u] == -1:
                            if c not in sat[u]:
                                sat[u].add(c)
                                affected.append(u)
                    
                    if backtrack(v): # Pass v? No, select_vertex picks next.
                        return True
                    
                    # Undo
                    for u in affected:
                        sat[u].remove(c)
                    color[v] = -1
                
                return False

            # Start
            # We need a way to know when all are colored.
            # select_vertex returns -1 if none uncolored.
            # But backtrack needs to know when to stop.
            # My recursive call `backtrack(v)` is slightly wrong. It should just call `backtrack(None)` or similar.
            # Let's refactor to a cleaner recursion.
            
            def solve():
                # Count colored
                # Actually, select_vertex handles termination.
                # But we need to ensure we don't recurse infinitely.
                # Base case: if no uncolored vertex, return True.
                
                v = select_vertex()
                if v == -1:
                    return True
                
                for c in range(K):
                    if c in sat[v]:
                        continue
                    
                    color[v] = c
                    affected = []
                    for u in conflicts[v]:
                        if color[u] == -1:
                            if c not in sat[u]:
                                sat[u].add(c)
                                affected.append(u)
                    
                    if solve():
                        return True
                    
                    for u in affected:
                        sat[u].remove(c)
                    color[v] = -1
                
                return False
            
            return solve()

        # Try K from 1 to UB
        # We can optimize by starting from a better LB if we had one, but 1 is fine for small graphs.
        # Actually, let's start from max(conf_degrees)+1? No, that's not a valid LB for chromatic number.
        # But it's a good heuristic start.
        # Let's just iterate.
        
        best_solution = init_color
        best_k = ub
        
        # Try to improve
        # We can try K from 1 to ub-1
        # If we find a solution for K, that's optimal for that K.
        # We want the smallest K.
        
        # To save time, we can check if K=1 is possible (no edges? or independent set of size m?)
        # For each K in range(1, ub):
        #   if cp_solver(K):
        #       return color
        
        # But CP solver might be slow if K is too small.
        # Let's try K from ub down to 1? No, we want min.
        # Let's try K from 1 up.
        
        # Time budget per K?
        # Total time 145s.
        # Let's allocate time wisely.
        
        # If m is small, just run it.
        
        # Let's try to find the minimum K.
        for K in range(1, ub + 1):
            if time.time() > max_time:
                break
                
            # Reset state
            # The cp_solver function above uses closures over color, sat.
            # We need to reset them for each K.
            
            color = [-1] * m
            sat = [set() for _ in range(m)]
            
            # Re-define select and solve inside this loop or pass state?
            # Let's make a local function that takes K and returns solution.
            
            def run_cp(K_val):
                col = [-1] * m
                sat_loc = [set() for _ in range(m)]
                
                def sel():
                    best_idx = -1
                    best_sat = -1
                    best_deg = -1
                    for i in range(m):
                        if col[i] == -1:
                            s = len(sat_loc[i])
                            d = conf_degrees[i]
                            if s > best_sat or (s == best_sat and d > best_deg):
                                best_sat = s
                                best_deg = d
                                best_idx = i
                    return best_idx
                
                def rec():
                    if time.time() > max_time:
                        return False
                    v = sel()
                    if v == -1:
                        return True
                    
                    for c in range(K_val):
                        if c in sat_loc[v]:
                            continue
                        col[v] = c
                        affected = []
                        for u in conflicts[v]:
                            if col[u] == -1:
                                if c not in sat_loc[u]:
                                    sat_loc[u].add(c)
                                    affected.append(u)
                        
                        if rec():
                            return True
                        
                        for u in affected:
                            sat_loc[u].remove(c)
                        col[v] = -1
                    return False
                
                if rec():
                    return col
                return None

            sol = run_cp(K)
            if sol is not None:
                best_solution = sol
                best_k = K
                break # Found minimum K
        
        return best_solution

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    results = []
    
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Time allocation per graph
        # Total 145s. 3 graphs. ~45s each.
        # But first graph is smallest, last is largest.
        # Let's give more time to larger graphs.
        # Or just use remaining time.
        
        current_time = time.time()
        remaining = deadline - current_time
        
        # Heuristic time allocation:
        # Graph 1 (40 nodes, d=4): 10s
        # Graph 2 (60 nodes, d=5): 25s
        # Graph 3 (80 nodes, d=6): 60s
        # Total 95s. Buffer 50s.
        
        if len(results) == 0:
            t_alloc = 15.0
        elif len(results) == 1:
            t_alloc = 30.0
        else:
            t_alloc = 70.0
            
        # Cap by remaining time
        t_alloc = min(t_alloc, remaining - 5.0) # 5s buffer for next
        
        if t_alloc < 1.0:
            # Not enough time, use greedy
            # Re-implement quick greedy here or reuse?
            # Reuse logic from solve_graph's greedy part
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            conflicts = get_conflicts(edges, adj)
            order = list(range(m))
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)
            color = [-1] * m
            for i in order:
                forbidden = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        forbidden.add(color[j])
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
            results.append(color)
            continue

        sol = solve_graph(edges, adj, time.time() + t_alloc)
        results.append(sol)
        
    return results
