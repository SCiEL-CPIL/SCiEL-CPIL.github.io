import time
import itertools
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Public Task Contract Interface
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    """
    Solves the triangle-free diameter-four augmentation problem.
    Strategy: 
    1. Generate candidate edges (non-edges of the path).
    2. Identify triangle-conflict pairs among candidates.
    3. Use Integer Linear Programming (ILP) to find the minimum number of 
       non-conflicting edges that reduce diameter to <= 4.
       Since standard MILP solvers don't handle non-linear diameter constraints 
       directly, we use a "branch and bound" style approach guided by ILP 
       feasibility checks on subsets, or more simply, we use a heuristic 
       initialization + local search which is robust, BUT the directive asks for 
       a DIFFERENT strategy. 
       
       Actually, pure ILP for diameter is hard to encode directly without 
       exponential variables. A better "different" strategy that is exact 
       and uses different machinery is **Iterative Deepening DFS with 
       Constraint Propagation** or **Backtracking with Bitsets**.
       
       Let's stick to the "Maximum Independent Set" framing mentioned in the 
       directive, implemented via **Bitset Backtracking**. This is a 
       structurally different search method (systematic exact search with 
       pruning) compared to the heuristic hub-and-greedy approach.
       
       The "triangle-conflict graph" approach:
       - Nodes in conflict graph = candidate edges.
       - Edge in conflict graph = two candidate edges form a triangle with base path or each other.
       - We need a set of candidate edges (independent set in conflict graph) 
         such that the resulting graph has diameter <= 4.
       - We search for the smallest such independent set.
    """
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # Generate all possible non-base edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
                
    # Pre-compute triangle conflicts
    # Two edges (u,v) and (x,y) conflict if adding both creates a triangle.
    # A triangle is formed if:
    # 1. They share a vertex and the other two endpoints are connected in the base graph (or by another added edge? No, we check static conflicts first).
    # Actually, triangle-freeness is a global property. 
    # If we add edge (u,v), we must ensure that for any neighbor w of u, (w,v) is not an edge in the final graph.
    # This is complex for dynamic checking. 
    # However, if we restrict our search to **bipartite** augmentations, we guarantee triangle-freeness.
    # The base path is bipartite (even/odd). 
    # If we only add edges between even and odd vertices, the graph remains bipartite and thus triangle-free.
    # This reduces the candidate set significantly and simplifies the conflict check to just "don't add duplicates".
    # Does a bipartite solution exist with minimal edges? 
    # The current best score is 12.1 (approx 10 edges? No, score is seed/value. Seed is 12.1? 
    # Wait, the prompt says "score 12.1000" for the current program which adds 10 edges.
    # Score = Seed / Value. If Seed is fixed, lower Value (edges) = Higher Score.
    # The current program adds 10 edges. Score 12.1 implies Seed = 121? 
    # Let's check: 121 / 10 = 12.1. Yes.
    # So we want to minimize edges.
    
    # Bipartite Candidate Edges: (even, odd)
    bipartite_candidates = []
    for u, v in candidates:
        if (u % 2) != (v % 2):
            bipartite_candidates.append((u, v))
            
    # We need to find the smallest subset of bipartite_candidates 
    # such that diameter <= 4.
    
    # Optimization: Use BFS to check diameter.
    # To make this fast, we can use a heuristic to guide the search, 
    # but the directive asks for a structural change.
    # Let's use **A* Search** or **Iterative Deepening DFS** on the set of edges.
    # Given the small N=24 and the fact that we likely need ~8-10 edges, 
    # a pure combinatorial search is too slow (C(200, 10) is huge).
    
    # Alternative Structural Change: **Greedy Set Cover with Lookahead**
    # This is different from the "Hub" strategy.
    # The Hub strategy forces a specific topology.
    # Set Cover models the problem as:
    # Each added edge covers a set of (s,t) pairs whose distance is reduced.
    # We need to cover all pairs (s,t) with distance > 4 in the base graph.
    
    # 1. Identify "critical pairs" (s,t) where dist_base(s,t) > 4.
    # 2. For each candidate edge e, calculate which critical pairs it "helps" (reduces distance).
    # 3. Solve Minimum Set Cover.
    
    # Base distances
    def get_base_dist():
        # Path graph distances are just |u-v|
        dist = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dist[i][j] = abs(i-j)
        return dist
        
    base_dist = get_base_dist()
    
    critical_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if base_dist[i][j] > 4:
                critical_pairs.append((i, j))
                
    # For each candidate edge, determine which critical pairs it can potentially resolve.
    # An edge (u,v) resolves pair (s,t) if using (u,v) in a path from s to t 
    # results in a distance <= 4.
    # Path length via (u,v) is dist(s,u) + 1 + dist(v,t) or dist(s,v) + 1 + dist(u,t).
    # Note: dist here is base path distance, but adding multiple edges complicates this.
    # However, for a *first pass* or *greedy step*, we can approximate.
    # Better: Use a heuristic that picks edges that connect distant parts of the path.
    
    # Let's try a **Genetic Algorithm** or **Simulated Annealing**? 
    # No, let's try **Local Search with Random Restarts** but with a **Bitset-based Diameter Check** 
    # which is significantly faster than BFS, allowing for many more restarts.
    # This is a "different optimization method" (stochastic global search with efficient evaluation) 
    # vs the current deterministic heuristic.
    
    # Actually, the most robust "different" approach for small N is **Complete Search with Pruning** 
    # using **Branch and Bound** on the number of edges, guided by a **Lower Bound** 
    # calculated from the number of remaining critical pairs and the maximum coverage per edge.
    
    # Let's implement **Branch and Bound** for Minimum Set Cover approximation.
    
    # Pre-calculate coverage:
    # For each candidate edge e, let C_e be the set of critical pairs (s,t) such that 
    # min(base_dist[s][u] + 1 + base_dist[v][t], base_dist[s][v] + 1 + base_dist[u][t]) <= 4 
    # IS NOT SUFFICIENT because other added edges might help further.
    # But if we just want to find *a* valid set, we can be more aggressive.
    
    # Let's stick to the **Bipartite Constraint** + **Greedy + Local Search** but 
    # implemented with **Bitsets** for speed, allowing a much deeper search.
    # The current program uses a fixed hub. This one will search for the best hubs/edges.
    
    # Efficient Diameter Check using Bitsets:
    # Represent adjacency as list of bitmasks (integers).
    # BFS using bitset operations is very fast.
    
    def build_adj_bitsets(added_edges):
        adj = [0] * n
        for u, v in base_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u, v in added_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def check_diameter_bitset(adj, limit=4):
        # Check if diameter <= limit
        # For each node, do BFS up to depth limit
        for src in range(n):
            visited = 1 << src
            frontier = 1 << src
            for depth in range(limit):
                # Expand frontier
                new_frontier = 0
                f = frontier
                while f:
                    lsb = f & -f
                    f ^= lsb
                    idx = lsb.bit_length() - 1
                    # Add neighbors of idx not in visited
                    neighbors = adj[idx] & ~visited
                    new_frontier |= neighbors
                if new_frontier == 0:
                    break
                visited |= new_frontier
                frontier = new_frontier
            if visited != (1 << n) - 1:
                return False
        return True

    # Triangle Free Check for Bipartite:
    # Since we only add edges between even and odd, and base is bipartite, 
    # the graph is always bipartite, hence triangle-free.
    # We just need to ensure we don't add duplicate edges (handled by set).
    
    # Search Strategy: Simulated Annealing / Random Local Search
    # Start with a random valid bipartite set, or a greedy one.
    # Move: Add an edge, Remove an edge, Swap an edge.
    # Objective: Minimize number of edges subject to Diameter <= 4.
    # If invalid, penalty = large number.
    
    def get_penalty(added_edges):
        adj = build_adj_bitsets(added_edges)
        if not check_diameter_bitset(adj, 4):
            return 1000 + len(added_edges)
        return len(added_edges)
        
    # Greedy Initialization
    def greedy_init():
        # Add edges that connect farthest apart nodes first?
        # Sort candidates by "potential" (e.g., max distance reduction)
        # Simple: Randomly select a subset, then repair.
        # Let's use a heuristic: Connect 0 to 20, 1 to 19, etc.
        # Actually, let's just pick random bipartite edges until valid.
        # Better: Start with empty, add edge that reduces diameter most.
        
        # For speed, let's use a fixed "smart" random start
        import random
        random.seed(42)
        
        # Candidate pool: Bipartite edges that are "long" (distance > 5)
        long_candidates = [e for e in bipartite_candidates if abs(e[0]-e[1]) > 5]
        
        current_set = set()
        # Try to add edges to satisfy diameter
        # We will run a local search from a few random starts
        
        starts = []
        for _ in range(50):
            # Random subset of long candidates
            size = random.randint(8, 12)
            subset = random.sample(long_candidates, min(size, len(long_candidates)))
            starts.append(set(subset))
            
        # Also add the "Hub" solution as a start (known to work)
        hub_solution = set()
        hub = 11
        for v in [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]:
            hub_solution.add((min(hub, v), max(hub, v)))
        starts.append(hub_solution)
        
        return starts

    starts = greedy_init()
    
    best_score = float('inf')
    best_set = None
    
    # Local Search
    # Deadline guard
    start_time = time.time()
    time_limit = 50.0 # 50s to leave margin
    
    all_candidates_set = set(bipartite_candidates)
    
    for start_set in starts:
        current_set = start_set.copy()
        current_score = get_penalty(current_set)
        
        # If start is already good, try to improve
        no_improvement_count = 0
        
        while time.time() - start_time < time_limit:
            if no_improvement_count > 50:
                break
                
            # Pick a random move: Add, Remove, Swap
            move_type = random.choice(['add', 'remove', 'swap'])
            new_set = current_set.copy()
            
            if move_type == 'add':
                # Pick a random candidate not in set
                candidates_to_add = list(all_candidates_set - current_set)
                if candidates_to_add:
                    e = random.choice(candidates_to_add)
                    new_set.add(e)
            elif move_type == 'remove':
                if current_set:
                    e = random.choice(list(current_set))
                    new_set.remove(e)
            elif move_type == 'swap':
                if current_set:
                    e_in = random.choice(list(current_set))
                    candidates_to_add = list(all_candidates_set - current_set)
                    if candidates_to_add:
                        e_out = random.choice(candidates_to_add)
                        new_set.remove(e_in)
                        new_set.add(e_out)
                        
            new_score = get_penalty(new_set)
            
            # Accept if better, or with probability if worse (Simulated Annealing style)
            # Simple Hill Climbing for now, with random restarts handled by 'starts'
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                no_improvement_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set.copy()
                    # If we found a very small set, we can stop early?
                    # Theoretical minimum is likely 8 or 9.
                    if best_score <= 8:
                        return [[u, v] for u, v in best_set]
            else:
                no_improvement_count += 1
                
    if best_set is not None:
        return [[u, v] for u, v in best_set]
    else:
        # Fallback to hub solution
        hub_solution = []
        hub = 11
        for v in [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]:
            hub_solution.append([hub, v])
        return hub_solution

# EVOLVE-BLOCK-END
