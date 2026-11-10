"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix and list of edge indices
    adj_matrix = np.zeros((n, n), dtype=int)
    edge_list = []
    for u, v in edges:
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
        edge_list.append((u, v))
        
    m = len(edge_list)
    
    # Variables:
    # 0..n-1: x_v (1 if v is a branch)
    # n..n+m-1: y_e (1 if edge e is used in any path)
    # n+m..n+m+n-1: z_v (1 if v is an interior vertex used in any path)
    
    num_vars = n + m + n
    x_idx = lambda v: v
    y_idx = lambda e: n + e
    z_idx = lambda v: n + m + v
    
    # Objective: Maximize sum(x_v)
    # scipy milp minimizes, so use -1 for x_v, 0 for others
    c = np.zeros(num_vars)
    for v in range(n):
        c[x_idx(v)] = -1.0
        
    # Bounds
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    
    # Constraints:
    # 1. Vertex capacity: x_v + z_v <= 1 (A vertex is either a branch or an interior vertex, not both)
    # 2. Flow conservation for each pair (u, v) of branches?
    #    This is the hard part. Standard ILP for disjoint paths usually requires variables per pair.
    #    Let f_{e}^{(u,v)} be flow from u to v on edge e.
    #    This explodes variables: C(n,2) * m. For n=24, C(24,2)=276, m~100. 27600 vars. Manageable.
    
    # Let's re-define variables to include flow per pair.
    # This is a "topological minor" check.
    
    # Actually, a simpler ILP formulation for "clique subdivision" exists.
    # However, the flow-based one is more robust for finding the *largest* set.
    
    # Let's try the flow-based ILP.
    # Variables:
    # X_v: 1 if v is branch.
    # Z_v: 1 if v is interior.
    # F_{e, p}: Flow on edge e for pair p. (Binary)
    
    # Pairs are dynamic based on X. This makes it non-linear (X_u * X_v * F).
    # To linearize, we can assume a fixed set of potential pairs? No, that's not right.
    
    # Alternative: Use the property that if we have a set of branches B, we need internally disjoint paths.
    # This is equivalent to finding a set of vertices B such that the graph contains a subdivision of K_|B|.
    
    # A known ILP formulation for "Maximum Clique Subdivision" or "Topological Minor":
    # It is NP-hard.
    
    # Given the small size (n=24), we can use a branch-and-bound or heuristic if ILP is too complex.
    # But let's try a simpler ILP that enforces:
    # For any two branches u, v, there is a path.
    
    # Actually, there is a simpler approach using **Edge-Disjoint Paths** constraints.
    # If we fix the set of branches, we can check if they can be connected.
    # But we want to *choose* the branches.
    
    # Let's use a heuristic that is smarter than greedy: **Local Search / Simulated Annealing**
    # on the set of branches.
    # Objective: Maximize |B| such that internally disjoint paths exist.
    # Move: Add a vertex, remove a vertex, swap.
    # Evaluation: Check if disjoint paths exist (using max-flow or matching).
    
    # This avoids the massive variable explosion of the full ILP with pair flows.
    # And it's structurally different from the seed's "Bron-Kerbosch + Direct Edges".
    
    # Strategy:
    # 1. Start with a large clique (from seed logic or greedy).
    # 2. Try to add vertices one by one.
    # 3. For each candidate set, verify if internally disjoint paths exist.
    #    Verification:
    #    - Remove branch vertices.
    #    - For each pair of branches, we need a path.
    #    - This is the "Disjoint Paths" problem, which is NP-hard in general, but for fixed k (number of branches), it's polynomial.
    #    - Here k is small (<= 10 maybe).
    #    - We can use a reduction to Max Flow with vertex capacities.
    #    - Create a super source S and super sink T? No, we need paths for ALL pairs.
    #    - This is the "Group Steiner" or "Multicommodity Flow" problem.
    #    - For small k, we can use a matching-based check or simply BFS with forbidden vertices iteratively?
    #    - Actually, if we just need *existence*, we can try to find paths greedily? No, that fails.
    #    - We can use **Hopcroft-Karp** on a bipartite graph? No, paths can be long.
    
    # Let's use a **Max-Flow with Vertex Splitting** approach for a SINGLE pair to check connectivity, but we need ALL pairs simultaneously.
    
    # Wait, the condition is "internally disjoint".
    # If we select branches B, can we find |B| choose 2 internally disjoint paths?
    # This is equivalent to: In the graph G - B, can we find vertex-disjoint paths connecting the neighbors of each pair?
    
    # A simpler sufficient condition:
    # If the graph is "highly connected", large cliques exist.
    # The seed found a clique of size 6.
    # The inspiration found size 7-8 using BFS routing.
    
    # Let's implement a **Greedy Expansion with Backtracking** and **BFS Routing**.
    # This is structurally different from the pure Bron-Kerbosch + Direct Edge approach because it allows non-clique branches connected by longer paths.
    
    # 1. Generate candidate branch sets.
    #    - Start with all vertices? No.
    #    - Start with a maximal clique.
    #    - Try to add vertices from the remaining set.
    # 2. For a given set of branches B, check if internally disjoint paths exist.
    #    - We can model this as a flow problem if we fix the order? No.
    #    - We can use a **Match** based approach if paths are short?
    #    - Let's use **Iterative BFS with Conflict Resolution**.
    #      - Try to find a path for each pair.
    #      - If conflict, try to reroute.
    #      - This is heuristic.
    
    # Better: **Max Flow for each pair sequentially?** No, they must be disjoint.
    
    # Let's use **Integer Linear Programming** but with a simplified constraint set:
    # We assume that if a set of branches is valid, there exists an assignment of interior vertices to paths.
    
    # Given the time, let's stick to a **Smart Heuristic**:
    # 1. Find Max Clique (Bron-Kerbosch).
    # 2. Try to expand this clique by adding vertices that are "close" to multiple existing branches.
    # 3. For each expansion, verify using a **Max-Flow with Vertex Capacities** for the *whole set*?
    #    Actually, there is a theorem: A graph contains a subdivision of K_k if and only if ...
    #    There is no simple max-flow check for all pairs simultaneously.
    
    # However, for k <= 8, we can use **Color-Coding** or **DP on Tree Decomposition**?
    # n=24 is small.
    
    # Let's use **Simulated Annealing** on the set of branches.
    # Energy function: -|B| if valid, 0 if invalid.
    # Validity check:
    #   - For each pair (u, v) in B, find a path.
    #   - Ensure interiors are disjoint.
    #   - Use a simple **Greedy Path Finder** with **Backtracking** for the check.
    #     - Order pairs by distance (shortest first).
    #     - Find shortest path avoiding used vertices.
    #     - If fail, try next shortest.
    #     - If all fail, invalid.
    
    # This check is fast for n=24.
    
    # Algorithm:
    # 1. Init B with Max Clique.
    # 2. Loop:
    #    - Pick a random vertex v not in B.
    #    - New B = B U {v}.
    #    - Check if B is valid.
    #    - If valid, accept.
    #    - If invalid, reject.
    #    - Also try removing a vertex from B to see if we can add two? (Swap)
    # 3. Return best B.
    
    def find_max_clique():
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_clique_size = 0
        max_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal max_clique_size, max_clique
            if len(P) == 0 and len(X) == 0:
                if len(R) > max_clique_size:
                    max_clique_size = len(R)
                    max_clique = list(R)
                return
            if len(P) == 0:
                if len(R) > max_clique_size:
                    max_clique_size = len(R)
                    max_clique = list(R)
                return
            union_p_x = P.union(X)
            if not union_p_x:
                return
            best_pivot = None
            max_uncovered = -1
            for u in union_p_x:
                uncovered = len(P.difference(adj[u]))
                if uncovered > max_uncovered:
                    max_uncovered = uncovered
                    best_pivot = u
            if best_pivot is None:
                return
            for v in list(P.difference(adj[best_pivot])):
                bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)
                
        bron_kerbosch(set(), set(range(n)), set())
        return max_clique

    initial_clique = find_max_clique()
    current_branches = set(initial_clique)
    best_branches = set(initial_clique)
    best_score = len(best_branches)
    
    # Precompute adjacency for fast access
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Validity Check:
    # Can we find internally disjoint paths for all pairs in B?
    # We use a greedy approach:
    # 1. Generate all pairs.
    # 2. Sort pairs by shortest path length (avoiding B \ {u,v}).
    # 3. For each pair, find a path using BFS, avoiding used interior vertices.
    # 4. If any pair fails, return False.
    
    def is_valid_branch_set(B):
        if len(B) < 2:
            return False
        B_list = list(B)
        pairs = list(itertools.combinations(B_list, 2))
        
        # Sort pairs by heuristic: sum of degrees? or just random?
        # Let's sort by shortest path length in G - (B \ {u,v})
        def get_dist(u, v, forbidden):
            # BFS from u to v avoiding forbidden
            visited = {u}
            queue = [(u, [u])]
            while queue:
                curr, path = queue.pop(0)
                if curr == v:
                    return len(path) - 1
                for nei in adj[curr]:
                    if nei not in visited and nei not in forbidden:
                        visited.add(nei)
                        queue.append((nei, path + [nei]))
            return float('inf')
            
        # We need to be careful with the "forbidden" set during sorting.
        # It changes as we assign paths.
        # So we can't pre-sort perfectly.
        # We will just iterate.
        
        used_interiors = set()
        
        # To improve success rate, we should process "tight" pairs first.
        # Let's estimate tightness by distance in G - B.
        B_set = set(B)
        
        # Calculate distances for all pairs in G - (B \ {u,v}) is expensive to do dynamically.
        # Let's just pick a random order or degree-based order.
        # High degree vertices are easier to route around?
        
        # Let's try multiple random orders (restarts within the check)
        for _ in range(5): # 5 attempts with different orders
            used_interiors = set()
            pairs_copy = list(pairs)
            # Sort by random or by distance
            # Let's sort by distance in G - B
            def dist_key(pair):
                u, v = pair
                forbidden = B_set - {u, v}
                return get_dist(u, v, forbidden)
                
            pairs_copy.sort(key=dist_key)
            
            success = True
            for u, v in pairs_copy:
                # Find path from u to v avoiding used_interiors and B \ {u,v}
                forbidden = used_interiors.union(B_set - {u, v})
                
                # BFS
                visited = {u}
                queue = [(u, [u])]
                path_found = None
                while queue:
                    curr, path = queue.pop(0)
                    if curr == v:
                        path_found = path
                        break
                    for nei in adj[curr]:
                        if nei not in visited and nei not in forbidden:
                            visited.add(nei)
                            queue.append((nei, path + [nei]))
                            
                if not path_found:
                    success = False
                    break
                else:
                    # Add interior vertices to used
                    for node in path[1:-1]:
                        used_interiors.add(node)
                        
            if success:
                return True, path_found # Wait, we need all paths
                # Actually, we just need to return True for validity.
                # But we need the paths to return in the final answer.
                # So we should return the paths too.
                
        return False, None

    # Refined Validity Check that returns paths
    def check_and_get_paths(B):
        B_list = list(B)
        pairs = list(itertools.combinations(B_list, 2))
        B_set = set(B)
        
        # Try a few random permutations of pairs to handle ordering sensitivity
        for _ in range(10):
            used_interiors = set()
            paths = {}
            pairs_copy = list(pairs)
            import random
            random.shuffle(pairs_copy)
            
            success = True
            for u, v in pairs_copy:
                forbidden = used_interiors.union(B_set - {u, v})
                
                # BFS
                visited = {u}
                queue = [(u, [u])]
                path_found = None
                while queue:
                    curr, path = queue.pop(0)
                    if curr == v:
                        path_found = path
                        break
                    for nei in adj[curr]:
                        if nei not in visited and nei not in forbidden:
                            visited.add(nei)
                            queue.append((nei, path + [nei]))
                            
                if not path_found:
                    success = False
                    break
                else:
                    paths[(u, v)] = path_found
                    for node in path_found[1:-1]:
                        used_interiors.add(node)
                        
            if success:
                return True, paths
        
        return False, None

    # Local Search
    current_B = set(initial_clique)
    best_B = set(current_B)
    best_paths = None
    valid, paths = check_and_get_paths(current_B)
    if valid:
        best_paths = paths
        best_score = len(current_B)
        
    # Candidates for addition: vertices not in B
    # Candidates for removal: vertices in B
    
    # We will try to add vertices.
    # If adding fails, we might try swapping.
    
    # To speed up, we only consider vertices that are "connected" to B.
    
    for step in range(1000): # Limit steps
        if time.time() > deadline:
            break
            
        # Try to add a random vertex
        remaining = list(set(range(n)) - current_B)
        if remaining:
            # Sort by degree or connectivity to B
            # Let's pick one with high degree in G
            v = max(remaining, key=lambda x: len(adj[x]))
            
            new_B = current_B.union([v])
            valid, paths = check_and_get_paths(new_B)
            if valid:
                current_B = new_B
                best_B = set(current_B)
                best_paths = paths
                best_score = len(current_B)
                # Restart counter?
            else:
                # Try swapping
                # Remove a vertex from B, add v
                # Pick a random vertex from B to remove
                if len(current_B) > 2:
                    remove_v = random.choice(list(current_B))
                    swap_B = (current_B - {remove_v}).union([v])
                    valid, paths = check_and_get_paths(swap_B)
                    if valid:
                        current_B = swap_B
                        # Don't update best immediately, but keep current
                        # Actually, if size is same, it's a neutral move.
                        # We want to grow.
                        pass
        
    # Final check
    if best_paths is None:
        # Fallback to clique
        best_B = set(initial_clique)
        valid, paths = check_and_get_paths(best_B)
        if valid:
            best_paths = paths
        else:
            # Should not happen for clique
            best_paths = {}
            for u, v in itertools.combinations(list(best_B), 2):
                best_paths[(u, v)] = [u, v]

    # Construct return value
    branches = [int(v) for v in best_B]
    paths_list = []
    for u, v in itertools.combinations(branches, 2):
        # Ensure key exists
        key = (u, v)
        if key not in best_paths:
            key = (v, u)
        paths_list.append([int(x) for x in best_paths[key]])
        
    return {'branches': branches, 'paths': paths_list}

# EVOLVE-BLOCK-END
