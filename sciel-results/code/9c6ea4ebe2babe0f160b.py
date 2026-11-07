"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency for fast access
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def has_bicolored_cycle(colors, c1, c2):
        """
        Check if the subgraph induced by colors c1 and c2 contains a cycle.
        Uses DFS to detect cycles in the undirected graph formed by these two colors.
        """
        # Identify vertices with color c1 or c2
        vertices_in_subgraph = []
        for i in range(n):
            if colors[i] == c1 or colors[i] == c2:
                vertices_in_subgraph.append(i)
                
        if len(vertices_in_subgraph) < 3:
            return False
            
        # Build local adjacency for the subgraph
        # We only care about edges between vertices in the subgraph
        # And we only care about edges where both endpoints are in {c1, c2}
        # Actually, the subgraph is defined by the vertices with color c1 or c2,
        # and the edges between them.
        
        # For efficiency, we can just iterate through all edges and check if both endpoints are in the subgraph
        # But since n is small, we can do a DFS on the global graph restricted to these colors.
        
        visited = {v: False for v in vertices_in_subgraph}
        # To speed up, we can use a set for O(1) lookup
        subgraph_set = set(vertices_in_subgraph)
        
        def dfs(u, parent):
            visited[u] = True
            for v in adj[u]:
                if v not in subgraph_set:
                    continue
                if not visited[v]:
                    if dfs(v, u):
                        return True
                elif v != parent:
                    return True
            return False
            
        for v in vertices_in_subgraph:
            if not visited[v]:
                if dfs(v, -1):
                    return True
        return False

    def evaluate(colors):
        """
        Returns (num_violations, num_distinct_colors).
        Violations:
        1. Adjacent vertices with same color.
        2. Bicolored cycles for any pair of colors.
        """
        violations = 0
        
        # 1. Check adjacent same color
        for u, v in edges:
            if colors[u] == colors[v]:
                violations += 1
                
        # 2. Check bicolored cycles
        used_colors = set(colors)
        color_list = list(used_colors)
        for i in range(len(color_list)):
            for j in range(i + 1, len(color_list)):
                c1 = color_list[i]
                c2 = color_list[j]
                if has_bicolored_cycle(colors, c1, c2):
                    violations += 1
                    
        return violations, len(used_colors)

    best_solution = None
    best_violations = float('inf')
    best_distinct = n + 1
    
    # Try to find a valid solution with minimal distinct colors
    # We iterate max_colors from 2 up to n
    # For each max_colors, we run Simulated Annealing to find a valid solution (0 violations)
    
    random.seed(42) # Fixed seed for reproducibility during development, but maybe randomize in production? 
                    # The task says "Minimize number of distinct colors". A fixed seed might be risky if it fails.
                    # Let's use multiple random restarts.
    
    for max_colors in range(2, 15): # Limit to 15, unlikely to need more
        # Run SA for a fixed number of iterations
        # We want to find ANY valid solution with <= max_colors colors
        # Objective: Minimize violations. If 0, then we have a valid solution.
        
        found_valid = False
        best_sa_solution = None
        best_sa_violations = float('inf')
        
        # Multiple restarts
        for restart in range(20):
            # Initialize random colors
            colors = [random.randint(0, max_colors - 1) for _ in range(n)]
            
            # Initial evaluation
            viol, distinct = evaluate(colors)
            if viol == 0:
                found_valid = True
                best_sa_solution = colors[:]
                break
            
            T = 1.0
            T_min = 0.01
            alpha = 0.95
            steps = 1000
            current_viol = viol
            current_distinct = distinct
            best_local_viol = viol
            best_local_solution = colors[:]
            
            for step in range(steps):
                # Pick random vertex
                v = random.randint(0, n - 1)
                old_color = colors[v]
                # Pick new color
                new_color = random.randint(0, max_colors - 1)
                
                if old_color == new_color:
                    continue
                    
                colors[v] = new_color
                
                new_viol, new_distinct = evaluate(colors)
                
                delta = new_viol - current_viol
                
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_viol = new_viol
                    current_distinct = new_distinct
                    if new_viol < best_local_viol:
                        best_local_viol = new_viol
                        best_local_solution = colors[:]
                        if new_viol == 0:
                            break
                else:
                    colors[v] = old_color
                    
                T *= alpha
                
            if best_local_viol < best_sa_violations:
                best_sa_violations = best_local_viol
                best_sa_solution = best_local_solution
                
            if best_sa_violations == 0:
                found_valid = True
                break
                
        if found_valid:
            # We found a valid solution with max_colors
            # Check if it uses fewer distinct colors than max_colors
            distinct_used = len(set(best_sa_solution))
            if distinct_used < best_distinct:
                best_distinct = distinct_used
                best_solution = best_sa_solution
            # If we found a valid solution, we can try to see if we can do better with fewer colors
            # But the loop increases max_colors. 
            # Wait, the loop is `for max_colors in range(2, 15)`.
            # If we find a valid solution with max_colors=3, we should record it.
            # Then we try max_colors=4, etc. But we want the MINIMUM.
            # So if we find a valid solution with K colors, we should break?
            # No, because a valid solution with K colors might use K distinct colors.
            # We want the minimum distinct colors.
            # If we find a valid solution with max_colors=3, it uses <=3 colors.
            # Can we find one with 2?
            # The loop starts at 2. So if it fails at 2, it tries 3.
            # If it succeeds at 3, we have a solution with <=3 colors.
            # We should break because we are looking for the minimum K.
            # However, SA might fail to find a solution with K=2 even if one exists, due to local minima.
            # So we should not break immediately. We should continue to higher K to ensure we find *a* solution, 
            # but we want the *minimum* distinct colors.
            # Actually, if we find a valid solution with K=3, the distinct colors is <=3.
            # If we later find a valid solution with K=4, the distinct colors is <=4, which is worse.
            # So we should keep the best (lowest distinct) solution found.
            # But if we find a valid solution with K=2, that's the best possible (since 1 color is impossible for non-trivial graph).
            # So we can break if best_distinct == 2.
            
            if best_distinct == 2:
                break

    if best_solution is None:
        # Fallback: Use a greedy coloring that might not be optimal but is valid?
        # Or just return a valid high-color solution.
        # Let's do a simple greedy coloring to ensure we return something valid.
        colors = [-1] * n
        for v in range(n):
            used_by_neighbors = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used_by_neighbors.add(colors[u])
            c = 0
            while c in used_by_neighbors:
                c += 1
            colors[v] = c
        best_solution = colors
        best_distinct = len(set(colors))
        
    return [int(c) for c in best_solution]
# EVOLVE-BLOCK-END
