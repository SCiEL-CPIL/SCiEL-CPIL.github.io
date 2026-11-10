"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def has_bicolored_cycle(coloring, c1, c2):
        """Check if the subgraph induced by colors c1 and c2 has a cycle using union-find."""
        vertices = [i for i in range(n) if coloring[i] in (c1, c2)]
        if len(vertices) < 3:
            return False
        
        # Union-find
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False  # cycle detected
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Check all edges within the bicolored subgraph
        for u in vertices:
            for v in adj[u]:
                if v > u and coloring[v] in (c1, c2):
                    if not union(u, v):
                        return True
        return False
    
    def is_valid(coloring):
        """Check if coloring is valid: proper and no bicolored cycles."""
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored cycles
        colors_used = set(coloring)
        for c1, c2 in itertools.combinations(colors_used, 2):
            if has_bicolored_cycle(coloring, c1, c2):
                return False
        return True
    
    def count_violations(coloring):
        """Count total violations: improper edges + bicolored cycles."""
        violations = 0
        # Count improper edges
        for u, v in edges:
            if coloring[u] == coloring[v]:
                violations += 1
        
        # Count bicolored cycles
        colors_used = set(coloring)
        for c1, c2 in itertools.combinations(colors_used, 2):
            if has_bicolored_cycle(coloring, c1, c2):
                violations += 1
        return violations
    
    def greedy_coloring(num_colors):
        """Greedy coloring with random tie-breaking."""
        coloring = [-1] * n
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            used_by_neighbors = set(coloring[nb] for nb in adj[v] if coloring[nb] != -1)
            available = [c for c in range(num_colors) if c not in used_by_neighbors]
            if available:
                coloring[v] = random.choice(available)
            else:
                coloring[v] = 0  # fallback
        
        return coloring
    
    def local_search(coloring, num_colors, max_iters=5000):
        """Local search to reduce violations."""
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_violations == 0:
                break
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            old_color = coloring[v]
            
            # Try all possible colors
            best_new_color = old_color
            best_new_violations = current_violations
            
            for new_color in range(num_colors):
                if new_color == old_color:
                    continue
                
                # Check if this color is used by any neighbor
                if any(coloring[nb] == new_color for nb in adj[v]):
                    continue
                
                # Tentatively assign and count violations
                coloring[v] = new_color
                new_violations = count_violations(coloring)
                
                if new_violations < best_new_violations:
                    best_new_violations = new_violations
                    best_new_color = new_color
            
            if best_new_color != old_color:
                coloring[v] = best_new_color
                current_violations = best_new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                coloring[v] = old_color
        
        return best_coloring, best_violations
    
    def try_find_valid_coloring(num_colors, num_restarts=100):
        """Try to find a valid coloring with given number of colors."""
        best_result = None
        best_score = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Greedy initialization
            coloring = greedy_coloring(num_colors)
            
            # Local search
            coloring, violations = local_search(coloring, num_colors, max_iters=2000)
            
            if violations == 0:
                if is_valid(coloring):
                    return coloring
            
            if violations < best_score:
                best_score = violations
                best_result = coloring[:]
        
        return best_result
    
    # Try increasing number of colors
    for num_colors in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        result = try_find_valid_coloring(num_colors, num_restarts=50)
        if result is not None:
            if is_valid(result):
                return [int(c) for c in result]
    
    # Fallback: use backtracking for 4 colors
    def backtrack_coloring(num_colors):
        coloring = [-1] * n
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True
            
            v = order[idx]
            for c in range(num_colors):
                # Check neighbors
                allowed = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        allowed = False
                        break
                if not allowed:
                    continue
                
                coloring[v] = c
                if backtrack(idx + 1):
                    return True
                coloring[v] = -1
            
            return False
        
        if backtrack(0):
            if is_valid(coloring):
                return coloring
        return None
    
    result = backtrack_coloring(4)
    if result is not None:
        return [int(c) for c in result]
    
    # Final fallback: all distinct colors
    return [int(c) for c in range(n)]
# EVOLVE-BLOCK-END
