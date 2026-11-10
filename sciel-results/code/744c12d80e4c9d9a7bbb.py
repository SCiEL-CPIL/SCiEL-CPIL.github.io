"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def get_bicolored_cycles(coloring, max_color):
        """Returns list of (c1, c2, vertices_in_cycle) for each bicolored cycle found."""
        cycles = []
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                parent = list(range(n))
                rank = [0] * n
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return True
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return False
                
                cycle_verts = []
                has_cycle = False
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if coloring[u] != coloring[v]:
                            if union(u, v):
                                has_cycle = True
                                # Collect vertices in the component that has the cycle
                                # For simplicity, just mark that this pair has a cycle
                                break
                if has_cycle:
                    # Get all vertices with these colors to pick from
                    verts = [i for i in range(n) if coloring[i] in (c1, c2)]
                    cycles.append((c1, c2, verts))
        return cycles

    def is_valid(coloring):
        # Check proper
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check bicolored cycles
        cycles = get_bicolored_cycles(coloring, max(coloring)+1)
        return len(cycles) == 0

    def solve_with_k(k, seed_val=42):
        random.seed(seed_val)
        coloring = [random.randint(0, k-1) for _ in range(n)]
        
        # Local search
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        max_iters = 50000
        
        def get_cost(coloring):
            cost = 0
            # Proper coloring violations
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    cost += 10
            # Bicolored cycles
            cycles = get_bicolored_cycles(coloring, k)
            cost += len(cycles) * 5
            return cost

        current_cost = get_cost(coloring)
        best_coloring = coloring[:]
        best_cost = current_cost
        
        for i in range(max_iters):
            if current_cost == 0:
                break
            if T < T_min:
                break
                
            # Pick a vertex to recolor
            v = random.randint(0, n-1)
            old_c = coloring[v]
            new_c = random.randint(0, k-1)
            
            if old_c == new_c:
                continue
                
            coloring[v] = new_c
            new_cost = get_cost(coloring)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_coloring = coloring[:]
            else:
                coloring[v] = old_c
                
            T *= alpha
            
        if best_cost == 0:
            return best_coloring
        return None

    # Try k=3 first (though likely impossible for 4-regular)
    # Then k=4
    # Then k=5
    for k in [3, 4, 5, 6]:
        for seed_val in range(10):
            res = solve_with_k(k, seed_val)
            if res is not None:
                # Verify
                if is_valid(res):
                    # Try to reduce colors if possible? 
                    # If k=4 works, we return 4.
                    # We can try to map colors to 0..k-1 to ensure compactness
                    # The raw value is the number of distinct labels.
                    # If we used k colors, but some are unused, we should re-index.
                    unique_colors = sorted(set(res))
                    mapping = {c: i for i, c in enumerate(unique_colors)}
                    final_res = [mapping[c] for c in res]
                    return final_res
    
    # Fallback to greedy if local search fails
    def greedy_coloring(max_colors):
        coloring = [-1] * n
        for i in range(n):
            neighbor_colors = set()
            for nb in adj[i]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            for c in range(max_colors):
                if c not in neighbor_colors:
                    coloring[i] = c
                    break
            if coloring[i] == -1:
                coloring[i] = max_colors
                max_colors += 1
        return coloring

    # Use the original fix logic as fallback
    def fix_bicolored_cycles(coloring, max_colors, iterations=10000):
        coloring = coloring[:]
        for it in range(iterations):
            colors_used = set(coloring)
            cycle_found = False
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
                    parent = list(range(n))
                    rank = [0] * n
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        px, py = find(x), find(y)
                        if px == py:
                            return True
                        if rank[px] < rank[py]:
                            px, py = py, px
                        parent[py] = px
                        if rank[px] == rank[py]:
                            rank[px] += 1
                        return False
                    for u, v in edges:
                        if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                            if coloring[u] != coloring[v]:
                                if union(u, v):
                                    bicolored_verts = [x for x in range(n) if coloring[x] in (c1, c2)]
                                    if bicolored_verts:
                                        v_to_recolor = random.choice(bicolored_verts)
                                        neighbor_colors = set()
                                        for nb in adj[v_to_recolor]:
                                            neighbor_colors.add(coloring[nb])
                                        for new_c in range(max_colors):
                                            if new_c not in neighbor_colors and new_c != coloring[v_to_recolor]:
                                                coloring[v_to_recolor] = new_c
                                                cycle_found = True
                                                break
                                        if not cycle_found:
                                            coloring[v_to_recolor] = max_colors
                                            max_colors += 1
                                            cycle_found = True
                                    break
                if cycle_found:
                    break
            if not cycle_found:
                break
        return coloring, max_colors

    random.seed(42)
    coloring = greedy_coloring(10)
    coloring, k_used = fix_bicolored_cycles(coloring, 10, iterations=5000)
    if is_valid(coloring):
         unique_colors = sorted(set(coloring))
         mapping = {c: i for i, c in enumerate(unique_colors)}
         return [mapping[c] for c in coloring]
         
    # Absolute fallback
    return list(range(n))

# EVOLVE-BLOCK-END
