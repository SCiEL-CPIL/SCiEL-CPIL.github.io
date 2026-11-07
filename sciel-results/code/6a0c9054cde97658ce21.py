"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def check_validity(colors):
        # Check adjacent different
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Check bicolored cycles
        used_colors = set(colors)
        for i in used_colors:
            for j in used_colors:
                if i >= j:
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
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True

                has_cycle = False
                for u, v in edges:
                    cu, cv = colors[u], colors[v]
                    if (cu == i and cv == j) or (cu == j and cv == i):
                        if not union(u, v):
                            has_cycle = True
                            break
                if has_cycle:
                    return False
        return True

    def get_cost(colors):
        cost = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                cost += 100
        
        used_colors = set(colors)
        for i in used_colors:
            for j in used_colors:
                if i >= j:
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
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True

                for u, v in edges:
                    cu, cv = colors[u], colors[v]
                    if (cu == i and cv == j) or (cu == j and cv == i):
                        if not union(u, v):
                            cost += 1000
                            break
        return cost

    def local_search(max_colors, iterations=2000):
        colors = [random.randint(0, max_colors - 1) for _ in range(n)]
        current_cost = get_cost(colors)
        best_colors = colors[:]
        best_cost = current_cost
        
        temperature = 1000.0
        cooling = 0.99
        
        for _ in range(iterations):
            v = random.randint(0, n - 1)
            old_color = colors[v]
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            colors[v] = new_color
            new_cost = get_cost(colors)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                colors[v] = old_color
                
            temperature *= cooling
            
        if best_cost == 0:
            return best_colors
        return None

    # Try to find a solution with 3 colors
    for k in [3, 4, 5]:
        for attempt in range(10):
            solution = local_search(k, iterations=2000)
            if solution:
                if check_validity(solution):
                    return [int(c) for c in solution]
        
    # Fallback: simple greedy
    for k in [3, 4, 5, 6]:
        colors = [-1] * n
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        success = True
        for v in order:
            possible = False
            for c in range(k):
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                colors[v] = c
                possible = True
                break
            if not possible:
                success = False
                break
        if success:
            if check_validity(colors):
                return [int(c) for c in colors]
    
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
