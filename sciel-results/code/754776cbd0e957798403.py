"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

# Build adjacency list
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def has_bicolored_cycle(coloring, k):
    """Check if any pair of colors forms a cycle in their induced subgraph."""
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        return True
    return False

def count_distinct_colors(coloring):
    return len(set(coloring))

def validate(coloring):
    """Check all constraints: proper coloring and no bicolored cycles."""
    k = count_distinct_colors(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    if has_bicolored_cycle(coloring, k):
        return False
    return True

def construct():
    """
    Find a valid coloring of the degree-4 graph with minimum distinct colors
    such that no two colors form a bicolored cycle.
    """
    deadline = 85.0
    start = time.time()
    
    # The graph is a 50-cycle with chords. Let's try to find a good coloring.
    # Since it's degree 4, 3 colors might not suffice for proper coloring, 
    # but let's try 3, 4, 5 colors.
    
    best_coloring = None
    best_k = n + 1
    
    # Try k from 3 upward
    for k in range(3, 8):
        if time.time() - start > deadline:
            break
        
        # Try multiple random restarts with greedy + local search
        for seed in range(20):
            if time.time() - start > deadline:
                break
            
            random.seed(seed)
            coloring = [-1] * n
            
            # Greedy: order by degree (all same here), then random
            order = list(range(n))
            random.shuffle(order)
            
            success = True
            for v in order:
                # Try each color
                for c in range(k):
                    coloring[v] = c
                    if validate(coloring):
                        break
                else:
                    coloring[v] = -1
                    success = False
                    break
            
            if not success:
                continue
            
            k_used = count_distinct_colors(coloring)
            if k_used < best_k:
                best_k = k_used
                best_coloring = list(coloring)
            
            # Local search to reduce colors further
            if time.time() - start < deadline - 3.0:
                coloring = list(best_coloring)
                improved = True
                while improved and time.time() - start < deadline - 3.0:
                    improved = False
                    k_cur = count_distinct_colors(coloring)
                    if k_cur <= 3:
                        break
                    
                    # Try to eliminate one color
                    color_counts = {}
                    for v in range(n):
                        c = coloring[v]
                        color_counts[c] = color_counts.get(c, 0) + 1
                    
                    min_count = min(color_counts.values())
                    min_colors = [c for c, cnt in color_counts.items() if cnt == min_count]
                    random.shuffle(min_colors)
                    
                    for target_color in min_colors:
                        vertices_to_reassign = [v for v in range(n) if coloring[v] == target_color]
                        random.shuffle(vertices_to_reassign)
                        
                        all_success = True
                        for v in vertices_to_reassign:
                            old_c = coloring[v]
                            placed = False
                            for new_c in range(k_cur):
                                if new_c == target_color:
                                    continue
                                coloring[v] = new_c
                                if validate(coloring):
                                    placed = True
                                    break
                            if not placed:
                                coloring[v] = old_c
                                all_success = False
                                break
                        
                        if all_success:
                            improved = True
                            break
            
            k_used = count_distinct_colors(coloring)
            if k_used < best_k:
                best_k = k_used
                best_coloring = list(coloring)
    
    if best_coloring is not None:
        return list(best_coloring)
    
    # Fallback: all distinct colors
    return list(range(n))

# EVOLVE-BLOCK-END
