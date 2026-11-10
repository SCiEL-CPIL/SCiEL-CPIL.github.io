"""Rewritten program for bicolored-cycle-free coloring using local search."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _find_cycles_in_pair(coloring, c1, c2, n, edges):
    """
    Returns a set of vertices involved in cycles in the subgraph induced by colors c1 and c2.
    Uses Union-Find to detect cycles.
    """
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

    cycle_vertices = set()
    
    # We need to process edges that connect vertices of c1 and c2.
    # If union returns False, a cycle is formed.
    # The cycle consists of the path in the tree connecting x and y, plus the edge (x,y).
    # To find the vertices in the cycle, we can trace back from x and y to their LCA in the union-find tree?
    # Union-find doesn't store the tree structure explicitly in a way that makes LCA easy without parent pointers.
    # However, we just need *a* vertex to recolor. 
    # A simpler heuristic: if a cycle is detected, we can pick a random vertex from the connected component 
    # that is part of the cycle. 
    # Actually, if we just detect *any* cycle, we can recolor a random vertex from the set of vertices 
    # in that component that have color c1 or c2. This is likely to break the cycle.
    
    # Let's collect components that have cycles.
    # To be more precise, we can store the edges that formed the cycle.
    
    # For simplicity and speed, if union(x,y) fails, we know x and y are in the same component.
    # We can add x and y to a list of "conflict" vertices.
    
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
            if not union(u, v):
                # Cycle found
                cycle_vertices.add(u)
                cycle_vertices.add(v)
    
    return cycle_vertices

def _count_violations(coloring, n, edges, k):
    """
    Returns (proper_violations, bicolored_cycle_vertices).
    proper_violations: list of edges (u,v) where coloring[u] == coloring[v].
    bicolored_cycle_vertices: set of vertices involved in bicolored cycles.
    """
    proper_violations = []
    for u, v in edges:
        if coloring[u] == coloring[v]:
            proper_violations.append((u, v))
            
    # Get distinct colors used
    used_colors = set(coloring)
    
    bicolored_cycle_vertices = set()
    
    # Check all pairs of used colors
    colors_list = list(used_colors)
    for i in range(len(colors_list)):
        for j in range(i + 1, len(colors_list)):
            c1, c2 = colors_list[i], colors_list[j]
            # Optimization: only check if there is at least one edge between these colors
            # But _find_cycles_in_pair handles this by iterating edges.
            # We can add a quick check: if no edge connects c1 and c2, skip.
            has_edge = False
            for u, v in edges:
                if (coloring[u] == c1 and coloring[v] == c2) or (coloring[u] == c2 and coloring[v] == c1):
                    has_edge = True
                    break
            if not has_edge:
                continue
                
            cv = _find_cycles_in_pair(coloring, c1, c2, n, edges)
            bicolored_cycle_vertices.update(cv)
            
    return proper_violations, bicolored_cycle_vertices

def _local_search(n, edges, k, deadline):
    """
    Attempts to find a valid coloring with k colors using local search.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Initialize randomly
    coloring = [random.randint(0, k - 1) for _ in range(n)]
    
    best_coloring = None
    best_score = float('inf')
    
    # Simulated Annealing / Min-Conflicts hybrid
    # Score = len(proper_violations) * 10 + len(bicolored_cycle_vertices)
    # We want to minimize this.
    
    max_steps = 50000
    step = 0
    
    while step < max_steps and time.time() < deadline:
        proper_violations, bicolored_cycle_vertices = _count_violations(coloring, n, edges, k)
        
        score = len(proper_violations) * 10 + len(bicolored_cycle_vertices)
        
        if score < best_score:
            best_score = score
            best_coloring = coloring[:]
            
        if score == 0:
            return best_coloring
            
        # Choose a vertex to recolor
        # Priority: 
        # 1. Vertices in proper violations
        # 2. Vertices in bicolored cycles
        
        candidate_vertices = set()
        for u, v in proper_violations:
            candidate_vertices.add(u)
            candidate_vertices.add(v)
        candidate_vertices.update(bicolored_cycle_vertices)
        
        if not candidate_vertices:
            # Should not happen if score > 0, but just in case
            candidate_vertices = set(range(n))
            
        v = random.choice(list(candidate_vertices))
        
        # Choose a new color
        # Avoid colors of neighbors to fix proper violations
        neighbor_colors = set(coloring[u] for u in adj[v])
        available_colors = [c for c in range(k) if c not in neighbor_colors]
        
        if not available_colors:
            # If no available colors (shouldn't happen if k > max_degree), pick random
            new_color = random.randint(0, k - 1)
        else:
            # Heuristic: pick a color that might break bicolored cycles
            # Just pick random from available for now
            new_color = random.choice(available_colors)
            
        coloring[v] = new_color
        step += 1
        
    if best_coloring and best_score == 0:
        return best_coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = P['n']
    edges = P['edges']
    
    # Try increasing number of colors
    # The graph is 4-regular. 
    # Proper coloring needs at least 4 colors (chromatic number).
    # Bicolored forest constraint is stricter.
    # Let's try from 4 upwards.
    
    for k in range(4, n + 1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for each k
        found = None
        for restart in range(20):
            if time.time() > deadline:
                break
            result = _local_search(n, edges, k, deadline)
            if result is not None:
                # Verify
                pv, cv = _count_violations(result, n, edges, k)
                if len(pv) == 0 and len(cv) == 0:
                    found = result
                    break
        
        if found is not None:
            return list(found)
            
    # Fallback: use all different colors (always valid)
    return list(range(n))
