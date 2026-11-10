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
    
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
            if not union(u, v):
                cycle_vertices.add(u)
                cycle_vertices.add(v)
    
    return cycle_vertices

def _count_violations(coloring, n, edges, k):
    """
    Returns (proper_violations, bicolored_cycle_vertices).
    """
    proper_violations = []
    for u, v in edges:
        if coloring[u] == coloring[v]:
            proper_violations.append((u, v))
            
    used_colors = set(coloring)
    bicolored_cycle_vertices = set()
    
    colors_list = list(used_colors)
    for i in range(len(colors_list)):
        for j in range(i + 1, len(colors_list)):
            c1, c2 = colors_list[i], colors_list[j]
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

def _local_search(n, edges, k, deadline, adj):
    """
    Attempts to find a valid coloring with k colors using local search.
    Uses incremental edge-pair tracking for speed.
    """
    # Precompute: for each edge, store (u, v)
    # We'll track edge_count[(c1, c2)] for unordered color pairs
    
    # Initialize
    coloring = [random.randint(0, k - 1) for _ in range(n)]
    
    # Build initial edge color-pair counts
    # edge_pair_count[(c1, c2)] where c1 < c2: number of edges with these two colors
    # Also track which edges have same color (proper violations)
    def _build_state(coloring):
        edge_pair_count = defaultdict(int)
        proper_viol_count = 0
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if cu == cv:
                proper_viol_count += 1
            else:
                key = (min(cu, cv), max(cu, cv))
                edge_pair_count[key] += 1
        return edge_pair_count, proper_viol_count
    
    edge_pair_count, proper_viol_count = _build_state(coloring)
    
    best_coloring = None
    best_score = float('inf')
    
    max_steps = 100000
    step = 0
    
    while step < max_steps and time.time() < deadline:
        # Compute score: proper violations + bicolored cycle count
        # For bicolored cycles: for each color pair (c1,c2), if edge_count > vertex_count_in_pair - components, there's a cycle
        # Actually simpler: if edge_count >= vertex_count for that pair's induced subgraph, likely has cycle
        # Let's just check pairs where edge_count > 0 and do a quick cycle check
        
        score = proper_viol_count * 10
        
        # Check for bicolored cycles only in pairs that have edges
        cycle_vertices = set()
        for (c1, c2), cnt in edge_pair_count.items():
            if cnt > 0:
                # Quick check: if the subgraph on colors c1,c2 has a cycle
                # A graph with V vertices and E edges has a cycle if E >= V
                # Count vertices with color c1 or c2
                v_count = sum(1 for x in range(n) if coloring[x] in (c1, c2))
                if cnt >= v_count:
                    # Possible cycle - do full check
                    cv = _find_cycles_in_pair(coloring, c1, c2, n, edges)
                    if cv:
                        cycle_vertices.update(cv)
                        score += len(cv)
        
        if score < best_score:
            best_score = score
            best_coloring = coloring[:]
            
        if score == 0:
            return best_coloring
            
        # Choose vertex to recolor
        candidate_vertices = set()
        
        # Add vertices from proper violations
        for u, v in edges:
            if coloring[u] == coloring[v]:
                candidate_vertices.add(u)
                candidate_vertices.add(v)
        
        # Add vertices from bicolored cycles
        candidate_vertices.update(cycle_vertices)
        
        if not candidate_vertices:
            candidate_vertices = set(range(n))
            
        v = random.choice(list(candidate_vertices))
        
        old_color = coloring[v]
        
        # Choose new color
        neighbor_colors = set(coloring[u] for u in adj[v])
        available_colors = [c for c in range(k) if c not in neighbor_colors]
        
        if not available_colors:
            new_color = random.randint(0, k - 1)
        else:
            # Heuristic: prefer colors that break cycles
            # Try each available color and pick the one that minimizes new cycle risk
            best_new_color = available_colors[0]
            best_new_score = float('inf')
            
            for nc in available_colors:
                # Simulate: what's the impact?
                # For each neighbor, the edge (v, neighbor) changes from (old_color, neighbor_color) to (nc, neighbor_color)
                # This affects edge_pair_count
                # Quick heuristic: prefer colors not already heavily used in cycle-forming pairs
                nc_score = 0
                for u in adj[v]:
                    nc_u = coloring[u]
                    if nc_u == nc:
                        nc_score += 10  # would create proper violation
                    else:
                        key = (min(nc, nc_u), max(nc, nc_u))
                        nc_score += edge_pair_count.get(key, 0)
                if nc_score < best_new_score:
                    best_new_score = nc_score
                    best_new_color = nc
            
            new_color = best_new_color
            
        # Update state incrementally
        if new_color != old_color:
            # Remove old edges from count
            for u in adj[v]:
                cu = coloring[u]
                if cu == old_color:
                    proper_viol_count -= 1
                else:
                    key = (min(old_color, cu), max(old_color, cu))
                    edge_pair_count[key] -= 1
                    if edge_pair_count[key] <= 0:
                        del edge_pair_count[key]
            
            # Add new edges to count
            for u in adj[v]:
                cu = coloring[u]
                if cu == new_color:
                    proper_viol_count += 1
                else:
                    key = (min(new_color, cu), max(new_color, cu))
                    edge_pair_count[key] += 1
            
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
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try increasing number of colors
    # Start with k=3 (might work for this graph)
    for k in range(3, n + 1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for each k
        found = None
        max_restarts = 30 if k <= 4 else 15
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            result = _local_search(n, edges, k, deadline, adj)
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
