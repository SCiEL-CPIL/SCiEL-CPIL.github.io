"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_bicolored_forest(colors, c1, c2):
        """Check if subgraph induced by colors c1 and c2 is a forest. Returns (is_forest, cycle_vertices)."""
        vertices = set()
        for i, c in enumerate(colors):
            if c == c1 or c == c2:
                vertices.add(i)
        if len(vertices) < 2:
            return True, []
        # Union-find
        parent = {v: v for v in vertices}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False, (x, y)
            parent[rx] = ry
            return True, None
        for u, v in edges:
            if u in vertices and v in vertices:
                ok, cycle_info = union(u, v)
                if not ok:
                    # Found a cycle - return the vertices involved
                    return False, [u, v]
        return True, []
    
    def is_valid(colors):
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check bicolored forest constraint
        distinct = sorted(set(colors))
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                ok, _ = check_bicolored_forest(colors, distinct[i], distinct[j])
                if not ok:
                    return False
        return True
    
    def count_bicolored_cycles(colors, k):
        """Count total bicolored cycles across all pairs of colors in [0, k)."""
        total = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                ok, _ = check_bicolored_forest(colors, c1, c2)
                if not ok:
                    total += 1
        return total
    
    def find_bicolored_cycle_vertices(colors, c1, c2):
        """Find vertices in a bicolored cycle between colors c1 and c2."""
        vertices = set()
        for i, c in enumerate(colors):
            if c == c1 or c == c2:
                vertices.add(i)
        if len(vertices) < 2:
            return []
        
        # Build subgraph edges
        sub_edges = []
        for u, v in edges:
            if u in vertices and v in vertices:
                sub_edges.append((u, v))
        
        # Find cycle using DFS
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        cycle_vertices = []
        for u, v in sub_edges:
            if not union(u, v):
                # Found a cycle - collect vertices in this connected component
                root = find(u)
                comp = [x for x in vertices if find(x) == root]
                cycle_vertices = comp
                break
        
        return cycle_vertices
    
    # Constraint satisfaction approach: start with a valid coloring, then try to reduce colors
    # by merging color classes while fixing bicolored cycles
    
    # Start with a proper k-coloring using backtracking
    def backtrack_coloring(k, vertex_order, colors):
        """Backtrack to find a proper k-coloring."""
        if time.time() > deadline:
            return False
        
        # Find first uncolored vertex
        idx = -1
        for i in range(n):
            if colors[i] is None:
                idx = i
                break
        
        if idx == -1:
            # All colored, check validity
            return is_valid(colors)
        
        v = vertex_order[idx] if idx < len(vertex_order) else idx
        
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] is not None:
                forbidden.add(colors[nb])
        
        for c in range(k):
            if c in forbidden:
                continue
            
            colors[v] = c
            if backtrack_coloring(k, vertex_order, colors):
                return True
            
            colors[v] = None
        
        return False
    
    # Try to find a valid coloring with as few colors as possible
    best_result = None
    best_num_colors = n + 1
    
    # Strategy: Start with more colors, then iteratively merge colors and fix cycles
    # This is the constraint-satisfaction approach
    
    # First, get a valid 4-coloring (or fewer)
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Try multiple random orderings
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            vertex_order = list(range(n))
            random.shuffle(vertex_order)
            # Sort by degree descending for better pruning
            vertex_order.sort(key=lambda x: -len(adj[x]))
            
            colors = [None] * n
            if backtrack_coloring(k, vertex_order, colors):
                if is_valid(colors):
                    best_result = colors[:]
                    best_num_colors = k
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: use all distinct colors
        best_result = list(range(n))
        best_num_colors = n
        return [int(c) for c in best_result]
    
    # Now try to reduce colors by merging and fixing bicolored cycles
    # The key insight: if we merge two color classes, we may create bicolored cycles.
    # We can fix these by recoloring one vertex in each cycle to a third color.
    
    def try_merge_and_fix(colors, c_merge, c_target, k_max):
        """
        Try to merge color c_merge into c_target.
        If this creates bicolored cycles, try to fix them by recoloring.
        Returns a new valid coloring or None.
        """
        if time.time() > deadline:
            return None
        
        new_colors = colors[:]
        
        # Merge: change all c_merge vertices to c_target
        for i in range(n):
            if new_colors[i] == c_merge:
                new_colors[i] = c_target
        
        # Check for bicolored cycles
        distinct = sorted(set(new_colors))
        has_cycle = False
        cycle_info = None
        
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                ok, _ = check_bicolored_forest(new_colors, distinct[i], distinct[j])
                if not ok:
                    has_cycle = True
                    cycle_info = (distinct[i], distinct[j])
                    break
            if has_cycle:
                break
        
        if not has_cycle:
            # Merge was successful without creating cycles
            return new_colors
        
        # Try to fix bicolored cycles by recoloring vertices
        # For each bicolored cycle, try recoloring one vertex to a third color
        max_fix_attempts = 50
        for attempt in range(max_fix_attempts):
            if time.time() > deadline:
                return None
            
            # Find a bicolored cycle
            cycle_found = False
            for i in range(len(distinct)):
                for j in range(i+1, len(distinct)):
                    ok, _ = check_bicolored_forest(new_colors, distinct[i], distinct[j])
                    if not ok:
                        # Found a cycle between colors distinct[i] and distinct[j]
                        cycle_verts = find_bicolored_cycle_vertices(new_colors, distinct[i], distinct[j])
                        if cycle_verts:
                            # Try to recolor one of these vertices to break the cycle
                            for v in cycle_verts:
                                # Try each available color
                                for c_new in range(k_max):
                                    if c_new == new_colors[v]:
                                        continue
                                    
                                    # Check if v can take color c_new (proper coloring)
                                    valid = True
                                    for nb in adj[v]:
                                        if new_colors[nb] == c_new:
                                            valid = False
                                            break
                                    
                                    if not valid:
                                        continue
                                    
                                    # Try this recoloring
                                    old_color = new_colors[v]
                                    new_colors[v] = c_new
                                    
                                    # Check if this fixes the cycle
                                    distinct_now = sorted(set(new_colors))
                                    still_has_cycle = False
                                    for a in range(len(distinct_now)):
                                        for b in range(a+1, len(distinct_now)):
                                            ok2, _ = check_bicolored_forest(new_colors, distinct_now[a], distinct_now[b])
                                            if not ok2:
                                                still_has_cycle = True
                                                break
                                        if still_has_cycle:
                                            break
                                    
                                    if not still_has_cycle:
                                        # Fixed it!
                                        return new_colors
                                    
                                    # Undo
                                    new_colors[v] = old_color
                            
                            # If we found a cycle but couldn't fix it with one recoloring,
                            # try a different cycle or give up
                            cycle_found = True
                            break
                if cycle_found:
                    break
            
            if not cycle_found:
                break
        
        return None
    
    # Iteratively try to reduce the number of colors
    current_colors = best_result[:]
    current_k = best_num_colors
    
    # Try to reduce by merging pairs of colors
    for reduction_pass in range(10):
        if time.time() > deadline:
            break
        
        if current_k <= 2:
            break
        
        merged = False
        distinct_colors = sorted(set(current_colors))
        
        for i in range(len(distinct_colors)):
            for j in range(i+1, len(distinct_colors)):
                if time.time() > deadline:
                    break
                
                c_merge = distinct_colors[j]
                c_target = distinct_colors[i]
                
                # Try to merge c_merge into c_target
                result = try_merge_and_fix(current_colors, c_merge, c_target, current_k)
                
                if result is not None:
                    # Check if this actually reduces the number of distinct colors
                    new_distinct = set(result)
                    if len(new_distinct) < len(set(current_colors)):
                        current_colors = result
                        current_k = len(new_distinct)
                        merged = True
                        break
            
            if merged:
                break
        
        if not merged:
            break
    
    # Final validation
    if is_valid(current_colors):
        best_result = current_colors
        best_num_colors = len(set(current_colors))
    
    # Ensure all are Python ints
    best_result = [int(c) for c in best_result]
    return best_result

# EVOLVE-BLOCK-END
