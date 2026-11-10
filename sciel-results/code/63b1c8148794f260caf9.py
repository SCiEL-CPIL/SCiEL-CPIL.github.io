"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 40.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge indices for fast lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def full_check(colors):
        """Full validity check: adjacency + bicolored forest for all color pairs."""
        # Rule 1: adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Rule 2: for every pair of colors, induced subgraph is a forest
        color_to_verts = defaultdict(set)
        for i, c in enumerate(colors):
            color_to_verts[c].add(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_set = color_to_verts[c1] | color_to_verts[c2]
                # Check for cycles using union-find
                parent = {v: v for v in verts_set}
                rank = {v: 0 for v in verts_set}
                
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
                    if u in verts_set and v in verts_set:
                        if not union(u, v):
                            return False
        return True
    
    def check_pair_forest(c1, c2, colors, color_to_verts):
        """Check if the subgraph induced by vertices with color c1 or c2 is a forest."""
        verts_set = color_to_verts[c1] | color_to_verts[c2]
        if len(verts_set) <= 1:
            return True
        
        parent = {v: v for v in verts_set}
        rank = {v: 0 for v in verts_set}
        
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
            if u in verts_set and v in verts_set:
                if not union(u, v):
                    return False
        return True
    
    def try_color_change(v, new_c, colors, color_to_verts):
        """
        Try to change vertex v's color to new_c.
        Returns True if the change maintains validity (adjacency + bicolored forest).
        Does NOT modify colors; just checks feasibility.
        """
        old_c = colors[v]
        if new_c == old_c:
            return True
        
        # Check adjacency: no neighbor has new_c
        for u in adj[v]:
            if colors[u] == new_c:
                return False
        
        # Check bicolored forest constraints:
        # The only pairs affected are those involving old_c or new_c
        # Specifically: (old_c, old_c), (old_c, new_c), (new_c, new_c)
        # But since we're removing v from old_c group and adding to new_c group,
        # we need to check:
        # 1. Pair (old_c, new_c): v is in both groups conceptually? No.
        #    v leaves old_c, enters new_c. So we need to check:
        #    - The subgraph induced by (old_c \ {v}) ∪ (new_c ∪ {v}) is a forest
        #    - The subgraph induced by (old_c \ {v}) ∪ (old_c \ {v}) i.e. old_c without v is a forest (trivially, removing a vertex can't create a cycle)
        #    - The subgraph induced by (new_c ∪ {v}) ∪ (new_c ∪ {v}) i.e. new_c with v added is a forest
        
        # Actually, let's think more carefully. The pairs that could be affected:
        # For any color c, the pair (c, c) means the subgraph induced by vertices of color c.
        # Pair (c1, c2) for c1 != c2 means the subgraph induced by vertices of color c1 OR c2.
        
        # When v changes from old_c to new_c:
        # - For pair (old_c, old_c): v is removed. Removing a vertex can't create a cycle, so this is fine.
        # - For pair (new_c, new_c): v is added. We need to check that adding v to new_c doesn't create a cycle in the new_c subgraph.
        # - For pair (old_c, new_c): v was in old_c group, now in new_c group. The induced set changes:
        #   Before: (old_c) ∪ (new_c)
        #   After: (old_c \ {v}) ∪ (new_c ∪ {v})
        #   The vertex set is the same! So the induced subgraph is the same. No need to check.
        # - For any other pair (c, d) where c,d are neither old_c nor new_c: unaffected.
        # - For pair (old_c, c) where c != new_c, c != old_c: v is removed from the set. Can't create cycle.
        # - For pair (new_c, c) where c != old_c, c != new_c: v is added to the set. Need to check!
        
        # So we need to check:
        # 1. new_c subgraph with v added (pair (new_c, new_c))
        # 2. For each other color c (c != old_c, c != new_c): pair (new_c, c) with v added
        
        # But checking all pairs (new_c, c) is expensive. Let's be smarter:
        # Actually, the constraint is that for EVERY pair of colors, the induced subgraph is a forest.
        # When we add v to new_c, the only new edges involving v are to its neighbors.
        # v's neighbors have various colors. For each neighbor u with color c_u:
        #   The edge (v, u) is in the induced subgraph of pair (new_c, c_u) if c_u != new_c,
        #   or in pair (new_c, new_c) if c_u == new_c (but we already checked adjacency, so no neighbor has new_c).
        
        # So for each neighbor u of v with color c_u (where c_u != new_c, which is all of them since adjacency is satisfied):
        #   The edge (v, u) is part of the induced subgraph for pair (new_c, c_u).
        #   We need to verify that adding this edge doesn't create a cycle in that subgraph.
        
        # Efficient approach: for the new_c group, build the union-find structure and check
        # that v's addition doesn't create a cycle. Then for each neighbor's color, check
        # the cross-pair.
        
        # Let's do a targeted check:
        # Build the vertex set for pair (new_c, new_c) = all vertices with color new_c plus v
        # Check it's a forest.
        
        # Then for each distinct color c_u among v's neighbors (c_u != new_c):
        #   Build vertex set for pair (new_c, c_u) = (vertices with color new_c) ∪ {v} ∪ (vertices with color c_u)
        #   Check it's a forest.
        
        # This is still potentially expensive but much less than full check.
        
        # Even better: just do a full check. With 50 vertices and at most ~10 colors,
        # the full check is O(k^2 * |E|) = O(100 * 80) = 8000 union-find ops, which is fast.
        
        # Let's just do the full check for simplicity and correctness.
        test_colors = colors[:]
        test_colors[v] = new_c
        return full_check(test_colors)
    
    # Strategy: try k from 2 upward
    # For each k, use local search with structure-aware initialization
    
    # Precompute: for each k, a good initial coloring
    # The graph has a natural structure: spine edges i->i+1 and chord edges.
    # A modular coloring i % k works well for the spine if k >= 2.
    
    for k in range(2, 15):
        if time.time() > deadline:
            break
        
        best_found = None
        
        # Multiple restarts with different initializations
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Modular by k
                colors = [i % k for i in range(n)]
            elif restart == 1:
                # Modular by k with offset
                colors = [(i + 1) % k for i in range(n)]
            elif restart == 2:
                # Two-phase: first k/2 colors, then second k/2
                colors = [(i % (k // 2)) if i < n // 2 else (k // 2 + i % (k // 2)) for i in range(n)] if k >= 4 else [i % k for i in range(n)]
            elif restart % 2 == 0:
                # Random
                colors = [random.randint(0, k - 1) for _ in range(n)]
            else:
                # Random with bias toward low indices
                colors = [random.randint(0, min(i, k - 1)) for i in range(n)]
            
            # First, fix adjacency violations greedily
            for _ in range(100):
                changed = False
                for v in range(n):
                    # Check if v has a neighbor with same color
                    conflict = False
                    for u in adj[v]:
                        if colors[u] == colors[v]:
                            conflict = True
                            break
                    if conflict:
                        # Try to find a color for v that doesn't conflict
                        for c in range(k):
                            if c == colors[v]:
                                continue
                            ok = True
                            for u in adj[v]:
                                if colors[u] == c:
                                    ok = False
                                    break
                            if ok:
                                colors[v] = c
                                changed = True
                                break
                        if not changed:
                            # Stuck, try random
                            colors[v] = random.randint(0, k - 1)
                            changed = True
                if not changed:
                    break
            
            # Now do local search: try to reduce number of distinct colors used
            # while maintaining validity
            
            # Check if current coloring is valid
            if full_check(colors):
                distinct = len(set(colors))
                if best_found is None or distinct < len(set(best_found)):
                    best_found = colors[:]
                    if distinct <= k:
                        # Great, we found a valid coloring with <= k colors
                        # Try to reduce further
                        pass
            
            # Local search: iteratively try to change colors to reduce distinct count
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                current_distinct = len(set(colors))
                
                # Try to merge colors: pick a vertex with a "rare" color and try to change it
                color_counts = defaultdict(int)
                for c in colors:
                    color_counts[c] += 1
                
                # Sort colors by frequency (ascending) - try to eliminate rare colors
                sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])
                
                improved = False
                for target_color in sorted_colors:
                    if time.time() > deadline:
                        break
                    
                    # Try to change one vertex with target_color to another color
                    vertices_with_color = [v for v in range(n) if colors[v] == target_color]
                    random.shuffle(vertices_with_color)
                    
                    for v in vertices_with_color:
                        # Try changing v to each other color
                        for new_c in range(k):
                            if new_c == target_color:
                                continue
                            # Check if this is feasible
                            if try_color_change(v, new_c, colors, None):
                                colors[v] = new_c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    # Stuck, try a random perturbation
                    if random.random() < 0.3:
                        v = random.randint(0, n - 1)
                        new_c = random.randint(0, k - 1)
                        if new_c != colors[v]:
                            if try_color_change(v, new_c, colors, None):
                                colors[v] = new_c
                    else:
                        break
            
            # Final check
            if full_check(colors):
                distinct = len(set(colors))
                if best_found is None or distinct < len(set(best_found)):
                    best_found = colors[:]
        
        if best_found is not None:
            distinct = len(set(best_found))
            # Verify one more time
            if full_check(best_found):
                # Remap colors to 0..distinct-1
                color_map = {}
                next_id = 0
                remapped = []
                for c in best_found:
                    if c not in color_map:
                        color_map[c] = next_id
                        next_id += 1
                    remapped.append(color_map[c])
                return [int(c) for c in remapped]
    
    # Fallback: return all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
