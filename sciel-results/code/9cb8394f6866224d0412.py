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

    def find_bicolored_cycle(coloring, c1, c2):
        """Find a bicolored cycle using union-find. Returns (cycle_verts, edge) or (None, None)."""
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
        
        # Only consider edges where both endpoints have color c1 or c2
        for u, v in edges:
            if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                if coloring[u] != coloring[v]:
                    if union(u, v):
                        # Cycle found - collect all vertices with these two colors in this component
                        root = find(u)
                        cycle_verts = []
                        for i in range(n):
                            if coloring[i] in (c1, c2) and find(i) == root:
                                cycle_verts.append(i)
                        return cycle_verts, (u, v)
        return None, None

    def count_bicolored_cycles(coloring):
        """Count number of bicolored cycle pairs."""
        colors_used = sorted(set(coloring))
        count = 0
        for i, c1 in enumerate(colors_used):
            for c2 in colors_used[i+1:]:
                verts, _ = find_bicolored_cycle(coloring, c1, c2)
                if verts is not None:
                    count += 1
        return count

    def is_valid(coloring):
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check no bicolored cycles
        return count_bicolored_cycles(coloring) == 0

    def solve_constraint(k, seed_val):
        """
        Constraint satisfaction approach:
        1. Start with random proper coloring using k colors
        2. Iteratively find and break bicolored cycles by recoloring a vertex in the cycle
        3. Use greedy initialization biased toward avoiding bicolored cycles
        """
        random.seed(seed_val)
        
        # Greedy initialization with bicolored cycle awareness
        coloring = [-1] * n
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            neighbor_colors = set()
            for nb in adj[idx]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            
            # Try colors in random order
            candidates = list(range(k))
            random.shuffle(candidates)
            
            best_color = None
            best_score = float('inf')
            
            for c in candidates:
                if c in neighbor_colors:
                    continue
                # Tentatively assign and check if it creates a bicolored cycle
                coloring[idx] = c
                creates_cycle = False
                for c2 in neighbor_colors:
                    if c != c2:
                        verts, _ = find_bicolored_cycle(coloring, c, c2)
                        if verts is not None:
                            creates_cycle = True
                            break
                coloring[idx] = -1
                
                if not creates_cycle:
                    best_color = c
                    break
                else:
                    # Score based on how many bicolored cycles this would create
                    cycle_count = 0
                    for c2 in neighbor_colors:
                        if c != c2:
                            # Count edges in potential bicolored subgraph
                            cycle_count += 1
                    if cycle_count < best_score:
                        best_score = cycle_count
                        best_color = c
            
            if best_color is None:
                # Fallback: use first available color
                for c in candidates:
                    if c not in neighbor_colors:
                        best_color = c
                        break
                if best_color is None:
                    best_color = candidates[0]
            
            coloring[idx] = best_color
        
        # Now iteratively fix any remaining bicolored cycles
        max_iters = 10000
        for it in range(max_iters):
            # Find any bicolored cycle
            colors_used = sorted(set(coloring))
            cycle_found = False
            for i, c1 in enumerate(colors_used):
                for c2 in colors_used[i+1:]:
                    verts, edge = find_bicolored_cycle(coloring, c1, c2)
                    if verts is not None:
                        # Recolor a vertex in the cycle to a third color
                        # Prefer vertices that have a neighbor with a color not in {c1, c2}
                        # so we can recolor to that color without creating new violations
                        
                        # Score vertices in the cycle
                        scored_verts = []
                        for v in verts:
                            neighbor_colors = set()
                            for nb in adj[v]:
                                neighbor_colors.add(coloring[nb])
                            # Available colors that don't conflict with neighbors
                            available = [c for c in range(k) if c not in neighbor_colors]
                            # Prefer colors not in {c1, c2} to break the bicolored subgraph
                            preferred = [c for c in available if c not in (c1, c2)]
                            if preferred:
                                scored_verts.append((v, preferred[0], 0))
                            elif available:
                                scored_verts.append((v, available[0], 1))
                        
                        if scored_verts:
                            # Pick the vertex with best score (prefer those that can use a third color)
                            scored_verts.sort(key=lambda x: x[2])
                            v_to_recolor, new_color, _ = scored_verts[0]
                            coloring[v_to_recolor] = new_color
                            cycle_found = True
                            break
                if cycle_found:
                    break
            
            if not cycle_found:
                break
        
        # Check if valid
        if is_valid(coloring):
            # Compact the colors
            unique_colors = sorted(set(coloring))
            mapping = {c: i for i, c in enumerate(unique_colors)}
            return [mapping[c] for c in coloring]
        return None

    # Try increasing k
    for k in [3, 4, 5, 6, 7, 8]:
        for seed_val in range(20):
            result = solve_constraint(k, seed_val)
            if result is not None:
                return result
    
    # Fallback: simple proper coloring
    coloring = [0] * n
    for i in range(n):
        neighbor_colors = set()
        for nb in adj[i]:
            if nb < i:
                neighbor_colors.add(coloring[nb])
        for c in range(50):
            if c not in neighbor_colors:
                coloring[i] = c
                break
    return coloring

# EVOLVE-BLOCK-END
