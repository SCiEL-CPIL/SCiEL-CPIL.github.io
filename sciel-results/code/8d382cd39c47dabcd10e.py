"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        m = len(edges)
        if m == 0:
            return []
        
        edge_vertices = [(u, v) for u, v in edges]
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edge_vertices[i]
            for j in range(i + 1, m):
                u2, v2 = edge_vertices[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict

    def greedy_max_induced_matching(conflict, remaining_set):
        """
        Find a large induced matching (independent set in conflict graph)
        using greedy min-degree selection.
        """
        if not remaining_set:
            return []
        
        # Build local conflict for remaining edges
        rem_list = sorted(remaining_set)
        rem_set = set(rem_list)
        rem_to_local = {i: k for k, i in enumerate(rem_list)}
        n_rem = len(rem_list)
        
        if n_rem == 0:
            return []
        
        local_conflict = [set() for _ in range(n_rem)]
        
        for ki, i in enumerate(rem_list):
            for j in conflict[i]:
                if j in rem_set:
                    kj = rem_to_local[j]
                    local_conflict[ki].add(kj)
        
        # Greedy: repeatedly pick vertex with min degree in active subgraph
        selected = []
        active = set(range(n_rem))
        
        while active:
            min_deg = float('inf')
            chosen = None
            for v in active:
                deg = len(local_conflict[v] & active)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            
            if chosen is None:
                break
                
            selected.append(chosen)
            active.discard(chosen)
            for nb in local_conflict[chosen]:
                active.discard(nb)
        
        # Convert back to global edge indices
        return [rem_list[i] for i in selected]

    def initial_coloring(edges, adj):
        """Initial coloring using greedy max induced matching per color class."""
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict_graph(edges, adj)
        color = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        
        while remaining:
            selected = greedy_max_induced_matching(conflict, remaining)
            for idx in selected:
                color[idx] = color_idx
                remaining.discard(idx)
            color_idx += 1
        
        return color

    def try_reassign(color, conflict, target_color):
        """
        Try to remove target_color by reassigning its edges to other colors.
        Returns True if successful, False otherwise.
        """
        # Find all edges with target_color
        target_edges = [i for i, c in enumerate(color) if c == target_color]
        if not target_edges:
            return True
        
        # Try to reassign each target edge to an existing color
        # Sort by number of conflicts (easier to reassign first)
        target_edges.sort(key=lambda i: len(conflict[i]))
        
        new_color = color[:]
        
        for idx in target_edges:
            # Find a color that doesn't conflict with idx
            found = False
            # Try lower colors first, then higher
            max_c = max(new_color) if new_color else 0
            for c in sorted(range(max_c + 1), key=lambda x: abs(x - target_color)):
                if c == target_color:
                    continue
                # Check if idx can take color c
                can_take = True
                for nb in conflict[idx]:
                    if new_color[nb] == c:
                        can_take = False
                        break
                if can_take:
                    new_color[idx] = c
                    found = True
                    break
            
            if not found:
                # Cannot reassign this edge
                return False
        
        color[:] = new_color
        return True

    def swap_local_search(color, conflict, max_iters=1000):
        """
        Local search with swaps to remove colors.
        Tries to remove the highest color by swapping edges between colors.
        """
        m = len(color)
        if m == 0:
            return color
        
        best_color = color[:]
        best_num_colors = max(color) + 1 if color else 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            current_num_colors = max(best_color) + 1 if best_color else 0
            if current_num_colors <= 1:
                break
            
            # Try to remove the highest color
            target_color = current_num_colors - 1
            target_edges = [i for i, c in enumerate(best_color) if c == target_color]
            
            if not target_edges:
                continue
            
            # Try direct reassignment
            if try_reassign(best_color, conflict, target_color):
                best_num_colors = max(best_color) + 1 if best_color else 0
                continue
            
            # Try swap-based approach
            # For each target edge, try swapping it with an edge in another color
            swapped = False
            for idx in target_edges:
                # Find edges in other colors that could swap with idx
                for other_idx in range(m):
                    if other_idx == idx:
                        continue
                    other_color = best_color[other_idx]
                    if other_color == target_color:
                        continue
                    
                    # Check if swapping idx and other_idx is valid
                    # After swap: idx gets other_color, other_idx gets target_color
                    # But we want to remove target_color, so other_idx should not get target_color
                    # Instead, let's try: move idx to other_color, and move other_idx to some other color
                    # This is complex; let's use a simpler approach
                    
                    # Try to move idx to other_color
                    can_move = True
                    for nb in conflict[idx]:
                        if best_color[nb] == other_color:
                            can_move = False
                            break
                    
                    if can_move:
                        # Move idx to other_color
                        new_color = best_color[:]
                        new_color[idx] = other_color
                        swapped = True
                        best_color = new_color
                        break
                
                if swapped:
                    break
            
            if not swapped:
                # Try a different heuristic: randomly pick a target edge and try to reassign
                random.shuffle(target_edges)
                for idx in target_edges:
                    max_c = max(best_color) if best_color else 0
                    for c in sorted(range(max_c + 1)):
                        if c == target_color:
                            continue
                        can_take = True
                        for nb in conflict[idx]:
                            if best_color[nb] == c:
                                can_take = False
                                break
                        if can_take:
                            best_color[idx] = c
                            swapped = True
                            break
                    if swapped:
                        break
            
            if swapped:
                # Check if we can now remove target_color
                if try_reassign(best_color, conflict, target_color):
                    best_num_colors = max(best_color) + 1 if best_color else 0
        
        return best_color

    def polish(color, conflict):
        """
        Final polish pass: try to reduce colors further with more aggressive strategies.
        """
        m = len(color)
        if m == 0:
            return color
        
        best_color = color[:]
        
        # Multiple passes of trying to remove colors
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            current_num_colors = max(best_color) + 1 if best_color else 0
            
            # Try to remove each color from highest to lowest
            for target_color in range(current_num_colors - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                target_edges = [i for i, c in enumerate(best_color) if c == target_color]
                if not target_edges:
                    continue
                
                # Try reassignment with lower-color preference
                success = True
                for idx in sorted(target_edges, key=lambda i: len(conflict[i])):
                    found = False
                    # Prefer lower colors
                    for c in range(target_color):
                        can_take = True
                        for nb in conflict[idx]:
                            if best_color[nb] == c:
                                can_take = False
                                break
                        if can_take:
                            best_color[idx] = c
                            found = True
                            break
                    
                    if not found:
                        # Try any other color
                        for c in range(current_num_colors):
                            if c == target_color:
                                continue
                            can_take = True
                            for nb in conflict[idx]:
                                if best_color[nb] == c:
                                    can_take = False
                                    break
                            if can_take:
                                best_color[idx] = c
                                found = True
                                break
                    
                    if not found:
                        success = False
                        break
                
                if success:
                    # Color target_color is now unused
                    pass
        
        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 30:
            # Not enough time, return best effort
            break
        
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            out.append([])
            continue
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict = build_conflict_graph(edges, adj)
        
        # Initial coloring
        color = initial_coloring(edges, adj)
        
        # Local search with swaps
        color = swap_local_search(color, conflict, max_iters=500)
        
        # Polish
        color = polish(color, conflict)
        
        out.append(color)
    
    # If we ran out of time, fill remaining with empty lists
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
