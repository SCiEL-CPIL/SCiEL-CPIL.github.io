import time
import random
from collections import defaultdict

_DEADLINE = 150.0  # wall-clock seconds allowed by evaluator

def construct():
    t0 = time.time()
    
    def remaining_time():
        return _DEADLINE - 1.0 - (time.time() - t0)  # 1s safety margin
    
    def bounded_degree_graph(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def solve_instance(edges, adj, max_time):
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # Precompute conflict sets: for each edge i, which edges j conflict with it
        # Two edges conflict if they share a vertex or are joined by an edge (i.e., the 4 endpoints induce a K2,2 or path of length 2 between them)
        # Actually: edges e1=(a,b) and e2=(c,d) conflict if:
        #   - they share a vertex (a==c, a==d, b==c, b==d), OR
        #   - there's an edge connecting {a,b} to {c,d}, i.e., (a,c) in adj, (a,d) in adj, (b,c) in adj, or (b,d) in adj
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build conflict sets as bitmasks for speed
        # n_edges up to ~240, use Python ints as bitmasks
        conflicts = [0] * n_edges  # bitmask of conflicting edges
        
        for i in range(n_edges):
            a, b = edges[i]
            for j in range(i + 1, n_edges):
                c, d = edges[j]
                # Check if edges conflict
                conflict = False
                # Share vertex?
                if a == c or a == d or b == c or b == d:
                    conflict = True
                else:
                    # Joined by an edge?
                    if c in adj.get(a, set()) or d in adj.get(a, set()) or c in adj.get(b, set()) or d in adj.get(b, set()):
                        conflict = True
                if conflict:
                    conflicts[i] |= (1 << j)
                    conflicts[j] |= (1 << i)
        
        # Greedy coloring using maximum induced matching per color
        # Strategy: repeatedly find a large induced matching (set of pairwise non-conflicting edges),
        # assign them the next color, remove them, repeat.
        
        # For finding a good induced matching in the remaining graph, use a greedy approach:
        # Order remaining edges by "degree" in conflict graph (fewer conflicts = more flexible),
        # then greedily add edges that don't conflict with already-selected ones.
        
        uncolored = (1 << n_edges) - 1
        coloring = [-1] * n_edges
        colors_used = 0
        color_classes = []  # list of lists of edge indices per color
        
        # Precompute: for each edge, its conflict count (degree in conflict graph)
        conflict_deg = [bin(conflicts[i]).count('1') for i in range(n_edges)]
        
        while uncolored and remaining_time() > 0.5:
            # Find a good induced matching among uncolored edges
            # Use greedy: sort uncolored edges by (conflict_deg ascending, index)
            # and pick edges that don't conflict with already-picked in this round
            
            uncolored_list = [i for i in range(n_edges) if (uncolored >> i) & 1]
            if not uncolored_list:
                break
            
            # Sort by conflict degree (fewer conflicts first) for better packing
            uncolored_list.sort(key=lambda i: (conflict_deg[i], i))
            
            selected = []
            selected_mask = 0
            for i in uncolored_list:
                if (selected_mask & conflicts[i]) == 0:
                    selected.append(i)
                    selected_mask |= (1 << i)
            
            # Assign this color class
            for i in selected:
                coloring[i] = colors_used
            color_classes.append(selected)
            colors_used += 1
            uncolored &= ~selected_mask
        
        # Local search: try to reduce colors_used
        # Move: move an edge from color c1 to color c2 if c2 < c1 and no conflict
        # Swap: swap colors of two edges if it reduces total colors
        
        best_colors = colors_used
        best_coloring = coloring[:]
        
        # Helper: check if edge e can be assigned to color c
        def can_assign(e, c):
            """Check if edge e can be moved to color c (no conflicts with edges in color c)."""
            for f in color_classes[c]:
                if (conflicts[e] >> f) & 1:
                    return False
            return True
        
        # Helper: check if two edges e1, e2 can swap colors
        def can_swap(e1, e2):
            """Check if e1 and e2 can swap colors."""
            c1, c2 = coloring[e1], coloring[e2]
            if c1 == c2:
                return False
            # After swap: e1 goes to c2, e2 goes to c1
            # e1 must not conflict with any edge in c2 (except e2 if e2 is in c2... but e2 is moving out)
            # Actually e2 is currently in c2, so we need e1 to not conflict with edges in c2 excluding e2
            # And e2 must not conflict with edges in c1 excluding e1
            for f in color_classes[c2]:
                if f != e2 and (conflicts[e1] >> f) & 1:
                    return False
            for f in color_classes[c1]:
                if f != e1 and (conflicts[e2] >> f) & 1:
                    return False
            return True
        
        # Iterative local search
        improved = True
        while improved and remaining_time() > 0.5:
            improved = False
            # Try moves: move edge from higher color to lower color
            for c in range(colors_used - 1, 0, -1):
                if remaining_time() < 0.5:
                    break
                for e in color_classes[c]:
                    for c2 in range(c):
                        if can_assign(e, c2):
                            # Perform move
                            color_classes[c].remove(e)
                            color_classes[c2].append(e)
                            coloring[e] = c2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Try swaps
            if not improved:
                for i in range(len(color_classes)):
                    if remaining_time() < 0.5:
                        break
                    for j in range(i + 1, len(color_classes)):
                        if remaining_time() < 0.5:
                            break
                        for e in color_classes[i]:
                            for f in color_classes[j]:
                                if can_swap(e, f):
                                    # Perform swap
                                    color_classes[i].remove(e)
                                    color_classes[j].remove(f)
                                    color_classes[i].append(f)
                                    color_classes[j].append(e)
                                    coloring[e], coloring[f] = coloring[f], coloring[e]
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Compact colors: renumber so colors are 0..k-1 with no gaps
        if improved:
            # Rebuild color_classes compactly
            new_coloring = [-1] * n_edges
            new_classes = []
            color_map = {}
            for c in range(colors_used):
                if color_classes[c]:
                    new_c = len(new_classes)
                    color_map[c] = new_c
                    new_classes.append(color_classes[c][:])
                    for e in color_classes[c]:
                        new_coloring[e] = new_c
            # Update
            coloring = new_coloring
            color_classes = new_classes
            colors_used = len(color_classes)
        
        return coloring
    
    results = []
    for (n, d, seed) in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if remaining_time() < 5.0:
            # Not enough time; use fast greedy for remaining instances
            edges = bounded_degree_graph(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            # Fast greedy
            color = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near.update(adj.get(x, set()))
                taken = set()
                for f, c in color.items():
                    if set(f) & near:
                        taken.add(c)
                color[e] = next(c for c in range(len(edges)) if c not in taken)
            results.append([color[e] for e in edges])
            continue
        
        edges = bounded_degree_graph(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        coloring = solve_instance(edges, adj, remaining_time())
        results.append(coloring)
    
    return results
