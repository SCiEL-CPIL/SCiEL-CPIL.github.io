"""Strong edge colourings via multi-restart greedy + local merging."""
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def build_adj(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_conflicts(e, adj, all_edges_set):
        """Return set of edges that conflict with e (cannot share a color)."""
        u, v = e
        # All edges sharing u or v, plus edges with endpoints in N(u) ∪ N(v)
        near_vertices = {u, v}
        near_vertices.update(adj[u])
        near_vertices.update(adj[v])
        conflicts = set()
        for x in near_vertices:
            for y in adj[x]:
                edge = (min(x, y), max(x, y))
                if edge != e:
                    conflicts.add(edge)
        return conflicts

    def greedy_color(edges, adj, order=None):
        """Color edges in given order (or sorted order) using greedy."""
        if order is None:
            order = list(range(len(edges)))
        color = {}
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        for idx in order:
            e = edges[idx]
            u, v = e
            near = {u, v}
            near.update(adj[u])
            near.update(adj[v])
            taken = set()
            for f, c in color.items():
                fu, fv = f
                if fu in near or fv in near:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            color[e] = c
        return color

    def count_colors(color):
        if not color:
            return 0
        return max(color.values()) + 1

    def try_reduce_colors(edges, adj, color):
        """Try to reduce number of colors by reassigning edges."""
        n_colors = count_colors(color)
        if n_colors == 0:
            return color, n_colors
        
        # Try to eliminate each color from highest to lowest
        for target_color in range(n_colors - 1, -1, -1):
            # Get all edges with this color
            edges_with_color = [e for e, c in color.items() if c == target_color]
            if not edges_with_color:
                continue
            
            # Try to reassign each edge to a lower color
            new_color = dict(color)
            success = True
            for e in edges_with_color:
                u, v = e
                near = {u, v}
                near.update(adj[u])
                near.update(adj[v])
                # Find a color < target_color that doesn't conflict
                assigned = False
                for c in range(target_color):
                    conflict = False
                    for f, fc in new_color.items():
                        if fc != c:
                            continue
                        fu, fv = f
                        if fu in near or fv in near:
                            conflict = True
                            break
                    if not conflict:
                        new_color[e] = c
                        assigned = True
                        break
                if not assigned:
                    success = False
                    break
            
            if success:
                return new_color, n_colors - 1
        
        return color, n_colors

    def local_search(edges, adj, color, max_iters=50):
        """Local search: try to reduce colors by swapping edge assignments."""
        n_colors = count_colors(color)
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for target_color in range(n_colors - 1, -1, -1):
                edges_with_color = [e for e, c in color.items() if c == target_color]
                if not edges_with_color:
                    continue
                # Try reassigning one edge at a time
                for e in edges_with_color:
                    u, v = e
                    near = {u, v}
                    near.update(adj[u])
                    near.update(adj[v])
                    for c in range(target_color):
                        conflict = False
                        for f, fc in color.items():
                            if fc != c:
                                continue
                            fu, fv = f
                            if fu in near or fv in near:
                                conflict = True
                                break
                        if not conflict:
                            color[e] = c
                            n_colors = count_colors(color)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return color, n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        
        best_color = None
        best_count = float('inf')
        
        # Try multiple random orderings
        rng = random.Random(seed + 999)
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            order = list(range(len(edges)))
            rng.shuffle(order)
            color = greedy_color(edges, adj, order)
            color, n_colors = try_reduce_colors(edges, adj, color)
            if n_colors < best_count:
                best_count = n_colors
                best_color = dict(color)
        
        # Also try sorted order
        color = greedy_color(edges, adj)
        color, n_colors = try_reduce_colors(edges, adj, color)
        if n_colors < best_count:
            best_count = n_colors
            best_color = dict(color)
        
        # Local search on best
        if best_color is not None:
            best_color, best_count = local_search(edges, adj, best_color)
        
        results.append([best_color[e] for e in edges])
    
    return results
