"""Strong edge colouring of bounded-degree graphs via DSATUR + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 12  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges), degree

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def precompute_conflicts(edges, adj):
        """Precompute conflict sets: edges that share a vertex or are at distance 2."""
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        edge_idx = {e: i for i, e in enumerate(edges)}
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)
        
        for i, (u1, v1) in enumerate(edges):
            # Edges sharing a vertex
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            
            # Distance-2 edges
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)
        
        return conflicts

    def dsatur_coloring(edges, conflicts):
        """DSATUR heuristic for initial coloring."""
        n_edges = len(edges)
        color = [-1] * n_edges
        # Track which colors are used by neighbors for each edge
        neighbor_colors = [set() for _ in range(n_edges)]
        color_count = 0
        
        for _ in range(n_edges):
            if time.time() > deadline:
                break
            # Find uncolored edge with max saturation degree, then max degree
            best_idx = -1
            best_sat = -1
            best_deg = -1
            for i in range(n_edges):
                if color[i] != -1:
                    continue
                sat = len(neighbor_colors[i])
                deg = len(conflicts[i])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_idx = i
            
            if best_idx == -1:
                break
            
            # Find smallest available color
            c = 0
            while c in neighbor_colors[best_idx]:
                c += 1
            color[best_idx] = c
            color_count = max(color_count, c + 1)
            
            # Update neighbor color sets
            for j in conflicts[best_idx]:
                if color[j] == -1:
                    neighbor_colors[j].add(c)
        
        return color, color_count

    def greedy_coloring(edges, conflicts, order):
        """Greedy coloring in given order."""
        n_edges = len(edges)
        color = [-1] * n_edges
        color_count = 0
        
        for e_idx in order:
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c
            color_count = max(color_count, c + 1)
        
        return color, color_count

    def local_search(edges, conflicts, color, time_budget):
        """Local search to reduce color count by moving edges to lower colors."""
        n_edges = len(edges)
        color = list(color)
        color_count = max(color) + 1 if any(c != -1 for c in color) else 0
        rng = random.Random(42)
        
        # Build color class sets for fast lookup
        color_classes = defaultdict(set)
        for i, c in enumerate(color):
            if c != -1:
                color_classes[c].add(i)
        
        start = time.time()
        for iteration in range(200):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break
            
            improved = False
            order = list(range(n_edges))
            rng.shuffle(order)
            
            for e_idx in order:
                c_current = color[e_idx]
                if c_current == 0:
                    continue
                
                # Try to move to a lower color
                for new_c in range(c_current):
                    can_move = True
                    for c_idx in conflicts[e_idx]:
                        if color[c_idx] == new_c:
                            can_move = False
                            break
                    if can_move:
                        color_classes[c_current].discard(e_idx)
                        color_classes[new_c].add(e_idx)
                        color[e_idx] = new_c
                        improved = True
                        if c_current == color_count - 1 and not color_classes[c_current]:
                            color_count -= 1
                        break
            
            if not improved:
                break
        
        return color, color_count

    def optimize_instance(n, d, seed, time_budget):
        """Optimize coloring for a single instance."""
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)
        
        best_color = None
        best_count = float('inf')
        
        elapsed_start = time.time()
        
        # DSATUR initial coloring
        dsatur_color, dsatur_count = dsatur_coloring(edges, conflicts)
        if dsatur_count < best_count:
            best_color = dsatur_color
            best_count = dsatur_count
        
        # Local search on DSATUR
        if time.time() - elapsed_start < time_budget * 0.4:
            ls_color, ls_count = local_search(edges, conflicts, dsatur_color, time_budget * 0.4)
            if ls_count < best_count:
                best_color = ls_color
                best_count = ls_count
        
        # Try multiple greedy orderings
        remaining = time_budget - (time.time() - elapsed_start)
        n_restarts = max(3, int(remaining * 3))
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break
            
            rng = random.Random(seed + restart * 7919)
            order = list(range(n_edges))
            rng.shuffle(order)
            
            g_color, g_count = greedy_coloring(edges, conflicts, order)
            if g_count < best_count:
                best_color = g_color
                best_count = g_count
            
            # Local search on greedy
            if time.time() - elapsed_start < time_budget * 0.9:
                ls_color, ls_count = local_search(edges, conflicts, g_color, time_budget * 0.15)
                if ls_count < best_count:
                    best_color = ls_color
                    best_count = ls_count
        
        # Final local search on best
        if best_color is not None and time.time() - elapsed_start < time_budget * 0.95:
            final_color, final_count = local_search(edges, conflicts, best_color, time_budget * 0.05)
            if final_count < best_count:
                best_color = final_color
                best_count = final_count
        
        if best_color is None:
            best_color, _ = greedy_coloring(edges, conflicts, list(range(n_edges)))
        
        return [best_color[e] for e in range(n_edges)], best_count

    results = []
    total_colors = 0
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color, _ = greedy_coloring(edges, conflicts, list(range(len(edges))))
            results.append([color[e] for e in range(len(edges))])
            continue
        
        time_budget = remaining_time / (len(instances) - idx)
        coloring, count = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        total_colors += count
        remaining_time = deadline - time.time()
    
    return results

# EVOLVE-BLOCK-END
