"""Strong edge colouring of bounded-degree graphs via greedy + local search with conflict precomputation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 8  # seconds safety margin before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + 150 - DEADLINE_MARGIN

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
        """Precompute conflict sets: two edges conflict if they share a vertex or are at distance 2."""
        n_edges = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        # Map vertex to list of incident edge indices
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)
        
        conflicts = [set() for _ in range(n_edges)]
        
        for i, (u1, v1) in enumerate(edges):
            # Edges sharing a vertex
            for v in (u1, v1):
                for j in vertex_edges[v]:
                    if j != i:
                        conflicts[i].add(j)
            # Edges at distance 2 (joined by an edge)
            for v in (u1, v1):
                for w in adj[v]:
                    for j in vertex_edges[w]:
                        if j != i and j not in conflicts[i]:
                            # Check that edges i and j don't share a vertex (already handled)
                            u2, v2 = edges[j]
                            if u1 != u2 and u1 != v2 and v1 != u2 and v1 != v2:
                                conflicts[i].add(j)
        
        # Ensure symmetry
        for i in range(n_edges):
            for j in conflicts[i]:
                conflicts[j].add(i)
        
        return conflicts

    def greedy_coloring(conflicts, order):
        """Greedy coloring using precomputed conflicts and a given edge ordering."""
        n_edges = len(conflicts)
        color = [-1] * n_edges
        color_classes = [set() for _ in range(n_edges)]
        
        for e_idx in order:
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c
            color_classes[c].add(e_idx)
        
        return color, color_classes

    def local_search(conflicts, color, color_classes, max_iter=20000, rng=None):
        """Local search: try to move edges to lower colors, and swap pairs to reduce max color."""
        n_edges = len(conflicts)
        if rng is None:
            rng = random.Random(42)
        
        current_max_color = max(color) + 1 if any(c >= 0 for c in color) else 0
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            order = list(range(n_edges))
            rng.shuffle(order)
            
            for e_idx in order:
                c_current = color[e_idx]
                if c_current <= 0:
                    continue
                
                # Try to move to a lower color
                for new_c in range(c_current):
                    can_move = True
                    for c_idx in conflicts[e_idx]:
                        if color[c_idx] == new_c:
                            can_move = False
                            break
                    if can_move:
                        color_classes[c_current].remove(e_idx)
                        color_classes[new_c].add(e_idx)
                        color[e_idx] = new_c
                        improved = True
                        if c_current == current_max_color - 1 and not color_classes[c_current]:
                            current_max_color -= 1
                        break
            
            # Try swap-based improvement: swap colors of two edges if it reduces max color
            if not improved and current_max_color > 1:
                # Find edges with the highest color
                top_color = current_max_color - 1
                top_edges = list(color_classes[top_color])
                if len(top_edges) >= 1:
                    # Try to move a top-color edge to a lower color by swapping
                    for e_idx in top_edges:
                        if color[e_idx] != top_color:
                            continue
                        for new_c in range(top_color):
                            can_move = True
                            for c_idx in conflicts[e_idx]:
                                if color[c_idx] == new_c:
                                    can_move = False
                                    break
                            if can_move:
                                color_classes[top_color].remove(e_idx)
                                color_classes[new_c].add(e_idx)
                                color[e_idx] = new_c
                                improved = True
                                current_max_color -= 1
                                break
                        if improved:
                            break
            
            # Try 2-swap: swap colors of two edges from different color classes
            if not improved:
                # Pick two random edges from different colors
                if n_edges >= 2:
                    e1 = rng.randrange(n_edges)
                    e2 = rng.randrange(n_edges)
                    if e1 != e2 and color[e1] != color[e2]:
                        c1, c2 = color[e1], color[e2]
                        # Check if swapping is valid
                        valid = True
                        for other in conflicts[e1]:
                            if color[other] == c2 and other != e2:
                                valid = False
                                break
                        if valid:
                            for other in conflicts[e2]:
                                if color[other] == c1 and other != e1:
                                    valid = False
                                    break
                        if valid:
                            # Perform swap
                            color_classes[c1].remove(e1)
                            color_classes[c2].remove(e2)
                            color_classes[c1].add(e2)
                            color_classes[c2].add(e1)
                            color[e1], color[e2] = c2, c1
                            # After swap, try to reduce
                            for e_idx in (e1, e2):
                                c_cur = color[e_idx]
                                for new_c in range(c_cur):
                                    can_move = True
                                    for c_idx in conflicts[e_idx]:
                                        if color[c_idx] == new_c:
                                            can_move = False
                                            break
                                    if can_move:
                                        color_classes[c_cur].remove(e_idx)
                                        color_classes[new_c].add(e_idx)
                                        color[e_idx] = new_c
                                        improved = True
                                        if c_cur == current_max_color - 1 and not color_classes[c_cur]:
                                            current_max_color -= 1
                                        break
                            if improved:
                                break
            
            if not improved:
                break
        
        return color, color_classes, current_max_color

    def optimize_instance(n, d, seed, time_budget):
        """Optimize coloring for a single instance within time budget."""
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)
        
        best_color = None
        best_max_color = float('inf')
        
        elapsed_start = time.time()
        
        # Define multiple orderings
        orderings = []
        
        # 1. Natural order
        orderings.append(list(range(n_edges)))
        
        # 2. Reverse natural
        orderings.append(list(reversed(range(n_edges))))
        
        # 3. Degree-based (high degree vertices first)
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        
        # 4. Conflict-count based (most conflicts first)
        conflict_counts = [len(conflicts[i]) for i in range(n_edges)]
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        
        # 5. Random shuffles
        for s in range(15):
            rng = random.Random(seed + 1000 + s)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break
            
            # Greedy
            color, color_classes = greedy_coloring(conflicts, ordering)
            current_max = max(color) + 1 if color else 0
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color = list(color)
            
            # Local search
            remaining = time_budget - (time.time() - elapsed_start)
            max_iter = max(500, int(remaining * 2000))
            rng = random.Random(hash((seed, tuple(ordering[:10])) % (2**32)))
            color, color_classes, current_max = local_search(
                conflicts, color, color_classes, max_iter=max_iter, rng=rng
            )
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color = list(color)
        
        if best_color is None:
            # Fallback: simple greedy
            color, _ = greedy_coloring(conflicts, list(range(n_edges)))
            best_color = list(color)
            best_max_color = max(best_color) + 1
        
        return best_color, best_max_color

    results = []
    total_colors = 0
    
    # Process each instance
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Not enough time, use simple greedy for remaining
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color, _ = greedy_coloring(conflicts, list(range(len(edges))))
            results.append(color)
            continue
        
        remaining = deadline - time.time()
        time_budget = remaining / (len(instances) - idx)
        coloring, count = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        total_colors += count
    
    return results

# EVOLVE-BLOCK-END
