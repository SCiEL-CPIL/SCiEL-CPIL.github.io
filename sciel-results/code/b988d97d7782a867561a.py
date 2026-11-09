"""Strong edge colouring of bounded-degree graphs via greedy + local search with conflict precomputation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 5  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate a random bounded-degree graph
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
        """
        Precompute the set of edges that conflict with each edge.
        Two edges conflict if they share a vertex or are joined by an edge (distance 2).
        """
        edge_index = {e: i for i, e in enumerate(edges)}
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        
        # For each edge, find conflicting edges
        for i, e1 in enumerate(edges):
            u1, v1 = e1
            # Edges sharing a vertex
            for e2_idx, e2 in enumerate(edges):
                if e2_idx == i:
                    continue
                u2, v2 = e2
                # Share vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
                else:
                    # Distance 2? (joined by an edge)
                    # Check if any endpoint of e1 is adjacent to any endpoint of e2
                    if u2 in adj[u1] or u2 in adj[v1] or v2 in adj[u1] or v2 in adj[v1]:
                        conflicts[i].add(e2_idx)
                        conflicts[e2_idx].add(i)
        return conflicts

    def greedy_coloring(edges, conflicts, order):
        """
        Greedy coloring using precomputed conflicts.
        order: list of edge indices in the order to process.
        """
        n_edges = len(edges)
        color = [-1] * n_edges
        color_classes = [set() for _ in range(n_edges)] # Max possible colors
        
        for e_idx in order:
            # Find taken colors
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

    def local_search(edges, conflicts, color, color_classes, max_iter=10000):
        """
        Local search: try to move edges to lower colors.
        Returns improved color and color_classes.
        """
        n_edges = len(edges)
        current_max_color = max(color) + 1 if color else 0
        rng = random.Random(42)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Process edges in random order
            order = list(range(n_edges))
            rng.shuffle(order)
            
            for e_idx in order:
                c_current = color[e_idx]
                if c_current == 0:
                    continue
                
                # Try to move to a lower color
                for new_c in range(c_current):
                    # Check if e_idx conflicts with any edge in color_classes[new_c]
                    can_move = True
                    # Optimization: if the intersection of conflicts[e_idx] and color_classes[new_c] is empty, we can move
                    # We need to check if any neighbor in conflicts[e_idx] is in color_classes[new_c]
                    for c_idx in conflicts[e_idx]:
                        if color_classes[new_c] and c_idx in color_classes[new_c]:
                            can_move = False
                            break
                    
                    if can_move:
                        # Move e_idx to new_c
                        color_classes[c_current].remove(e_idx)
                        color_classes[new_c].add(e_idx)
                        color[e_idx] = new_c
                        improved = True
                        # Update max color if we removed the last edge of the highest color
                        if c_current == current_max_color - 1 and not color_classes[c_current]:
                            current_max_color -= 1
                        break
            
            if not improved:
                break
        
        return color, color_classes, current_max_color

    def optimize_instance(n, d, seed, time_budget):
        """
        Optimize coloring for a single instance within time budget.
        """
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)
        
        best_color = None
        best_max_color = float('inf')
        
        elapsed_start = time.time()
        
        # Define multiple orderings
        orderings = []
        
        # 1. Natural order (already sorted by input generation, but let's be explicit)
        orderings.append(list(range(n_edges)))
        
        # 2. Reverse natural
        orderings.append(list(reversed(range(n_edges))))
        
        # 3. Degree-based (high degree vertices first)
        # Sort edges by sum of degrees of endpoints
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        
        # 4. Random shuffles
        for s in range(20):
            rng = random.Random(seed + 1000 + s)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        # 5. Greedy with "worst first" (edges with most conflicts first)
        conflict_counts = [len(conflicts[i]) for i in range(n_edges)]
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break
            
            # Greedy
            color, color_classes = greedy_coloring(edges, conflicts, ordering)
            current_max = max(color) + 1 if color else 0
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color = dict(enumerate(color)) # Store as dict for easy access later? Or just list.
                # Let's keep list and map at the end.
                best_color_list = list(color)
            
            # Local Search
            # Adjust max_iter based on time budget and remaining time
            time_per_iter = 0.001 # rough estimate
            max_iter = max(100, int((time_budget - (time.time() - elapsed_start)) * 500))
            
            color, color_classes, current_max = local_search(edges, conflicts, color, color_classes, max_iter=max_iter)
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color_list = list(color)
        
        # If no coloring found (should not happen), fallback to simple greedy
        if best_color_list is None:
            color, _ = greedy_coloring(edges, conflicts, list(range(n_edges)))
            best_color_list = list(color)
            
        return best_color_list, best_max_color

    results = []
    
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Emergency fallback
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color, _ = greedy_coloring(edges, conflicts, list(range(len(edges))))
            results.append(color)
            continue
        
        time_budget = remaining_time / (len(instances) - idx)
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()
    
    return results

# EVOLVE-BLOCK-END
