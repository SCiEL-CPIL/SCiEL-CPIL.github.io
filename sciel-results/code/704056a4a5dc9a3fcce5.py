"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def build_conflict(edges, adj):
        """Precompute conflict matrix: conflict[i][j] = True if edges i and j conflict."""
        n_edges = len(edges)
        conflict = [bytearray(n_edges) for _ in range(n_edges)]
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Check conflict: share vertex or are joined by an edge
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i][j] = 1
                    conflict[j][i] = 1
                else:
                    # Check if any endpoint of i is adjacent to any endpoint of j
                    if (ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]):
                        conflict[i][j] = 1
                        conflict[j][i] = 1
        return conflict

    def greedy_color(edges, conflict, order):
        """Greedy coloring with a specific edge order."""
        n_edges = len(edges)
        colour = [-1] * n_edges
        max_color = 0
        
        for idx in order:
            # Find taken colors
            taken = set()
            for j in range(n_edges):
                if conflict[idx][j] and colour[j] != -1:
                    taken.add(colour[j])
            
            # Assign smallest available color
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c
            if c > max_color:
                max_color = c
        
        return colour, max_color + 1

    def local_improve(edges, conflict, colour, max_colors, max_iter=2000):
        """Try to reduce max color by reassigning edges."""
        n_edges = len(edges)
        current_max = max_colors - 1
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Find edges with the highest color
            high_edges = [i for i in range(n_edges) if colour[i] == current_max]
            if not high_edges:
                current_max -= 1
                continue
            
            random.shuffle(high_edges)
            
            for e_idx in high_edges:
                # Try to reassign this edge to a lower color
                taken = set()
                for j in range(n_edges):
                    if j != e_idx and conflict[e_idx][j]:
                        taken.add(colour[j])
                
                # Find smallest available color
                new_c = 0
                while new_c in taken:
                    new_c += 1
                
                if new_c < colour[e_idx]:
                    colour[e_idx] = new_c
                    # Check if we can reduce max
                    # Optimization: Only recompute max if we moved the last edge of current_max
                    # But for simplicity and correctness in this loop, we just check if current_max is still used
                    # If not, we decrement.
                    # Actually, since we only move edges *from* current_max to *lower*, 
                    # we just need to check if any edge still has color current_max.
                    still_used = False
                    for k in range(n_edges):
                        if colour[k] == current_max:
                            still_used = True
                            break
                    if not still_used:
                        current_max -= 1
                    improved = True
                    break
                else:
                    # Try swapping: see if we can move another edge from this color
                    # to free up a slot, then reassign e_idx
                    # This is complex, so we stick to simple reassignment for now
                    # but we can try to find a color c < current_max that is free
                    # The logic above already finds the smallest free color.
                    # If the smallest free color is >= current_max, it means all colors < current_max are taken.
                    # So we cannot improve this edge directly.
                    pass

        return colour, current_max + 1

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n_edges = len(edges)
        
        # Precompute conflict matrix
        conflict = build_conflict(edges, adj)
        
        # Generate multiple orders
        orders = []
        
        # 1. Original order (sorted)
        orders.append(list(range(n_edges)))
        
        # 2. Decreasing degree of endpoints
        edge_degrees = [len(adj[u]) + len(adj[v]) for u, v in edges]
        orders.append(sorted(range(n_edges), key=lambda i: -edge_degrees[i]))
        
        # 3. Decreasing constraint (number of conflicts)
        constraint_counts = [sum(conflict[i]) for i in range(n_edges)]
        orders.append(sorted(range(n_edges), key=lambda i: -constraint_counts[i]))
        
        # 4-6. Random permutations
        rng = random.Random(42)
        for _ in range(3):
            perm = list(range(n_edges))
            rng.shuffle(perm)
            orders.append(perm)
            
        # 7. Random but biased towards high constraint
        # We can do this by shuffling but keeping high constraint items earlier? 
        # Simpler: just try a few more randoms.
        for _ in range(5):
            perm = list(range(n_edges))
            rng.shuffle(perm)
            orders.append(perm)

        best_colour = None
        best_num_colors = float('inf')
        
        # Time budget per instance
        instance_deadline = deadline - 10 - (2 - len(out)) * 40 # Rough estimate, more time for earlier instances? 
        # Actually, just check global deadline.
        
        for order in orders:
            if time.time() > deadline - 10:
                break
                
            colour, num_colors = greedy_color(edges, conflict, order)
            
            # Local improvement
            if time.time() < deadline - 10:
                colour, num_colors = local_improve(edges, conflict, colour, num_colors, max_iter=500)
                
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colour = colour[:]
                
        if best_colour is None:
            # Fallback
            colour, _ = greedy_color(edges, conflict, list(range(n_edges)))
            best_colour = colour
        
        result = best_colour
        out.append(result)
        
    return out
# EVOLVE-BLOCK-END
