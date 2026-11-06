"""Strong edge colouring via Local Search (Simulated Annealing)."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try many times to build a bounded-degree graph
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring(edges, n, max_deg):
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict sets
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # conflict[i] is a list of indices j such that edge i and edge j conflict
        conflict = [[] for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                if x == u or x == v or y == u or y == v:
                    conflict[i].append(j)
                    conflict[j].append(i)
                else:
                    if x in adj[u] or x in adj[v] or y in adj[u] or y in adj[v]:
                        conflict[i].append(j)
                        conflict[j].append(i)
        
        # Get initial upper bound using greedy
        degs = [len(conflict[i]) for i in range(m)]
        order = sorted(range(m), key=lambda i: -degs[i])
        
        color_greedy = [-1] * m
        num_colors_greedy = 0
        for i in order:
            used = set()
            for j in conflict[i]:
                if color_greedy[j] != -1:
                    used.add(color_greedy[j])
            c = 0
            while c in used:
                c += 1
            color_greedy[i] = c
            num_colors_greedy = max(num_colors_greedy, c + 1)
            
        best_color = color_greedy[:]
        best_k = num_colors_greedy
        
        # Local Search / Simulated Annealing to improve
        # Objective: Minimize (Conflicts * 1000 + ColorsUsed)
        # We want 0 conflicts. Then minimize colors.
        
        def get_cost(color):
            conflicts = 0
            used_colors = set()
            for i in range(m):
                c = color[i]
                used_colors.add(c)
                for j in conflict[i]:
                    if color[j] == c:
                        conflicts += 1
            # Double count? No, we iterate all i and check j in conflict[i]. 
            # If we define conflict as symmetric, each pair is counted twice. 
            # Let's just count pairs.
            # Actually, let's just count distinct conflicting pairs to be safe.
            # But for speed, counting occurrences is fine if consistent.
            # Let's count number of edges involved in at least one conflict? No, total conflicts is better.
            # Let's use the standard: sum over i of count of j in conflict[i] with same color.
            # This counts each conflict twice. Divide by 2? Or just use as is.
            return (conflicts, len(used_colors))

        def compare_cost(c1, c2):
            # c1 < c2 means c1 is better
            if c1[0] != c2[0]:
                return c1[0] < c2[0]
            return c1[1] < c2[1]

        current_color = best_color[:]
        current_cost = get_cost(current_color)
        
        # Parameters for Simulated Annealing
        start_temp = 10.0
        end_temp = 0.01
        cooling_rate = 0.99
        max_iterations = 50000
        rng = random.Random(42)
        
        t = start_temp
        for it in range(max_iterations):
            if t <= end_temp:
                break
                
            # Choose a random edge
            i = rng.randrange(m)
            old_c = current_color[i]
            
            # Choose a new color
            # Bias towards existing colors or slightly larger
            # Let's pick a random color from 0 to best_k + 2
            new_c = rng.randrange(best_k + 3)
            
            if new_c == old_c:
                continue
                
            current_color[i] = new_c
            new_cost = get_cost(current_color)
            
            # Calculate delta
            # Cost is (conflicts, colors). We need a scalar value for SA.
            # Let's use conflicts * 1000 + colors
            old_val = current_cost[0] * 1000 + current_cost[1]
            new_val = new_cost[0] * 1000 + new_cost[1]
            delta = new_val - old_val
            
            if delta < 0:
                current_cost = new_cost
                # Update best if valid
                if new_cost[0] == 0:
                    if new_cost[1] < best_k:
                        best_k = new_cost[1]
                        best_color = current_color[:]
            else:
                # Accept with probability exp(-delta / t)
                if rng.random() < pow(2.71828, -delta / t):
                    current_cost = new_cost
                else:
                    # Revert
                    current_color[i] = old_c
                    # current_cost remains old_cost
            
            t *= cooling_rate
            
        # If we found a valid solution with fewer colors, use it
        if current_cost[0] == 0 and current_cost[1] < best_k:
            best_color = current_color[:]
            best_k = current_cost[1]
            
        # If we still have conflicts in best_color (shouldn't happen if greedy was valid), 
        # but greedy is always valid. So best_color is always valid.
        # The SA might find a better one.
        
        # Return the best coloring found
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        return [best_color[edge_to_idx[e]] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        if not edges:
            out.append([])
            continue
        colors = strong_edge_coloring(edges, n, d)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
