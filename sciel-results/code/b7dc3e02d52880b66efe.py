import time
import random
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate enough attempts to get a reasonably dense graph
        for _ in range(10 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        """Build the conflict graph H where vertices are edges of G,
        and two vertices are adjacent if the corresponding edges conflict
        (share a vertex or are joined by an edge in G)."""
        m = len(edges)
        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        # Set of edges for quick lookup
        edge_set = set(edges)
        
        h_adj = [set() for _ in range(m)]
        
        for i, (u1, v1) in enumerate(edges):
            # Edges sharing a vertex
            for j in vertex_to_edges[u1]:
                if j != i:
                    h_adj[i].add(j)
                    h_adj[j].add(i)
            for j in vertex_to_edges[v1]:
                if j != i:
                    h_adj[i].add(j)
                    h_adj[j].add(i)
            
            # Edges joined by an edge in G
            # For each endpoint of edge i, check if it's connected to an endpoint of edge j
            # We need to check all pairs of endpoints
            for j, (u2, v2) in enumerate(edges):
                if i == j:
                    continue
                if j in h_adj[i]:
                    continue  # already marked as conflicting
                # Check if (u1,u2), (u1,v2), (v1,u2), or (v1,v2) is an edge in G
                if (min(u1,u2), max(u1,u2)) in edge_set or \
                   (min(u1,v2), max(u1,v2)) in edge_set or \
                   (min(v1,u2), max(v1,u2)) in edge_set or \
                   (min(v1,v2), max(v1,v2)) in edge_set:
                    h_adj[i].add(j)
                    h_adj[j].add(i)
        
        return h_adj

    def dsatur_coloring(h_adj, m, rng_seed=42):
        """DSATUR greedy coloring of the conflict graph H."""
        rng = random.Random(rng_seed)
        color = [-1] * m
        sat = [set() for _ in range(m)]
        uncolored = set(range(m))
        deg_h = [len(h_adj[i]) for i in range(m)]
        
        while uncolored:
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                d = deg_h[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_v = v
                    best_sat = s
                    best_deg = d
                    
            v = best_v
            uncolored.remove(v)
            
            c = 0
            while c in sat[v]:
                c += 1
            color[v] = c
            
            for u in h_adj[v]:
                if u in uncolored:
                    sat[u].add(c)
                    
        return color

    def try_merge_colors(h_adj, color, m, rng, max_attempts=2000):
        """Try to merge color classes to reduce total number of colors."""
        # Build color sets
        num_colors = max(color) + 1
        color_sets = [set() for _ in range(num_colors)]
        for i in range(m):
            color_sets[color[i]].add(i)
        
        active = list(range(num_colors))
        
        # Precompute which colors are active
        attempts = 0
        improvements = 0
        
        while len(active) > 1 and attempts < max_attempts:
            # Try to find a mergeable pair
            found = False
            for idx in range(len(active)):
                if found:
                    break
                c1 = active[idx]
                for idx2 in range(idx+1, len(active)):
                    c2 = active[idx2]
                    # Check if c1 and c2 can be merged
                    mergeable = True
                    for u in color_sets[c1]:
                        for v in color_sets[c2]:
                            if v in h_adj[u]:
                                mergeable = False
                                break
                        if not mergeable:
                            break
                    if mergeable:
                        # Merge c2 into c1
                        for v in color_sets[c2]:
                            color[v] = c1
                            color_sets[c1].add(v)
                        color_sets[c2] = set()
                        active.remove(c2)
                        found = True
                        improvements += 1
                        break
            if not found:
                # Try swapping: move a vertex from one color to another to enable a merge
                # Pick a random vertex and try to recolor it
                if len(active) >= 2:
                    # Try to move a vertex from a larger color class to a smaller one
                    c_large = max(active, key=lambda c: len(color_sets[c]))
                    c_small = min(active, key=lambda c: len(color_sets[c]))
                    if len(color_sets[c_large]) > 1 and len(color_sets[c_small]) < len(color_sets[c_large]):
                        # Pick a random vertex from c_large
                        v = rng.choice(list(color_sets[c_large]))
                        # Check if v can be moved to c_small
                        can_move = True
                        for u in color_sets[c_small]:
                            if u in h_adj[v]:
                                can_move = False
                                break
                        if can_move:
                            color_sets[c_large].discard(v)
                            color_sets[c_small].add(v)
                            color[v] = c_small
                            improvements += 1
                            continue
            attempts += 1
        
        # Compress colors
        unique_colors = sorted(set(color))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        final_col = [color_map[c] for c in color]
        return final_col, len(unique_colors)

    def simulated_annealing(h_adj, m, initial_color, rng, start_temp=10.0, end_temp=0.1, steps=5000):
        """Simulated annealing to reduce number of colors."""
        # Work with a copy
        color = list(initial_color)
        num_colors = max(color) + 1
        color_sets = [set() for _ in range(num_colors)]
        for i in range(m):
            color_sets[color[i]].add(i)
        
        best_color = list(color)
        best_num = num_colors
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / steps)
        
        for step in range(steps):
            temp *= cooling
            
            # Pick a random vertex
            v = rng.randrange(m)
            old_color = color[v]
            
            # Try to assign a new color
            # Options: existing colors or a new color
            available_colors = list(range(num_colors))
            if rng.random() < 0.1:
                available_colors.append(num_colors)  # try a new color
            
            new_color = rng.choice(available_colors)
            
            if new_color == old_color:
                continue
            
            # Check if v can be assigned new_color
            # v must not conflict with any vertex in color class new_color
            valid = True
            if new_color < num_colors:
                for u in color_sets[new_color]:
                    if u in h_adj[v]:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Compute delta: change in number of colors
            # If we move v from old_color to new_color:
            # - If old_color becomes empty, we save 1 color
            # - If new_color was not used before, we add 1 color
            old_size = len(color_sets[old_color])
            new_size = len(color_sets[new_color]) if new_color < num_colors else 0
            
            delta = 0
            if old_size == 1:
                delta -= 1  # old color class becomes empty
            if new_color >= num_colors or new_size == 0:
                delta += 1  # new color class created
            
            # Accept if delta < 0 or with probability exp(-delta/temp)
            if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temp, 0.01)):
                # Make the move
                color_sets[old_color].discard(v)
                if new_color >= num_colors:
                    color_sets.append(set())
                color_sets[new_color].add(v)
                color[v] = new_color
                
                # Update num_colors
                current_num = max(color) + 1
                if current_num < best_num:
                    best_num = current_num
                    best_color = list(color)
        
        # Compress
        unique_colors = sorted(set(best_color))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        final_col = [color_map[c] for c in best_color]
        return final_col, len(unique_colors)

    results = []
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            results.append([])
            continue
        
        h_adj = build_conflict_graph(edges, n)
        
        # Get initial coloring with DSATUR
        col = dsatur_coloring(h_adj, m, rng_seed=42)
        num_colors = max(col) + 1
        
        # Try to improve with color merging
        rng = random.Random(12345)
        col, num_colors = try_merge_colors(h_adj, col, m, rng, max_attempts=3000)
        
        # Try simulated annealing for further improvement
        if time.time() < deadline - 10.0:
            sa_col, sa_num = simulated_annealing(h_adj, m, col, rng, start_temp=5.0, end_temp=0.05, steps=3000)
            if sa_num < num_colors:
                col = sa_col
                num_colors = sa_num
        
        # Try a few more restarts if time permits
        restarts = 0
        while time.time() < deadline - 15.0 and restarts < 3:
            col2 = dsatur_coloring(h_adj, m, rng_seed=100 + restarts)
            col2, num2 = try_merge_colors(h_adj, col2, m, rng, max_attempts=2000)
            if num2 < num_colors:
                col = col2
                num_colors = num2
            restarts += 1
        
        results.append(col)
    
    return results
