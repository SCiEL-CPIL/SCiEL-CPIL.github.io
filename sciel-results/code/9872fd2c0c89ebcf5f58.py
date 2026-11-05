import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
                        
            conflicts[i].discard(i)
            
        return conflicts

    def greedy_color(edges, conflicts, order):
        """Color edges in the given order using smallest available color."""
        m = len(edges)
        color_of_edge = [-1] * m
        used_colors = 0
        
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of_edge[j] != -1:
                    forbidden.add(color_of_edge[j])
            
            c = 0
            while c in forbidden:
                c += 1
            
            color_of_edge[i] = c
            if c + 1 > used_colors:
                used_colors = c + 1
        
        return color_of_edge, used_colors

    def local_search_improve(colors, conflicts, max_steps=5000):
        """Try to improve coloring by recoloring individual edges to lower colors."""
        m = len(colors)
        colors = colors[:]
        conflict_lists = [list(c) for c in conflicts]
        
        rng = random.Random(42)
        improved = True
        steps = 0
        
        while improved and steps < max_steps:
            improved = False
            order = list(range(m))
            rng.shuffle(order)
            
            for i in order:
                current_color = colors[i]
                if current_color == 0:
                    continue
                
                forbidden = set()
                for j in conflict_lists[i]:
                    forbidden.add(colors[j])
                
                new_color = None
                for c in range(current_color):
                    if c not in forbidden:
                        new_color = c
                        break
                
                if new_color is not None:
                    colors[i] = new_color
                    improved = True
                    steps += 1
                    if steps > max_steps:
                        break
            
        return colors, max(colors) + 1

    def simulated_annealing(edges, conflicts, initial_colors, time_budget=30.0):
        """Simulated annealing to minimize max color."""
        m = len(edges)
        colors = initial_colors[:]
        conflict_lists = [list(c) for c in conflicts]
        
        rng = random.Random(777)
        start = time.time()
        current_energy = max(colors) + 1
        best_energy = current_energy
        best_colors = colors[:]
        
        # Temperature schedule
        t0 = 1.0
        t_end = 0.01
        # We'll run until time budget exhausted
        
        while time.time() < start + time_budget:
            # Perturbation: pick a random edge and change its color
            i = rng.randrange(m)
            old_color = colors[i]
            
            # Try a new color: either a random color in [0, current_max+1] or a local move
            max_c = max(colors)
            new_color = rng.randint(0, max_c + 1)
            
            if new_color == old_color:
                continue
            
            # Check if new_color is valid for edge i
            valid = True
            for j in conflict_lists[i]:
                if colors[j] == new_color:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute energy change
            colors[i] = new_color
            new_energy = max(colors) + 1
            
            delta = new_energy - current_energy
            
            if delta <= 0:
                # Accept
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta/T)
                # T decays over time
                elapsed = time.time() - start
                progress = min(1.0, elapsed / time_budget)
                T = t0 * (t_end / t0) ** progress
                
                if rng.random() < (1.0 / (1.0 + delta)) * (1.0 - progress * 0.5):
                    current_energy = new_energy
                else:
                    # Reject
                    colors[i] = old_color
            
            # Occasionally do a local improvement pass
            if rng.random() < 0.05:
                colors, new_energy = local_search_improve(colors, conflicts, max_steps=100)
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = colors[:]
        
        return best_colors, best_energy

    def color_graph(edges, adj, time_budget=40.0):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflict_graph(edges, adj)
        
        # Initial greedy coloring
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        colors, score = greedy_color(edges, conflicts, order)
        
        # Local search improvement
        colors, score = local_search_improve(colors, conflicts, max_steps=2000)
        
        # Simulated annealing
        colors, score = simulated_annealing(edges, conflicts, colors, time_budget=time_budget)
        
        return colors, score

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Pre-generate graphs
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))
    
    best_results = []
    best_scores = [float('inf')] * len(graphs)
    
    # Allocate time to each graph
    for idx, (edges, adj) in enumerate(graphs):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Give each graph a fair share of remaining time
        time_per_graph = min(30.0, remaining / (len(graphs) - idx))
        
        colors, score = color_graph(edges, adj, time_budget=time_per_graph)
        
        best_scores[idx] = score
        best_results.append(colors)
    
    # If any graph didn't get a result, fill with a quick greedy
    for i in range(len(graphs)):
        if i >= len(best_results) or not best_results[i]:
            colors, _ = color_graph(graphs[i][0], graphs[i][1], time_budget=5.0)
            if i < len(best_results):
                best_results[i] = colors
            else:
                best_results.append(colors)
    
    return best_results
