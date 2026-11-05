import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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
        # Convert to lists for speed
        return [list(c) for c in conflicts]

    def greedy_color(m, conflicts, order):
        """Greedy coloring in given order. Returns colors array."""
        colors = [-1] * m
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if colors[j] != -1:
                    forbidden.add(colors[j])
            c = 0
            while c in forbidden:
                c += 1
            colors[i] = c
        return colors

    def max_color(colors):
        """Return max color index + 1 (number of colors)."""
        if not colors:
            return 0
        return max(colors) + 1

    def is_valid_coloring(colors, conflicts):
        """Verify that no two conflicting edges share a color."""
        m = len(colors)
        for i in range(m):
            ci = colors[i]
            for j in conflicts[i]:
                if j > i and colors[j] == ci:
                    return False
        return True

    def iterated_local_search(m, conflicts, max_iters=200, time_limit=None):
        """
        ILS with simulated annealing-style acceptance.
        State: colors array.
        Objective: minimize max_color(colors).
        Moves: single-edge recolor, 2-swap.
        """
        rng = random.Random(12345)
        
        # Initialize with greedy (descending degree)
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        colors = greedy_color(m, conflicts, order)
        best_colors = colors[:]
        best_obj = max_color(colors)
        
        # Precompute conflict neighbors as lists
        # For speed, we'll use the conflicts list directly
        
        current_temp = 10.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # --- Local Search Phase: greedy improvement ---
            improved = True
            while improved:
                improved = False
                # Try recoloring each edge to a lower color
                for i in rng.sample(range(m), m):
                    ci = colors[i]
                    if ci == 0:
                        continue
                    forbidden = set()
                    for j in conflicts[i]:
                        if colors[j] != -1:
                            forbidden.add(colors[j])
                    # Try to find a color < ci not in forbidden
                    for c in range(ci):
                        if c not in forbidden:
                            colors[i] = c
                            improved = True
                            break
                # Check if we improved the objective
                cur_obj = max_color(colors)
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_colors = colors[:]
            
            # --- Perturbation Phase ---
            # Randomly recolor a subset of edges to break local optimum
            num_perturb = max(1, m // 10)
            for _ in range(num_perturb):
                i = rng.randrange(m)
                forbidden = set()
                for j in conflicts[i]:
                    if colors[j] != -1:
                        forbidden.add(colors[j])
                # Pick a random available color
                available = []
                max_c = max_color(colors) + 2
                for c in range(max_c):
                    if c not in forbidden:
                        available.append(c)
                if available:
                    colors[i] = rng.choice(available)
            
            # Try 2-swap perturbation
            if m >= 2:
                i, j = rng.sample(range(m), 2)
                ci, cj = colors[i], colors[j]
                if ci != cj:
                    # Check validity of swap
                    valid = True
                    for k in conflicts[i]:
                        if k != j and colors[k] == cj:
                            valid = False
                            break
                    if valid:
                        for k in conflicts[j]:
                            if k != i and colors[k] == ci:
                                valid = False
                                break
                    if valid:
                        colors[i], colors[j] = cj, ci
            
            # --- Acceptance ---
            cur_obj = max_color(colors)
            if cur_obj <= best_obj:
                # Accept (it's an improvement or equal)
                current_temp *= cooling
            else:
                # Simulated annealing acceptance
                delta = cur_obj - best_obj
                if current_temp > 0.01:
                    import math
                    prob = math.exp(-delta / current_temp)
                    if rng.random() < prob:
                        current_temp *= cooling
                    else:
                        # Reject: revert to best
                        colors = best_colors[:]
                else:
                    # Reject: revert to best
                    colors = best_colors[:]
        
        return best_colors, best_obj

    def color_graph(edges, adj, time_budget=10.0):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflict_graph(edges, adj)
        
        # Run ILS with time budget
        t_start = time.time()
        t_limit = t_start + time_budget
        
        # Multiple restarts with different random seeds for ILS
        best_colors = None
        best_obj = float('inf')
        
        restart_count = 0
        while time.time() < t_limit:
            restart_count += 1
            # Each ILS run has its own internal time limit
            inner_limit = min(t_limit, time.time() + 3.0)
            
            # Modify the ILS to use a specific seed
            # For simplicity, we'll just call it; the internal rng is fixed.
            # To get diversity, we could pass a seed, but let's just run multiple.
            # Actually, let's add a seed parameter to ILS.
            
            colors, obj = _ils_with_seed(m, conflicts, seed=restart_count, time_limit=inner_limit)
            
            if obj < best_obj:
                best_obj = obj
                best_colors = colors[:]
            
            # If we hit the theoretical lower bound or conjecture bound, stop
            # Conjecture: 5d^2/4. We don't have d here directly, but we can estimate.
            # Actually, we can just stop if best_obj is small enough.
            # For now, just continue until time runs out.
        
        if best_colors is None:
            # Fallback: simple greedy
            order = list(range(m))
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)
            best_colors = greedy_color(m, conflicts, order)
            best_obj = max_color(best_colors)
        
        return best_colors, best_obj

    def _ils_with_seed(m, conflicts, seed=0, max_iters=100, time_limit=None):
        """ILS with a specific random seed."""
        rng = random.Random(seed)
        
        # Initialize with greedy (descending degree)
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        colors = greedy_color(m, conflicts, order)
        best_colors = colors[:]
        best_obj = max_color(colors)
        
        current_temp = 10.0
        cooling = 0.99
        
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # --- Local Search Phase: greedy improvement ---
            improved = True
            while improved:
                improved = False
                for i in rng.sample(range(m), m):
                    ci = colors[i]
                    if ci == 0:
                        continue
                    forbidden = set()
                    for j in conflicts[i]:
                        if colors[j] != -1:
                            forbidden.add(colors[j])
                    for c in range(ci):
                        if c not in forbidden:
                            colors[i] = c
                            improved = True
                            break
                cur_obj = max_color(colors)
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_colors = colors[:]
            
            # --- Perturbation Phase ---
            num_perturb = max(1, m // 8)
            for _ in range(num_perturb):
                i = rng.randrange(m)
                forbidden = set()
                for j in conflicts[i]:
                    if colors[j] != -1:
                        forbidden.add(colors[j])
                available = []
                max_c = max_color(colors) + 3
                for c in range(max_c):
                    if c not in forbidden:
                        available.append(c)
                if available:
                    colors[i] = rng.choice(available)
            
            # 2-swap perturbation
            if m >= 2:
                i, j = rng.sample(range(m), 2)
                ci, cj = colors[i], colors[j]
                if ci != cj:
                    valid = True
                    for k in conflicts[i]:
                        if k != j and colors[k] == cj:
                            valid = False
                            break
                    if valid:
                        for k in conflicts[j]:
                            if k != i and colors[k] == ci:
                                valid = False
                                break
                    if valid:
                        colors[i], colors[j] = cj, ci
            
            # --- Acceptance ---
            cur_obj = max_color(colors)
            if cur_obj <= best_obj:
                current_temp *= cooling
            else:
                delta = cur_obj - best_obj
                if current_temp > 0.01:
                    import math
                    prob = math.exp(-delta / current_temp)
                    if rng.random() < prob:
                        current_temp *= cooling
                    else:
                        colors = best_colors[:]
                else:
                    colors = best_colors[:]
        
        return best_colors, best_obj

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
    
    best_results = [None] * len(graphs)
    best_scores = [float('inf')] * len(graphs)
    
    # Allocate time per graph
    # We have ~145s total. Let's give each graph ~45s.
    time_per_graph = 45.0
    
    for idx, (edges, adj) in enumerate(graphs):
        t_start = time.time()
        t_limit = t_start + time_per_graph
        
        # Run ILS with multiple restarts
        restart = 0
        while time.time() < t_limit:
            restart += 1
            inner_limit = min(t_limit, time.time() + 5.0)
            
            colors, obj = _ils_with_seed(len(edges), build_conflict_graph(edges, adj), seed=restart, time_limit=inner_limit)
            
            if obj < best_scores[idx]:
                best_scores[idx] = obj
                best_results[idx] = colors
            
            # Check if we've hit a good bound
            # For d=4, conjecture is 20. For d=5, 31. For d=6, 45.
            # If we're close, we can stop early.
            d = specs[idx][1]
            conjecture = 5 * d * d // 4
            if best_scores[idx] <= conjecture:
                break
        
        # Ensure we have a result
        if best_results[idx] is None:
            conflicts = build_conflict_graph(edges, adj)
            order = list(range(len(edges)))
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)
            best_results[idx] = greedy_color(len(edges), conflicts, order)
            best_scores[idx] = max_color(best_results[idx])
    
    return best_results
