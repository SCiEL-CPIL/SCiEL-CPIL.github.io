import time
import sys
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, time_budget, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)
                        
        # Initial greedy coloring (DSATUR-like or simple degree ordering)
        deg = [len(neigh[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: deg[x], reverse=True)
        
        color = [-1] * m
        used_colors = set()
        # Simple greedy: assign smallest available color
        for v in order:
            forbidden = set()
            for u in neigh[v]:
                if color[u] != -1:
                    forbidden.add(color[u])
            c = 0
            while c in forbidden:
                c += 1
            color[v] = c
            used_colors.add(c)
            
        current_cost = max(color) + 1
        best_color = color[:]
        best_cost = current_cost
        
        # Local Search: Simulated Annealing
        # Objective: Minimize number of colors (max color index + 1)
        # Moves:
        # 1. Recolor vertex v to color c.
        #    - If c is already used by a non-neighbor, cost might decrease if c was the max color and v was the only one using it? No, cost is max color.
        #    - Actually, if we recolor v to an existing color c < max_color, cost doesn't change unless v was the only user of max_color.
        #    - If we recolor v to a new color > max_color, cost increases.
        #    - If we recolor v from max_color to an existing color < max_color, cost decreases if no other vertex has max_color.
        
        # Better Objective for SA: Sum of colors? No, we want min max.
        # Let's stick to Min Max Color.
        # However, SA needs a continuous or quasi-continuous landscape.
        # "Max color" is stepwise.
        # Alternative: Minimize sum of colors? Often correlates.
        # Or: Minimize number of color classes.
        
        # Let's use a standard graph coloring local search heuristic:
        # 1. Start with greedy.
        # 2. Try to reduce number of colors by:
        #    a. Removing a color: Find a color c that is used by few vertices. Try to recolor all vertices with color c to other colors. If successful, remove c.
        #    b. Recoloring: For a vertex v, try to change its color to a neighbor's color if valid.
        
        # Given the small size (m <= 200), we can do a more exhaustive local improvement.
        # Strategy:
        # 1. Greedy initialization.
        # 2. Iteratively try to reduce the number of colors by 1.
        #    To do this, we need to show that a K-1 coloring exists.
        #    We can use a randomized greedy re-coloring or a simple backtracking on the "merge" problem.
        #    
        # Actually, the current program already does backtracking. The inspiration says "local improvement iterations".
        # Let's implement a "color removal" heuristic.
        # To remove color K (the highest one):
        #   Let S be the set of vertices with color K.
        #   We need to recolor all v in S to colors < K.
        #   This is a subproblem.
        #   We can try to recolor them one by one, or use a small backtracking.
        
        # Let's implement a randomized restart local search that tries to minimize the number of colors.
        # We will use a "min-conflicts" style or "simulated annealing" on the number of colors.
        
        # Since the graph is small, we can afford to run multiple random restarts of a greedy algorithm with random ordering,
        # followed by a "color reduction" pass.
        
        def try_reduce_colors(color, K):
            # Try to reduce number of colors from K to K-1
            # Identify vertices with color K-1 (0-indexed, so the highest color)
            # Let's assume colors are 0..K-1. We want to remove color K-1.
            # Vertices with color K-1: V_K
            # We need to recolor each v in V_K to a color in 0..K-2 such that no conflict arises.
            
            V_K = [i for i in range(m) if color[i] == K-1]
            if not V_K:
                return color, K-1 # Already reduced? Unlikely if K is max+1.
            
            # Sort V_K by degree (highest first) to make it harder? Or lowest?
            # Usually, recoloring high degree vertices is harder.
            # Let's try to recolor them.
            
            # We can use a simple backtracking here just for these vertices.
            # Or randomized greedy.
            
            # Let's use a randomized greedy with restarts for this subproblem.
            # Actually, let's just do a full backtracking on the whole graph with K-1 colors?
            # That's what the original program did.
            
            # Instead, let's do a "move-based" local search.
            # State: color[]
            # Move: Pick a vertex v, change color[v] to c.
            # Cost: Number of colors used.
            
            # This is tricky because cost is discrete.
            
            # Let's stick to the "Iterated Local Search" on the coloring.
            # 1. Greedy coloring.
            # 2. Local Improvement: 
            #    - While possible:
            #       - Try to remove the highest color.
            #       - If fails, try to recolor vertices to merge other colors?
            
            # Given the time budget, let's do this:
            # 1. Generate a random permutation of vertices.
            # 2. Greedy color.
            # 3. Try to reduce colors by 1 using a simple backtracking on the vertices of the highest color.
            
            pass # Placeholder, we will implement the main loop below

        # Main Loop: Random Restarts + Greedy + Local Reduction
        rng = Random(seed * 12345 + 7)
        start_time = time.time()
        
        best_overall = best_color
        best_overall_cost = best_cost
        
        # We will try to improve upon the initial greedy result.
        # The initial greedy is decent.
        
        # Let's try to reduce the number of colors step by step.
        # Current K = best_cost.
        # Try to find a (K-1)-coloring.
        # If found, set K = K-1 and repeat.
        
        # To find a (K-1)-coloring, we can use the backtracking solver from the original program,
        # but restricted to K-1.
        # However, the original program's backtracking was the bottleneck.
        
        # Let's use a "Conflict-driven" local search.
        # 1. Start with a K-coloring.
        # 2. Randomly pick a vertex v.
        # 3. Randomly pick a color c < current_max_color.
        # 4. If valid, assign.
        # 5. If current_max_color is no longer used, we have reduced K!
        
        # This is essentially a random walk on the space of K-colorings, hoping to hit a (K-1)-coloring.
        
        def random_walk_reduce(color, K, steps, rng, deadline):
            # color is 0-indexed, max color is K-1
            # We want to eliminate color K-1.
            # We perform random valid recolorings.
            # If at any point color K-1 is not used, we return success.
            
            # Count usage of K-1
            count_max = sum(1 for c in color if c == K-1)
            
            for _ in range(steps):
                if time.time() > deadline:
                    break
                # Pick a random vertex
                v = rng.randrange(m)
                # Pick a random color from 0 to K-2
                c = rng.randrange(K-1)
                
                # Check if c is valid for v
                valid = True
                for u in neigh[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if valid:
                    old_c = color[v]
                    color[v] = c
                    
                    # If we just removed the last usage of K-1?
                    # No, we only change v to c < K-1.
                    # So if old_c was K-1, count_max decreases.
                    if old_c == K-1:
                        count_max -= 1
                        if count_max == 0:
                            return color, True # Success!
                    
                    # If we increased usage of some other color? Doesn't matter for max.
                    
                    # Optional: If we increased count of some other color to be the new max?
                    # No, we only assign c < K-1, so max color cannot increase.
                    # It can only decrease or stay same.
                    
            return color, False

        # Strategy:
        # 1. Get initial greedy coloring (K0).
        # 2. Loop:
        #    a. Try to reduce K by 1 using random_walk_reduce with many steps.
        #    b. If successful, K = K-1, continue.
        #    c. If failed, try a different random restart of the greedy to get a different starting K-coloring.
        
        current_color = best_color
        current_K = best_cost
        
        # We will use a time budget for this instance.
        # Let's say we spend 80% of the time on this.
        
        while time.time() < deadline - 1.0:
            # Try to reduce current_K by 1
            # We need a coloring with current_K colors.
            # We have current_color which uses current_K colors.
            
            # Run random walk for a certain number of steps or time slice
            steps = 10000
            new_color, success = random_walk_reduce(current_color[:], current_K, steps, rng, deadline - 1.0)
            
            if success:
                current_color = new_color
                current_K -= 1
                best_overall = current_color
                best_overall_cost = current_K
                # If we reach a lower bound, stop?
                # Lower bound is max_degree + 1.
                if current_K <= max(deg) + 1:
                    break
            else:
                # Failed to reduce.
                # Try a new random restart of the greedy to get a different K-coloring.
                # Maybe the current K-coloring is "stuck".
                
                # Generate new random permutation
                perm = list(range(m))
                rng.shuffle(perm)
                
                new_col = [-1] * m
                for v in perm:
                    forbidden = set()
                    for u in neigh[v]:
                        if new_col[u] != -1:
                            forbidden.add(new_col[u])
                    c = 0
                    while c in forbidden:
                        c += 1
                    new_col[v] = c
                
                new_K = max(new_col) + 1
                if new_K < current_K:
                    current_color = new_col
                    current_K = new_K
                    best_overall = current_color
                    best_overall_cost = current_K
                elif new_K == current_K:
                    # Use this as a new starting point for random walk
                    current_color = new_col
                    # Don't update best if not better, but use for search
                    # Actually, if it's the same K, it might be easier to reduce.
                    # So we keep searching with this new instance.
                
                # If we've been failing for a long time, maybe we should stop?
                # The loop condition handles the time.
        
        return compress(best_overall)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = min(remaining / (len(specs) - idx), 45.0)
            share = max(share, 0.5)
            out.append(solve_instance(n, d, seed, share, deadline))
        return out
    except Exception:
        return fallback()
