import time
import math
import numpy as np
from random import Random

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

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph
        # Conflict if share vertex or connected by edge
        neigh = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Share vertex?
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].add(j)
                    neigh[j].add(i)
                else:
                    # Joined by an edge?
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].add(j)
                        neigh[j].add(i)
        
        # Convert to list for faster iteration
        neigh_list = [list(s) for s in neigh]
        
        def find_max_is(active_set, rng):
            """
            Find a large independent set in the subgraph induced by active_set.
            Uses a greedy heuristic with randomization.
            """
            if not active_set:
                return set()
            
            # Calculate degrees within active_set
            # Since m is small, we can just count on the fly or precompute
            # Let's use a simple greedy: pick min degree, remove neighbors.
            # To make it robust, we do multiple restarts outside.
            
            active = set(active_set)
            is_set = set()
            
            while active:
                # Find vertex with min degree in active subgraph
                min_deg = float('inf')
                candidates = []
                for v in active:
                    deg = 0
                    for u in neigh_list[v]:
                        if u in active:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
                
                # Random selection among min-degree candidates to break ties randomly
                # This helps with restarts
                v = rng.choice(candidates)
                
                is_set.add(v)
                
                # Remove v and its neighbors from active
                active.remove(v)
                for u in neigh_list[v]:
                    if u in active:
                        active.remove(u)
                        
            return is_set

        def solve_with_peeling(rng, time_limit):
            active = set(range(m))
            colors = [-1] * m
            current_color = 0
            
            while active:
                if time.time() > time_limit:
                    break
                
                best_is = set()
                # Try multiple restarts to find a large IS
                # Number of restarts depends on remaining time and active size
                # Smaller active sets -> fewer restarts needed? Or more?
                # Let's scale restarts with time remaining.
                remaining_time = time_limit - time.time()
                if remaining_time < 0:
                    break
                
                # Heuristic: more restarts when we have more time
                # But we must ensure we finish.
                # Let's do a fixed number of restarts, say 50-200, depending on m.
                num_restarts = 100
                if len(active) < 20:
                    num_restarts = 500 # Small graph, can afford more
                elif len(active) < 50:
                    num_restarts = 200
                else:
                    num_restarts = 100
                
                for _ in range(num_restarts):
                    if time.time() > time_limit - 0.1: # Safety margin
                        break
                    is_set = find_max_is(active, rng)
                    if len(is_set) > len(best_is):
                        best_is = is_set
                
                if not best_is:
                    # Should not happen if active is not empty
                    # Fallback: take any vertex
                    best_is = {next(iter(active))}
                
                for v in best_is:
                    colors[v] = current_color
                
                active.difference_update(best_is)
                current_color += 1
            
            if -1 in colors:
                # Timeout or failure, assign remaining arbitrarily
                next_c = current_color
                for i in range(m):
                    if colors[i] == -1:
                        colors[i] = next_c
                        next_c += 1
                        # This might violate constraints, but we return best effort
                        # Actually, we should try to fix it or just return invalid?
                        # The task says "invalid result... scores 0".
                        # So we must ensure validity.
                        # If we time out, we need a valid coloring.
                        # The peeling process guarantees validity if it completes.
                        # If it breaks early, the remaining vertices are uncolored.
                        # We can just color them with new colors (1 color per vertex) to ensure validity.
                        pass
            
            return colors

        rng = Random(seed * 7919 + 42)
        
        # Try to solve with peeling
        # Allocate time
        start = time.time()
        # We don't know total budget here, but we get a deadline.
        # Let's assume the caller manages the deadline.
        
        best_coloring = None
        best_num_colors = float('inf')
        
        # Run peeling multiple times with different seeds to find a better packing
        # How many times?
        # We can run until time is up, keeping the best result.
        
        while time.time() < deadline - 1.0:
            # Use a new random seed for each attempt?
            # Or just continue with the same rng?
            # Let's use a fresh rng for each attempt to ensure diversity.
            attempt_rng = Random(rng.random())
            
            coloring = solve_with_peeling(attempt_rng, deadline - 1.0)
            
            # Check validity (optional, but good for debugging)
            # And count colors
            num_colors = max(coloring) + 1 if coloring else 0
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]
                
            # If we hit the theoretical lower bound or a very good number, we might stop early?
            # Lower bound is max degree of conflict graph? No, chromatic number >= max degree + 1?
            # For strong edge coloring, lower bound is not trivial.
            # But if we reach the Erdos-Nesetril bound, we can stop.
            en_bound = int(5 * d * d / 4)
            if best_num_colors <= en_bound:
                break

        if best_coloring is None:
            # Fallback: simple greedy
            # This should rarely happen if peeling works
            active = set(range(m))
            colors = [-1] * m
            c = 0
            while active:
                # Greedy: pick first vertex, color it, remove neighbors
                v = next(iter(active))
                colors[v] = c
                active.remove(v)
                for u in neigh_list[v]:
                    if u in active:
                        active.remove(u)
                c += 1
            best_coloring = colors
            
        return compress(best_coloring)

    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
