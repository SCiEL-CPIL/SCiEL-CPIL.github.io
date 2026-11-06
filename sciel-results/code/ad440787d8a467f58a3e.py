import time
import sys
import math
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

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph on edges (two edges conflict if they share a vertex
        # or are joined by an edge in the original graph)
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build adjacency as bitmasks for fast operations
        # m can be up to ~240, so use Python integers as bitsets
        neigh_mask = [0] * m
        for i in range(m):
            a, b = edges[i]
            mask_i = 0
            for j in range(m):
                if i == j:
                    continue
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    mask_i |= (1 << j)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        mask_i |= (1 << j)
            neigh_mask[i] = mask_i

        # Greedy coloring for upper bound (initial solution)
        def greedy_color():
            color = [-1] * m
            for i in range(m):
                forbidden = set()
                mask = neigh_mask[i]
                for j in range(m):
                    if (mask >> j) & 1 and color[j] != -1:
                        forbidden.add(color[j])
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
            return color

        color_greedy = greedy_color()
        K_upper = max(color_greedy) + 1

        # Branch and bound MIS solver
        # We want to find the maximum independent set in the conflict graph
        # An independent set in the conflict graph = induced matching in original

        def mis_branch_and_bound(candidates_mask, current_set, current_size, best_size, best_set, time_limit):
            """
            Find maximum independent set within candidates_mask.
            candidates_mask: bitmask of vertices still available
            current_set: list of vertices in current independent set
            Returns (size, set) of best found
            """
            if time.time() > time_limit:
                return current_size, list(current_set)

            # If no candidates left
            if candidates_mask == 0:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                    best_set[0] = list(current_set)
                return current_size, list(current_set)

            # Upper bound: current_size + popcount(candidates_mask)
            remaining = bin(candidates_mask).count('1')
            if current_size + remaining <= best_size[0]:
                return current_size, list(current_set)

            # Choose vertex with minimum degree in remaining subgraph (for branching)
            # For efficiency, pick the vertex with highest degree (branch on including/excluding)
            # Actually, for MIS, we branch: include v or exclude v
            # Better: pick vertex v, then all its neighbors cannot be in the set with v
            
            # Find a vertex to branch on - use the one with max degree in candidates
            best_v = -1
            best_deg = -1
            temp = candidates_mask
            while temp:
                # Get lowest set bit
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(neigh_mask[v] & candidates_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb

            if best_v == -1:
                return current_size, list(current_set)

            # Option 1: Include best_v in the independent set
            # Then we must exclude all neighbors of best_v
            v_mask = 1 << best_v
            neighbors = neigh_mask[best_v] & candidates_mask
            new_candidates = candidates_mask & ~v_mask & ~neighbors
            current_set.append(best_v)
            mis_branch_and_bound(new_candidates, current_set, current_size + 1, best_size, best_set, time_limit)
            current_set.pop()

            # Option 2: Exclude best_v
            new_candidates2 = candidates_mask & ~v_mask
            mis_branch_and_bound(new_candidates2, current_set, current_size, best_size, best_set, time_limit)

            return best_size[0], best_set[0]

        # Greedy heuristic for initial MIS lower bound
        def greedy_mis(candidates_mask):
            """Greedy: repeatedly pick vertex with min degree, add to set, remove it and its neighbors."""
            result = []
            mask = candidates_mask
            while mask:
                # Find vertex with minimum degree in remaining
                best_v = -1
                best_deg = float('inf')
                temp = mask
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    deg = bin(neigh_mask[v] & mask).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp ^= lsb
                result.append(best_v)
                v_mask = 1 << best_v
                mask &= ~v_mask & ~neigh_mask[best_v]
            return result

        # Extract color classes using B&B MIS
        color = [-1] * m
        remaining_mask = (1 << m) - 1  # all edges still uncolored
        K = 0

        # Use the greedy coloring as an initial upper bound on number of colors needed
        # Try to beat it by extracting large independent sets

        rng = Random(seed * 104729 + 7)

        # Strategy: iteratively extract maximum independent sets
        # Use B&B with time limit per extraction, but also try multiple orders
        # For small instances, B&B should be fast

        # First, get a quick greedy solution for reference
        greedy_solution = greedy_mis(remaining_mask)
        greedy_K = 0
        temp_mask = remaining_mask
        while temp_mask:
            greedy_solution = greedy_mis(temp_mask)
            for v in greedy_solution:
                color[v] = greedy_K
            for v in greedy_solution:
                temp_mask &= ~(1 << v) & ~neigh_mask[v]
            greedy_K += 1

        # Now try B&B to do better
        # Process vertices in order of decreasing degree for better pruning
        # Actually, let's just use B&B directly with a good time budget

        # Reinitialize
        color = [-1] * m
        remaining_mask = (1 << m) - 1
        K = 0
        best_total_K = greedy_K
        best_color = color_greedy[:]  # start with greedy coloring as fallback

        # Try to extract MIS using B&B for each color class
        # Time budget: use most of the deadline
        time_per_class = 2.0  # seconds per color class extraction
        max_restarts = 5

        for restart in range(max_restarts):
            if time.time() > deadline - 3.0:
                break

            color = [-1] * m
            remaining_mask = (1 << m) - 1
            K = 0

            while remaining_mask and time.time() < deadline - 3.0:
                if K >= best_total_K:
                    break  # already worse than best, stop

                # Extract maximum independent set from remaining_mask
                best_size = [0]
                best_set = [[]]
                time_limit = min(time.time() + time_per_class, deadline - 3.0)

                # Use B&B
                mis_branch_and_bound(remaining_mask, [], 0, best_size, best_set, time_limit)

                if best_size[0] == 0:
                    # Fallback: just pick one vertex
                    lsb = remaining_mask & (-remaining_mask)
                    v = lsb.bit_length() - 1
                    best_set[0] = [v]

                # Assign this color
                for v in best_set[0]:
                    color[v] = K

                # Remove these vertices and their neighbors from remaining
                for v in best_set[0]:
                    remaining_mask &= ~(1 << v) & ~neigh_mask[v]

                K += 1

            if K < best_total_K:
                best_total_K = K
                best_color = color[:]

        # If B&B didn't improve, use greedy
        if best_total_K >= K_upper:
            best_color = color_greedy
            best_total_K = K_upper

        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 135.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 42.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
