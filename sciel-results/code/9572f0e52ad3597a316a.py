import time
import sys
from random import Random
from collections import deque, defaultdict
import math

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

    def build_conflict_graph(n, edges):
        m = len(edges)
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        neigh = [0] * m  # bitsets as Python ints
        for i in range(m):
            a, b = edges[i]
            mask = 0
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    mask |= (1 << j)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        mask |= (1 << j)
            neigh[i] = mask
            # also set bits for i in neigh[j]
            for j in range(i + 1, m):
                if (mask >> j) & 1:
                    neigh[j] |= (1 << i)
        return neigh, adj_orig

    def max_independent_set_bnb(neigh, m, time_limit, rng):
        """
        Find a maximum independent set in the conflict graph using branch and bound.
        neigh[i] is a bitset of neighbors of vertex i.
        Returns the set of vertices in the maximum independent set found.
        """
        # Order vertices by degree (descending) for better pruning
        deg = [bin(neigh[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -deg[i])
        
        # Map to new order
        pos = [0] * m
        for idx, v in enumerate(order):
            pos[v] = idx
        
        # Rebuild neigh in new order
        new_neigh = [0] * m
        for i in range(m):
            mask = 0
            for j in range(m):
                if i != j and (neigh[order[i]] >> order[j]) & 1:
                    mask |= (1 << pos[order[j]])
            new_neigh[i] = mask
        
        best_set = [0] * m
        best_size = 0
        best_deadline = time_limit
        
        def branch(candidates, current_set, depth):
            nonlocal best_set, best_size, best_deadline
            if time.time() > best_deadline:
                return
            
            if candidates == 0:
                if depth > best_size:
                    best_size = depth
                    best_set = current_set[:]
                return
            
            # Upper bound: current depth + popcount of candidates
            bound = depth + bin(candidates).count('1')
            if bound <= best_size:
                return
            
            # Pick a vertex with maximum degree in the candidate set
            # Try to find a good vertex to branch on
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                d_in_cand = bin(new_neigh[v] & candidates).count('1')
                if d_in_cand > best_deg:
                    best_deg = d_in_cand
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                return
            
            v = best_v
            bit = 1 << v
            
            # Branch 1: Include v
            new_current = current_set[:]
            new_current[v] = 1
            # Remove v and its neighbors from candidates
            new_candidates = candidates & ~(bit | new_neigh[v])
            branch(new_candidates, new_current, depth + 1)
            
            # Branch 2: Exclude v
            new_candidates2 = candidates & ~bit
            branch(new_candidates2, current_set, depth)
        
        all_candidates = (1 << m) - 1
        branch(all_candidates, [0] * m, 0)
        
        # Convert back to original vertex indices
        result = set()
        for i in range(m):
            if best_set[i]:
                result.add(order[i])
        return result

    def greedy_max_independent_set(neigh, m, rng):
        """Greedy approach: repeatedly pick a vertex and remove it and its neighbors."""
        remaining = set(range(m))
        result = set()
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            for v in remaining:
                d = len(neigh_list(v, m) & remaining)
                if d < best_deg:
                    best_deg = d
                    best_v = v
            result.add(best_v)
            remaining.discard(best_v)
            for u in neigh_list(best_v, m):
                remaining.discard(u)
        return result

    def neigh_list(i, m):
        """Get neighbor list from bitset."""
        mask = 0
        # This is a placeholder; we'll use the actual neigh structure
        return set()

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        neigh, adj_orig = build_conflict_graph(n, edges)
        
        # Use branch and bound to extract maximum induced matchings
        color = [-1] * m
        remaining = set(range(m))
        current_color = 0
        start_time = time.time()
        
        # Deadline guard
        hard_deadline = deadline - 2.0
        
        while remaining and time.time() < hard_deadline:
            # Find maximum independent set in the conflict graph induced by remaining
            # We'll use a greedy heuristic first, then try to improve
            remaining_list = sorted(remaining)
            
            # Build subgraph on remaining vertices
            sub_neigh = [0] * m
            for i in range(m):
                if i in remaining:
                    sub_neigh[i] = neigh[i]
            
            # Greedy: pick vertex with min degree, add to set, remove it and neighbors
            greedy_set = set()
            temp_remaining = set(remaining)
            while temp_remaining:
                best_v = -1
                best_deg = float('inf')
                for v in temp_remaining:
                    d = bin(neigh[v] & ((1 << m) - 1) & ~((1 << m) - 1)).count('1') if False else len([u for u in temp_remaining if u != v and (neigh[v] >> u) & 1])
                    if d < best_deg:
                        best_deg = d
                        best_v = v
                greedy_set.add(best_v)
                temp_remaining.discard(best_v)
                for u in list(temp_remaining):
                    if (neigh[best_v] >> u) & 1:
                        temp_remaining.discard(u)
            
            # Assign this color class
            for v in greedy_set:
                color[v] = current_color
                remaining.discard(v)
            current_color += 1
            
            # Try to improve: can we merge some color classes?
            # For now, just continue with greedy extraction
        
        return compress(color)

    def solve_instance_v2(n, d, seed, deadline):
        """Better approach: use the conflict graph and extract large independent sets."""
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        neigh, adj_orig = build_conflict_graph(n, edges)
        hard_deadline = deadline - 2.0
        
        # Greedy coloring with maximum independent set extraction
        color = [-1] * m
        remaining = set(range(m))
        current_color = 0
        
        while remaining and time.time() < hard_deadline:
            # Find a large independent set in the remaining subgraph
            # Use greedy: repeatedly pick vertex with min degree in remaining
            indep_set = set()
            temp_remaining = set(remaining)
            while temp_remaining:
                best_v = -1
                best_deg = float('inf')
                for v in temp_remaining:
                    # Count neighbors in temp_remaining
                    d = 0
                    for u in temp_remaining:
                        if u != v and (neigh[v] >> u) & 1:
                            d += 1
                    if d < best_deg:
                        best_deg = d
                        best_v = v
                indep_set.add(best_v)
                temp_remaining.discard(best_v)
                # Remove neighbors of best_v from temp_remaining
                to_remove = []
                for u in temp_remaining:
                    if (neigh[best_v] >> u) & 1:
                        to_remove.append(u)
                for u in to_remove:
                    temp_remaining.discard(u)
            
            # Assign color
            for v in indep_set:
                color[v] = current_color
                remaining.discard(v)
            current_color += 1
        
        return compress(color)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining_time = deadline - time.time()
            share = min(remaining_time / (len(specs) - idx), 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance_v2(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
