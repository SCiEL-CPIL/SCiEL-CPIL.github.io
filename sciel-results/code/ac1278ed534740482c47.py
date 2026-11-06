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

        # Build conflict graph: two edges conflict if they share a vertex or are joined by an edge
        # adj_orig is the original graph adjacency
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # neigh[i] is a bitset of edges that conflict with edge i
        # Two edges (a,b) and (c,d) conflict if:
        # 1. They share a vertex: a==c, a==d, b==c, b==d
        # 2. They are joined by an edge in the original graph: c in adj[a], etc.
        neigh = [0] * m
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i] |= (1 << j)
                    neigh[j] |= (1 << i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i] |= (1 << j)
                        neigh[j] |= (1 << i)

        # Precompute degrees for ordering
        deg = [bin(neigh[i]).count('1') for i in range(m)]
        
        # Maximum Independent Set via Branch-and-Bound
        # We want to find the largest subset of edges that form an induced matching
        # available: bitset of available edges
        # best_size: current best size found
        # best_set: bitset of the best set found
        # deadline_check: function to check if we should stop
        
        def find_mis(available, best_size, deadline_check):
            """
            Branch and bound to find maximum independent set in available set.
            Returns (size, bitset_of_set)
            """
            if available == 0:
                return 0, 0
            if deadline_check():
                return 0, 0
            
            # Greedy upper bound
            # A simple bound: number of bits in available
            # A better bound: sum of 1 + max_degree for each vertex? No, that's not right.
            # Standard bound: For each vertex v, we can pick v and at most deg(v) others? No.
            # Tomita's bound: For each vertex v in available, we can pick v and then the MIS of the rest.
            # A simpler admissible bound: The size of the maximum independent set is at most the size of the minimum vertex cover.
            # Or simply: len(available) is a weak bound.
            
            # Let's use a greedy coloring bound (Tomita et al.)
            # Color the subgraph induced by 'available'. The number of colors is an upper bound on the size of the independent set.
            # But coloring is expensive.
            
            # Let's use a simpler bound: 
            # If we pick a vertex v, we remove v and its neighbors.
            # The bound is 1 + bound(available - v - N(v)).
            
            # Heuristic: Pick vertex with minimum degree in the induced subgraph?
            # Or maximum degree? 
            # For MIS, picking a vertex with high degree removes many candidates, which might be good for pruning?
            # Actually, picking a vertex with low degree is often better for finding large sets?
            # Let's try picking the vertex with the fewest neighbors in 'available'.
            
            avail_list = [i for i in range(m) if (available >> i) & 1]
            if not avail_list:
                return 0, 0
            
            # Calculate degree in induced subgraph
            min_deg = m
            pivot = avail_list[0]
            for v in avail_list:
                # Count neighbors in available
                cnt = bin(neigh[v] & available).count('1')
                if cnt < min_deg:
                    min_deg = cnt
                    pivot = v
                if min_deg == 0:
                    break
            
            # Branch 1: Include pivot
            # Remove pivot and its neighbors
            new_avail_1 = available & ~(1 << pivot) & ~neigh[pivot]
            size1, set1 = find_mis(new_avail_1, best_size, deadline_check)
            size1 += 1
            set1 |= (1 << pivot)
            
            # Branch 2: Exclude pivot
            new_avail_2 = available & ~(1 << pivot)
            size2, set2 = find_mis(new_avail_2, best_size, deadline_check)
            
            if size1 >= size2:
                return size1, set1
            else:
                return size2, set2

        # To speed up B&B, we can use a simple greedy coloring bound.
        # But let's first see if the simple B&B is fast enough.
        # For n=80, m~240, the conflict graph is sparse.
        
        # Let's implement a faster B&B with a greedy bound.
        # We need to color the subgraph induced by 'available' to get an upper bound.
        # This is complex. Let's stick to the simple one but optimize the pivot selection.
        
        # Optimization: Order vertices by degree.
        # The `find_mis` above recalculates degrees every time.
        
        # Let's try a different approach: 
        # 1. Get a greedy coloring of the conflict graph to get an upper bound on K.
        # 2. Try to find a K-coloring using B&B on the complement? No.
        
        # Let's stick to MIS extraction.
        # To make it faster, we can limit the depth or use a timeout per extraction.
        
        def greedy_coloring():
            """Standard greedy coloring with DSATUR-like ordering."""
            color = [-1] * m
            uncolored = list(range(m))
            while uncolored:
                # Pick vertex with highest saturation
                best_v = None
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(set(color[u] for u in range(m) if (neigh[v] >> u) & 1 and color[u] != -1))
                    if sat > best_sat or (sat == best_sat and deg[v] > best_deg):
                        best_sat = sat
                        best_deg = deg[v]
                        best_v = v
                forbidden = set(color[u] for u in range(m) if (neigh[best_v] >> u) & 1 and color[u] != -1)
                c = 0
                while c in forbidden:
                    c += 1
                color[best_v] = c
                uncolored.remove(best_v)
            return color

        color0 = greedy_coloring()
        K0 = max(color0) + 1 if m > 0 else 0
        
        # Build coloring by extracting MIS
        def build_coloring_mis(deadline):
            color = [-1] * m
            available = (1 << m) - 1
            color_idx = 0
            
            while available:
                if time.time() > deadline:
                    break
                # Find MIS in available
                # To avoid B&B explosion, we can use a greedy heuristic to get a good initial set,
                # then try to improve it? Or just use B&B with a small time limit?
                
                # Let's use B&B. If it takes too long, we fall back to greedy.
                # For now, let's just run B&B.
                
                size, set_bits = find_mis(available, 0, lambda: time.time() > deadline)
                
                # Assign color
                for i in range(m):
                    if (set_bits >> i) & 1:
                        color[i] = color_idx
                
                # Remove chosen edges and their neighbors from available
                # The next available set is: available - (set_bits | N(set_bits))
                # N(set_bits) is the union of neighbors of all vertices in set_bits
                removed = 0
                for i in range(m):
                    if (set_bits >> i) & 1:
                        removed |= (1 << i)
                        removed |= neigh[i]
                
                available &= ~removed
                color_idx += 1
                
            # Fill in any remaining uncolored with greedy
            if available:
                for i in range(m):
                    if color[i] == -1:
                        forbidden = set(color[u] for u in range(m) if (neigh[i] >> u) & 1 and color[u] != -1)
                        c = 0
                        while c in forbidden:
                            c += 1
                        color[i] = c
            
            return color

        color_mis = build_coloring_mis(deadline - 1.0)
        K_mis = max(color_mis) + 1 if m > 0 else 0

        if K_mis < K0:
            best_color = color_mis
            best_K = K_mis
        else:
            best_color = color0
            best_K = K0

        return compress(best_color)

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
