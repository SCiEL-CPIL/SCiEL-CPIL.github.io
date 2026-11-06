import time
import sys
from random import Random
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

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph on edges
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Conflict: two edges conflict if they share a vertex OR are joined by an edge
        conflict = [0] * m
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        def find_large_independent_set(vertices, time_limit):
            """Find a large independent set in the induced subgraph on 'vertices' (a bitset)."""
            if vertices == 0:
                return 0
            
            # Greedy heuristic to get a lower bound
            greedy_set = 0
            remaining = vertices
            while remaining:
                # Pick vertex with minimum degree in remaining
                min_v = -1
                min_deg = float('inf')
                v = (remaining & -remaining).bit_length() - 1
                while v < m:
                    deg_v = bin(conflict[v] & remaining).count('1')
                    if deg_v < min_deg:
                        min_deg = deg_v
                        min_v = v
                    remaining &= remaining - 1
                    v = (remaining & -remaining).bit_length() - 1 if remaining else -1
                if min_v == -1:
                    break
                greedy_set |= (1 << min_v)
                remaining &= ~(conflict[min_v] | (1 << min_v))
            
            lb = bin(greedy_set).count('1')
            
            # Branch and bound
            best_set = greedy_set
            best_size = lb
            
            def bbs(current, available, depth):
                nonlocal best_set, best_size
                if time.time() > time_limit:
                    return
                if available == 0:
                    if bin(current).count('1') > best_size:
                        best_size = bin(current).count('1')
                        best_set = current
                    return
                
                # Upper bound: current size + popcount(available)
                if bin(current).count('1') + bin(available).count('1') <= best_size:
                    return
                
                # Pick a vertex to branch on
                v = (available & -available).bit_length() - 1
                while v < m:
                    break
                    v = (available & -available).bit_length() - 1
                v = (available & -available).bit_length() - 1
                
                # Branch: include v
                new_current = current | (1 << v)
                new_available = available & ~(conflict[v] | (1 << v))
                bbs(new_current, new_available, depth + 1)
                
                # Branch: exclude v
                bbs(current, available & ~(1 << v), depth + 1)
            
            bbs(0, vertices, 0)
            return best_set

        def greedy_coloring(deadline):
            """Greedy coloring using large independent set extraction."""
            color = [-1] * m
            remaining = (1 << m) - 1
            color_id = 0
            
            while remaining and time.time() < deadline - 5.0:
                # Find large independent set in remaining
                iso_set = find_large_independent_set(remaining, deadline - 5.0)
                
                if iso_set == 0:
                    # Fallback: just pick one vertex
                    v = (remaining & -remaining).bit_length() - 1
                    iso_set = 1 << v
                
                # Assign color
                s = iso_set
                while s:
                    v = (s & -s).bit_length() - 1
                    color[v] = color_id
                    s &= s - 1
                
                remaining &= ~iso_set
                color_id += 1
            
            # Assign remaining
            if remaining:
                s = remaining
                while s:
                    v = (s & -s).bit_length() - 1
                    color[v] = color_id
                    s &= s - 1
                color_id += 1
            
            return color, color_id

        # Get initial coloring
        color, K = greedy_coloring(deadline)
        
        if time.time() > deadline - 3.0:
            return compress(color)
        
        # Try to improve by re-extracting with different orderings
        best_color = color[:]
        best_K = K
        rng = Random(seed * 999 + 1)
        
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            
            # Shuffle the order in which we process vertices
            order = list(range(m))
            rng.shuffle(order)
            
            # Try greedy with different vertex selection heuristic
            trial_color = [-1] * m
            remaining = (1 << m) - 1
            color_id = 0
            
            while remaining and time.time() < deadline - 5.0:
                # Find large independent set, but with some randomization
                # Use multiple random starts for the greedy part
                best_iso = 0
                best_size = 0
                for _ in range(5):
                    # Randomized greedy
                    greedy_set = 0
                    rem = remaining
                    shuffled_order = list(range(m))
                    rng.shuffle(shuffled_order)
                    for v in shuffled_order:
                        if rem & (1 << v):
                            if not (greedy_set & conflict[v]):
                                greedy_set |= (1 << v)
                                rem &= ~(conflict[v] | (1 << v))
                    if bin(greedy_set).count('1') > best_size:
                        best_size = bin(greedy_set).count('1')
                        best_iso = greedy_set
                
                if best_iso == 0:
                    v = (remaining & -remaining).bit_length() - 1
                    best_iso = 1 << v
                
                s = best_iso
                while s:
                    v = (s & -s).bit_length() - 1
                    trial_color[v] = color_id
                    s &= s - 1
                
                remaining &= ~best_iso
                color_id += 1
            
            if remaining:
                s = remaining
                while s:
                    v = (s & -s).bit_length() - 1
                    trial_color[v] = color_id
                    s &= s - 1
                color_id += 1
            
            if color_id < best_K:
                best_K = color_id
                best_color = trial_color[:]
        
        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
