import time
import sys
from itertools import combinations
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        used_edges = set()
        
        partition = []
        
        # Sort I by degree in C (ascending)
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)))
        
        # Group independent vertices by their neighborhood in C
        neighborhood_groups = defaultdict(list)
        for v in I_sorted:
            N_v = frozenset(adj[v] & set(C))
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        def greedy_coloring(h_vertices, h_adj, order_key=None, rng=None, trials=1):
            """Try multiple orderings and return best coloring."""
            best_colors = None
            best_count = float('inf')
            
            def run_coloring(order_func):
                colors = {}
                color_set = {u: set() for u in h_vertices}
                uncolored = set(h_vertices)
                while uncolored:
                    if order_func:
                        best = order_func(uncolored, color_set, h_adj)
                    else:
                        best = max(uncolored, key=lambda x: (len(color_set[x]), len(h_adj[x])))
                    used_colors = color_set[best]
                    c = 0
                    while c in used_colors:
                        c += 1
                    colors[best] = c
                    color_set[best].add(c)
                    uncolored.discard(best)
                    for w in h_adj[best]:
                        if w in uncolored and c not in color_set[w]:
                            color_set[w].add(c)
                return colors, (max(colors.values()) + 1 if colors else 0)
            
            # DSATUR
            colors, cnt = run_coloring(None)
            if cnt < best_count:
                best_count = cnt
                best_colors = dict(colors)
            
            # Largest degree first
            def order_ldf(uncolored, color_set, h_adj):
                return max(uncolored, key=lambda x: len(h_adj[x]))
            colors, cnt = run_coloring(order_ldf)
            if cnt < best_count:
                best_count = cnt
                best_colors = dict(colors)
            
            # Smallest degree first
            def order_sdf(uncolored, color_set, h_adj):
                return min(uncolored, key=lambda x: len(h_adj[x]))
            colors, cnt = run_coloring(order_sdf)
            if cnt < best_count:
                best_count = cnt
                best_colors = dict(colors)
            
            # Randomized restarts
            if rng and trials > 0:
                for _ in range(trials):
                    def order_rand(uncolored, color_set, h_adj):
                        return max(uncolored, key=lambda x: (len(color_set[x]), len(h_adj[x]), rng.random()))
                    colors, cnt = run_coloring(order_rand)
                    if cnt < best_count:
                        best_count = cnt
                        best_colors = dict(colors)
            
            return best_colors, best_count
        
        rng_global = Random(42)
        
        for N_v_frozen, group_vertices in neighborhood_groups.items():
            N_v = set(N_v_frozen)
            n_v = len(N_v)
            
            if n_v == 1:
                u = next(iter(N_v))
                for v in group_vertices:
                    partition.append(sorted([v, u]))
                continue
            
            # Build conflict graph H on N_v using used_edges
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
            
            # Get best coloring
            colors, color_count = greedy_coloring(h_vertices, h_adj, rng=rng_global, trials=8)
            
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                clique = S + group_vertices
                partition.append(sorted(clique))
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
        rem_set = set(remaining_edges)
        
        # Build bitmask adjacency for remaining edges
        rem_adj_mask = [0] * k
        for u, v in rem_set:
            rem_adj_mask[u] |= (1 << v)
            rem_adj_mask[v] |= (1 << u)
        
        # Greedy max-clique extraction
        while rem_set:
            # Pick edge with highest common neighbor count
            best_edge = None
            best_score = -1
            for e in rem_set:
                u, v = e
                common = rem_adj_mask[u] & rem_adj_mask[v]
                common &= ~(1 << u) & ~(1 << v)
                score = bin(common).count('1')
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(rem_set))
                
            u, v = best_edge
            current_clique = {u, v}
            
            # Greedily add vertices
            changed = True
            while changed:
                changed = False
                candidates = rem_adj_mask[u] & rem_adj_mask[v]
                for x in current_clique:
                    candidates &= rem_adj_mask[x]
                candidates &= ~(1 << u) & ~(1 << v)
                for x in current_clique:
                    candidates &= ~(1 << x)
                
                best_w = -1
                best_deg = -1
                w = 0
                while w < k:
                    if (candidates >> w) & 1:
                        deg = bin(rem_adj_mask[w]).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                    w += 1
                
                if best_w >= 0:
                    current_clique.add(best_w)
                    changed = True
                    for x in current_clique:
                        if x != best_w:
                            rem_adj_mask[x] &= ~(1 << best_w)
                            rem_adj_mask[best_w] &= ~(1 << x)
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            clique_sorted = sorted(current_clique)
            for i in range(len(clique_sorted)):
                for j in range(i+1, len(clique_sorted)):
                    a, b = clique_sorted[i], clique_sorted[j]
                    rem_set.discard((min(a, b), max(a, b)))
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
