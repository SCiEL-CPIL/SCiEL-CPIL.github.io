import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    from random import Random

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
        
        # Phase 1: Process I vertices using matching-inspired grouping
        # Sort by neighborhood size descending to handle dense connections first
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
            if len(N_v) == 1:
                u = list(N_v)[0]
                partition.append(sorted([v, u]))
                continue
            
            # Build conflict graph: two vertices in N_v conflict if their edge is used
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
            
            # DSATUR coloring for minimum color classes
            colors = {}
            sat_degree = {u: 0 for u in h_vertices}
            color_set = {u: set() for u in h_vertices}
            uncolored = set(h_vertices)
            
            while uncolored:
                best = max(uncolored, key=lambda x: (sat_degree[x], len(h_adj[x] & uncolored)))
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
                        sat_degree[w] = len(color_set[w])
            
            color_count = max(colors.values()) + 1 if colors else 0
            color_classes = {}
            for u, c in colors.items():
                color_classes.setdefault(c, []).append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                clique = S + [v]
                partition.append(sorted(clique))
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
        
        # Phase 2: Partition remaining C-edges using matching in line graph
        rem_set = set(internal_edges - used_edges)
        
        while rem_set:
            if time.time() > deadline:
                # Fallback: pair up remaining edges
                rem_list = list(rem_set)
                for e in rem_list:
                    partition.append(list(e))
                rem_set.clear()
                break
            
            # Find edge with maximum potential for clique extension
            # Score by number of common neighbors (triangles)
            best_edge = None
            best_score = -1
            edge_list = list(rem_set)
            for e in edge_list:
                u, v = e
                common = 0
                for w in range(k):
                    if w == u or w == v:
                        continue
                    eu = (min(u,w), max(u,w))
                    ev = (min(v,w), max(v,w))
                    if eu in rem_set and ev in rem_set:
                        common += 1
                if common > best_score:
                    best_score = common
                    best_edge = e
            
            if best_edge is None:
                best_edge = edge_list[0]
            
            u, v = best_edge
            current_clique = {u, v}
            
            # Greedy extension: add vertices connected to all current clique members
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w in current_clique:
                        continue
                    can_add = True
                    for x in current_clique:
                        e = (min(w, x), max(w, x))
                        if e not in rem_set:
                            can_add = False
                            break
                    if can_add:
                        current_clique.add(w)
                        changed = True
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    rem_set.discard((min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j])))
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = [[int(x) for x in clique] for clique in res]
        results.append(cleaned)
        
    return results
