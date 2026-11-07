import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def greedy_clique_cover(edges, vertices):
        if not edges:
            return []
        edges = set(edges)
        cliques = []
        while edges:
            deg = {v: 0 for v in vertices}
            for u, v in edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            if not deg:
                break
            start = max(deg, key=lambda x: deg[x])
            clique = {start}
            candidates = sorted([x for x in vertices if x != start], key=lambda x: deg.get(x, 0), reverse=True)
            for c in candidates:
                if all((min(c, x), max(c, x)) in edges for x in clique):
                    clique.add(c)
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for u, v in combinations(clique, 2):
                    e = (min(u, v), max(u, v))
                    edges.discard(e)
            else:
                e = next(iter(edges))
                cliques.append([e[0], e[1]])
                edges.discard(e)
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        all_internal_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_internal = set(all_internal_edges)
        cliques = []
        
        # Process v in I, sorted by degree descending to handle large neighborhoods first
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = neighbors[v]
            if not N_v:
                continue
            
            # Find a maximal clique in N_v using only remaining internal edges
            # This ensures we don't double-cover
            remaining_nodes = set(N_v)
            
            while remaining_nodes:
                # Find maximal clique in remaining_nodes using remaining_internal
                # Start with node of highest degree in remaining_internal induced subgraph
                deg_in_rem = {node: 0 for node in remaining_nodes}
                for node in remaining_nodes:
                    for other in remaining_nodes:
                        if node != other:
                            e = (min(node, other), max(node, other))
                            if e in remaining_internal:
                                deg_in_rem[node] += 1
                
                if not any(d > 0 for d in deg_in_rem.values()):
                    # No internal edges among remaining nodes, so any single node works
                    # But we need at least 2 for a clique with v
                    # So just take one node
                    node = next(iter(remaining_nodes))
                    clique = [node]
                else:
                    start = max(deg_in_rem, key=lambda x: deg_in_rem[x])
                    clique = {start}
                    candidates = sorted([x for x in remaining_nodes if x != start], 
                                        key=lambda x: deg_in_rem.get(x, 0), reverse=True)
                    for c in candidates:
                        if all((min(c, x), max(c, x)) in remaining_internal for x in clique):
                            clique.add(c)
                    clique = list(clique)
                
                if len(clique) >= 1:
                    full_clique = [v] + clique
                    cliques.append(full_clique)
                    
                    # Remove internal edges covered by this clique
                    for u, w in combinations(clique, 2):
                        e = (min(u, w), max(u, w))
                        remaining_internal.discard(e)
                    
                    remaining_nodes -= set(clique)
        
        # Cover remaining internal edges
        if remaining_internal:
            vertices_in_rem = set()
            for u, v in remaining_internal:
                vertices_in_rem.add(u)
                vertices_in_rem.add(v)
            rem_cliques = greedy_clique_cover(remaining_internal, vertices_in_rem)
            cliques.extend(rem_cliques)
        
        clean_cliques = [[int(x) for x in c] for c in cliques]
        return clean_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
