"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def greedy_max_clique(adj, n):
        """Find a maximal clique by greedy expansion from each start vertex."""
        best = []
        for start in range(n):
            if not adj[start]:
                continue
            clique = [start]
            for w in sorted(adj[start]):
                if all(w in adj[x] for x in clique):
                    clique.append(w)
            if len(clique) > len(best):
                best = clique
        return best

    def greedy_clique_cover(rem_edges, k, deadline):
        """Greedy set-cover: repeatedly pick clique covering most remaining edges."""
        cliques = []
        remaining = set(rem_edges)
        if not remaining:
            return cliques
        
        # Build adjacency
        adj = [set() for _ in range(k)]
        for (a, b) in remaining:
            adj[a].add(b)
            adj[b].add(a)
        
        while remaining and time.time() < deadline:
            # Find clique covering most remaining edges
            best_clique = None
            best_count = 0
            
            for start in range(k):
                if not adj[start]:
                    continue
                # Greedy expand, prioritizing vertices with high remaining degree
                clique = [start]
                candidates = sorted(adj[start], key=lambda w: len(adj[w]), reverse=True)
                for w in candidates:
                    if all(w in adj[x] for x in clique):
                        clique.append(w)
                
                # Count edges covered
                count = 0
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None or best_count == 0:
                # Fallback: just take any edge
                e = next(iter(remaining))
                best_clique = list(e)
                best_count = 1
            
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining:
                        remaining.discard(e)
                        adj[a].discard(b)
                        adj[b].discard(a)
            
            cliques.append(best_clique)
        
        # If we ran out of time, cover remaining edges individually
        if remaining:
            for (a, b) in remaining:
                cliques.append([a, b])
        
        return cliques

    def clique_partition(n, k, p, seed, deadline):
        edges = instance(n, k, p, seed)
        
        # Build edge set for quick lookup
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))
        
        # Find neighbors in C for each I-vertex
        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh
        
        # Sort I-vertices by degree (descending)
        i_vertices = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
        
        cliques = []
        used_cc_edges = set()  # C-C edges already consumed by Type B cliques
        
        for v in i_vertices:
            if time.time() > deadline - 2:
                # Fallback: individual edges
                for u in N_C[v]:
                    cliques.append([v, u])
                continue
            
            neighbors = list(N_C[v])
            d = len(neighbors)
            
            if d == 0:
                continue
            
            if d == 1:
                u = neighbors[0]
                cliques.append([v, u])
                continue
            
            # Check if all C-C edges within neighbors are available
            conflict_edges = []
            for i in range(d):
                for j in range(i + 1, d):
                    a, b = neighbors[i], neighbors[j]
                    e = (a, b) if a < b else (b, a)
                    if e in used_cc_edges:
                        conflict_edges.append(e)
            
            if not conflict_edges:
                # Use one big clique
                cliques.append([v] + neighbors)
                for i in range(d):
                    for j in range(i + 1, d):
                        a, b = neighbors[i], neighbors[j]
                        e = (a, b) if a < b else (b, a)
                        used_cc_edges.add(e)
            else:
                # Big-then-split: try to find a large subset of neighbors
                # whose C-C edges are all available, then handle the rest
                # Sort neighbors by how many conflict edges they participate in
                conflict_count = {}
                for e in conflict_edges:
                    conflict_count[e[0]] = conflict_count.get(e[0], 0) + 1
                    conflict_count[e[1]] = conflict_count.get(e[1], 0) + 1
                
                # Try greedy: build a clique from neighbors avoiding conflicts
                avail_neighbors = []
                for u in sorted(neighbors, key=lambda x: conflict_count.get(x, 0)):
                    # Check if u can be added without conflict
                    can_add = True
                    for w in avail_neighbors:
                        e = (u, w) if u < w else (w, u)
                        if e in used_cc_edges:
                            can_add = False
                            break
                    if can_add:
                        avail_neighbors.append(u)
                
                if len(avail_neighbors) >= 2:
                    # Use {v} + avail_neighbors as one clique
                    cliques.append([v] + avail_neighbors)
                    for i in range(len(avail_neighbors)):
                        for j in range(i + 1, len(avail_neighbors)):
                            a, b = avail_neighbors[i], avail_neighbors[j]
                            e = (a, b) if a < b else (b, a)
                            used_cc_edges.add(e)
                    # Handle remaining neighbors individually
                    for u in neighbors:
                        if u not in avail_neighbors:
                            cliques.append([v, u])
                else:
                    # Fall back to individual edge cliques
                    for u in neighbors:
                        cliques.append([v, u])
        
        # Cover remaining C-C edges with C-only cliques
        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_cc = all_cc_edges - used_cc_edges
        
        if remaining_cc:
            cc_cliques = greedy_clique_cover(remaining_cc, k, deadline)
            cliques.extend(cc_cliques)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 5:
            # Fallback: just do the simple thing quickly
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for (u, v) in edges:
                if not (u < k and v < k):
                    cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(clique_partition(*spec, deadline))
    
    return results
# EVOLVE-BLOCK-END
