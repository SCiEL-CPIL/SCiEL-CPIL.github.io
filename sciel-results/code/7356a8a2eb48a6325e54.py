"""Improved program using vertex-based greedy clique extraction."""
import random


def construct():
    def instance(n, p, seed):
        r = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_clique_partition_vertex_based(edges, n):
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        
        remaining = set(edges)
        cliques = []
        
        # Helper to get list of vertices from bitmask
        def bits_to_list(mask):
            return [i for i in range(n) if mask & (1 << i)]
            
        while remaining:
            # Find vertex with max degree in remaining graph
            degrees = [0] * n
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            # Pick vertex with highest degree
            v = max(range(n), key=lambda x: degrees[x])
            
            # Build largest clique containing v using only remaining edges
            clique = [v]
            # Candidates: neighbors of v in remaining graph
            cand_mask = adj[v]
            # We only consider candidates connected to v via remaining edge
            # Since v is in clique, we need edges (v, w) to be in remaining
            # Actually, if (v,w) is not in remaining, we can't add w.
            # But adj[v] contains all original neighbors. We must filter by remaining.
            
            # Efficiently get candidates: neighbors of v that have edge in remaining
            candidates = []
            for w in range(n):
                if w == v: continue
                edge = (min(v, w), max(v, w))
                if edge in remaining:
                    candidates.append(w)
                    
            if not candidates:
                # v is isolated in remaining graph? Should not happen if remaining is not empty
                # and we picked max degree. But if max degree is 0, we break.
                # If remaining is not empty, max degree >= 1.
                pass
            
            current_clique = [v]
            current_set = {v}
            
            # Greedy expansion
            while True:
                # Filter candidates to those connected to ALL vertices in current_clique
                valid_candidates = []
                for w in candidates:
                    if w in current_set:
                        continue
                    ok = True
                    for x in current_clique:
                        edge = (min(w, x), max(w, x))
                        if edge not in remaining:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(w)
                        
                if not valid_candidates:
                    break
                    
                # Pick candidate with max degree in the subgraph of valid_candidates
                # Calculate degrees within valid_candidates
                deg_in_cand = {w: 0 for w in valid_candidates}
                for w in valid_candidates:
                    for u in valid_candidates:
                        if w == u: continue
                        edge = (min(w, u), max(w, u))
                        if edge in remaining:
                            deg_in_cand[w] += 1
                
                best_w = max(valid_candidates, key=lambda w: deg_in_cand[w])
                current_clique.append(best_w)
                current_set.add(best_w)
                
            # Remove all edges within this clique from remaining
            clique_edges = set()
            for i in range(len(current_clique)):
                for j in range(i + 1, len(current_clique)):
                    a, b = current_clique[i], current_clique[j]
                    edge = (min(a, b), max(a, b))
                    if edge in remaining:
                        clique_edges.add(edge)
                        
            if clique_edges:
                remaining -= clique_edges
                cliques.append(current_clique)
            else:
                # Fallback: just take an edge
                e = next(iter(remaining))
                remaining.remove(e)
                cliques.append(list(e))
                
        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = greedy_clique_partition_vertex_based(edges, n)
        result.append(cliques)

    return result
