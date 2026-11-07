"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs

        # Step 1: Greedy set packing to select bundled I-vertices
        # For each v in I with |N_C(v)| >= 2, the set of internal edges it covers
        # is all pairs in N_C(v). We want to maximize coverage while keeping sets disjoint.
        
        candidates = []
        for v in range(k, n):
            neigh = neighbors[v]
            if len(neigh) >= 2:
                # Generate all pairs in neigh
                neigh_set = set(neigh)
                edges_v = set()
                neigh_list = list(neigh)
                for i in range(len(neigh_list)):
                    for j in range(i + 1, len(neigh_list)):
                        a, b = neigh_list[i], neigh_list[j]
                        e = (a, b) if a < b else (b, a)
                        edges_v.add(e)
                weight = len(neigh) - 1  # number of internal edges covered
                candidates.append((v, edges_v, weight))
        
        # Sort by weight descending for greedy
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        selected = set()
        covered_internal = set()
        
        for v, edges_v, weight in candidates:
            if not (edges_v & covered_internal):
                selected.add(v)
                covered_internal.update(edges_v)
        
        # Step 2: Build cliques
        cliques = []
        
        # Add bundled cliques
        for v in sorted(selected):
            neigh = neighbors[v]
            clique = list(neigh) + [v]
            cliques.append(clique)
        
        # Add unbundled cross edges as pairs
        for v in range(k, n):
            if v in selected:
                continue
            for u in neighbors[v]:
                cliques.append([u, v])
        
        # Step 3: Cover remaining internal edges with local search
        all_internal = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_internal.add((u, v))
        
        remaining_internal = all_internal - covered_internal
        
        if remaining_internal:
            # Greedy initial clique cover on remaining internal edges
            # Vertices involved
            verts_involved = set()
            for u, v in remaining_internal:
                verts_involved.add(u)
                verts_involved.add(v)
            
            rem_cliques = greedy_clique_cover(list(remaining_internal), verts_involved)
            
            # Local search: try to improve by merging or replacing
            rem_cliques = local_search_clique_cover(rem_cliques, remaining_internal, verts_involved)
            
            cliques.extend(rem_cliques)
        
        return cliques

    def greedy_clique_cover(edges, vertices):
        """Greedy: repeatedly pick an edge, extend to maximal clique in remaining subgraph."""
        if not edges:
            return []
        
        remaining = set(edges)
        cliques = []
        
        while remaining:
            # Pick edge with highest sum of degrees in remaining
            best_edge = None
            best_score = -1
            for e in remaining:
                u, v = e
                deg_u = sum(1 for e2 in remaining if u in e2 and e2 != e)
                deg_v = sum(1 for e2 in remaining if v in e2 and e2 != e)
                score = deg_u + deg_v
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            u, v = best_edge
            # Find all w such that (u,w) and (v,w) are in remaining
            candidates = set()
            for w in vertices:
                if w == u or w == v:
                    continue
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in remaining and e2 in remaining:
                    candidates.add(w)
            
            clique = [u, v] + list(candidates)
            cliques.append(clique)
            
            # Remove all edges covered by this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (a, b) if a < b else (b, a)
                    remaining.discard(e)
        
        return cliques

    def local_search_clique_cover(cliques, edges, vertices):
        """Try to reduce number of cliques by local moves."""
        if not cliques:
            return cliques
        
        edge_set = set(edges)
        current = [list(c) for c in cliques]
        best = list(current)
        best_count = len(current)
        
        # Try merging pairs of cliques if their union is a clique and covers no new edges
        # Actually, we can't merge if union isn't a clique. Instead, try replacing two cliques
        # with one larger clique that covers all edges they covered.
        
        max_iters = 50
        for _ in range(max_iters):
            improved = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    # Check if union of cliques i and j forms a clique
                    union = set(current[i]) | set(current[j])
                    is_clique = True
                    for a in union:
                        for b in union:
                            if a >= b:
                                continue
                            e = (a, b) if a < b else (b, a)
                            if e not in edge_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        # Replace cliques i and j with their union
                        new_cliques = current[:i] + [list(union)] + current[i+1:j] + current[j+1:]
                        if len(new_cliques) < len(best):
                            best = new_cliques
                            current = new_cliques
                            best_count = len(current)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return best

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        result.append(solve_split(*spec))
    
    return result
# EVOLVE-BLOCK-END
