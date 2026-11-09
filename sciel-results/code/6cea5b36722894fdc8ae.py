"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            break
            
        # Build adjacency
        C = list(range(k))
        I = list(range(k, n))
        
        # Neighbors in C for each v in I
        N = {v: set() for v in I}
        edge_set = set()
        
        # Generate edges
        rng = random.Random(seed)
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edge_set.add((u, v))
        
        # C-I edges
        for v in I:
            for u in C:
                if rng.random() < p:
                    N[v].add(u)
                    edge_set.add((min(u, v), max(u, v)))
                    
        # 1. Find max matching in K_k
        # Just pair (0,1), (2,3), ...
        matching = []
        for i in range(0, k, 2):
            if i + 1 < k:
                matching.append((i, i + 1))
        
        # 2. Assign matching edges to v in I
        assigned_to_v = defaultdict(list)
        used_edges = set()
        
        # Sort I vertices by degree descending to assign high-degree ones first?
        # Or just iterate.
        for v in I:
            if time.time() > deadline:
                break
            for (u, w) in matching:
                if (u, w) not in used_edges and u in N[v] and w in N[v]:
                    assigned_to_v[v].append((u, w))
                    used_edges.add((u, w))
        
        # 3. Construct I-cliques
        cliques = []
        covered_c_edges = set()
        
        for v in I:
            if not N[v]:
                continue
                
            # Get assigned matching edges for v
            v_matched = assigned_to_v.get(v, [])
            matched_vertices = set()
            for (u, w) in v_matched:
                matched_vertices.add(u)
                matched_vertices.add(w)
                covered_c_edges.add((min(u, w), max(u, w)))
                
            # Create cliques for matched pairs
            for (u, w) in v_matched:
                cliques.append([v, u, w])
                
            # Create cliques for remaining neighbors
            for u in N[v]:
                if u not in matched_vertices:
                    cliques.append([v, u])
                    
        # 4. Cover residual C-edges
        residual_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_c_edges:
                    residual_edges.append((u, v))
        
        if residual_edges:
            # Greedy clique cover for residual graph
            # Build adjacency for residual
            res_adj = {u: set() for u in C}
            for (u, v) in residual_edges:
                res_adj[u].add(v)
                res_adj[v].add(u)
                
            remaining_edges = set(residual_edges)
            
            while remaining_edges:
                if time.time() > deadline:
                    # Fallback: add edges as cliques
                    for (u, v) in remaining_edges:
                        cliques.append([u, v])
                    break
                
                # Find max clique in remaining graph
                # Simple greedy: pick vertex with max degree, expand
                # For small k, we can just try to find a large clique
                # Given density, a clique of size k/2 is likely
                
                # Heuristic: Pick a vertex, find its neighbors, check if they form clique
                # Since it's almost complete, just take all vertices?
                # No, missing edges.
                
                # Let's just use pairs if greedy is too slow
                # But for k=30, we can afford a bit more
                # Try to find a clique of size 15
                # Actually, just cover with edges is safe but slow score.
                # Let's try to cover with larger cliques.
                
                # Find a large clique
                # Start with a random vertex
                if not remaining_edges:
                    break
                    
                # Get a vertex from remaining edges
                u, v = next(iter(remaining_edges))
                # Build candidate clique
                clique = {u, v}
                # Try to add more
                # Neighbors of u in remaining graph
                candidates = [x for x in res_adj[u] if (min(u, x), max(u, x)) in remaining_edges]
                # Sort by degree?
                for w in candidates:
                    # Check if w is connected to all in clique
                    ok = True
                    for z in clique:
                        if (min(w, z), max(w, z)) not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        clique.add(w)
                        
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    # Remove edges in clique
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            a, b = sorted((list(clique)[i], list(clique)[j]))
                            if (a, b) in remaining_edges:
                                remaining_edges.remove((a, b))
                else:
                    # Fallback to edge
                    cliques.append([u, v])
                    remaining_edges.remove((u, v))
        
        # Convert to list of lists of ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        results.append(final_cliques)
        
    return results
# EVOLVE-BLOCK-END
