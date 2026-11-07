import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Precompute edges in K
        K_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                K_edges.add((u, w))
        
        # For each v in I, compute N(v) and E_v (edges within N(v))
        I_vertices = list(range(k, n))
        N_v = {v: set(neighbors[v]) for v in I_vertices}
        E_v = {}
        for v in I_vertices:
            edges = set()
            nv = list(N_v[v])
            for i in range(len(nv)):
                for j in range(i + 1, len(nv)):
                    u, w = nv[i], nv[j]
                    edges.add((min(u, w), max(u, w)))
            E_v[v] = edges
            
        # Build conflict graph
        # v, v' conflict if E_v & E_v' != empty
        # We want to select an independent set to maximize benefit.
        # Benefit approx: |N(v)| - 1 (cliques saved)
        # We will use a greedy approach with multiple restarts.
        
        best_cliques = None
        best_count = float('inf')
        
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Randomize order
            order = I_vertices[:]
            random.Random(seed + r).shuffle(order)
            
            selected_large = []
            covered_edges = set()
            
            for v in order:
                # Check if v conflicts with selected_large
                conflict = False
                for v_sel in selected_large:
                    if E_v[v] & E_v[v_sel]:
                        conflict = True
                        break
                if not conflict:
                    selected_large.append(v)
                    covered_edges.update(E_v[v])
            
            # Construct cliques
            cliques = []
            
            # Large cliques
            for v in selected_large:
                cliques.append(list(N_v[v]) + [v])
                
            # Singletons for others
            for v in I_vertices:
                if v not in selected_large:
                    for u in N_v[v]:
                        cliques.append([u, v])
                        
            # Cover remaining K edges
            remaining_edges = K_edges - covered_edges
            
            # Greedy clique cover for remaining_edges
            rem_edges_list = list(remaining_edges)
            while rem_edges_list:
                if time.time() > deadline:
                    # Fallback
                    for e in rem_edges_list:
                        cliques.append(list(e))
                    break
                
                # Pick an edge
                # Heuristic: pick edge with vertices having high degree in remaining graph
                # For simplicity, just pick first or random
                # Let's pick the one that is in the most cliques?
                # Or just the first one.
                u, v = rem_edges_list[0]
                
                # Find max clique containing (u, v) in the graph defined by rem_edges_list
                # Candidates are neighbors of u and v in the remaining graph
                adj_u = set()
                adj_v = set()
                for e in rem_edges_list:
                    a, b = e
                    if a == u: adj_u.add(b)
                    if b == u: adj_u.add(a)
                    if a == v: adj_v.add(b)
                    if b == v: adj_v.add(a)
                    
                candidates = adj_u & adj_v
                # Ensure candidates form a clique with u and v
                valid_cands = []
                for c in candidates:
                    ok = True
                    for x in valid_cands:
                        if (min(c, x), max(c, x)) not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        valid_cands.append(c)
                        
                clique = [u, v] + valid_cands
                cliques.append(clique)
                
                # Remove edges covered by this clique
                edges_to_remove = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        edges_to_remove.add((min(a, b), max(a, b)))
                        
                rem_edges_list = [e for e in rem_edges_list if e not in edges_to_remove]
                
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
                
        if best_cliques is None:
            # Fallback if no restarts finished
            cliques = []
            for v in I_vertices:
                for u in N_v[v]:
                    cliques.append([u, v])
            for e in K_edges:
                cliques.append(list(e))
            best_cliques = cliques
            
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Should not happen
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
