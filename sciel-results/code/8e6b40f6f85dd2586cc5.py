import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

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
        
        # Pre-compute for each pair (u, w) in K, the list of v in I connected to both
        # This is O(k^2 * (n-k)) which is fine for k<=30, n<=90
        common_neighbors = {}
        for u in range(k):
            for w in range(u + 1, k):
                common_neighbors[(u, w)] = []
        
        # Build reverse index: for each v, for each pair in neighbors[v], add v to common_neighbors
        # Actually, let's just iterate v and update common_neighbors for all pairs in N_v
        for v in range(k, n):
            N_v = neighbors[v]
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    pair = (min(u, w), max(u, w))
                    common_neighbors[pair].append(v)
        
        cliques = []
        
        # Step 1: Maximize triangles (3-cliques)
        # A triangle is {u, w, v} where u, w in K, v in I, and (u,w), (u,v), (w,v) are edges.
        # Each such triangle covers 1 K-edge and 2 cross-edges.
        # We want to select as many such triangles as possible, disjoint in terms of edges.
        # This is a 3-dimensional matching problem, which is hard.
        # However, we can use a greedy approach:
        # For each K-edge (u, w), if it has common neighbors, pick one v.
        # To maximize triangles, we should prioritize K-edges with many common neighbors?
        # Or just iterate and pick the first available?
        # Let's try a greedy strategy:
        # 1. Sort K-edges by number of common neighbors (descending).
        # 2. For each K-edge, if not yet covered, pick a common neighbor v that is not "used up" in a way that prevents other triangles?
        #    Actually, a vertex v can be in multiple triangles as long as the K-edges and cross-edges are distinct.
        #    But each cross-edge (u, v) can only be used once.
        #    So if we pick triangle (u, w, v), we consume (u, v) and (w, v).
        #    So v's capacity is limited by its degree in K? No, by the number of cross-edges incident to it.
        #    Each v has degree d_v in cross-edges. It can participate in at most d_v // 2 triangles.
        
        # Let's use a greedy matching-like approach on the K-edges.
        # available_k_edges: set of K-edges not yet covered.
        available_k_edges = set((u, w) for u in range(k) for w in range(u + 1, k))
        
        # Sort K-edges by number of common neighbors
        k_edges_sorted = sorted(available_k_edges, key=lambda e: len(common_neighbors[e]), reverse=True)
        
        # Track which cross-edges are used
        used_cross_edges = set()
        
        for (u, w) in k_edges_sorted:
            if (u, w) not in available_k_edges:
                continue
            candidates = common_neighbors[(u, w)]
            # Find a candidate v such that (u, v) and (w, v) are not in used_cross_edges
            found = False
            for v in candidates:
                e1 = (u, v)
                e2 = (w, v)
                if e1 not in used_cross_edges and e2 not in used_cross_edges:
                    # Use this triangle
                    cliques.append([u, w, v])
                    used_cross_edges.add(e1)
                    used_cross_edges.add(e2)
                    available_k_edges.remove((u, w))
                    found = True
                    break
            # If no candidate found, we leave (u, w) for later (or cover as 2-clique)
        
        # Step 2: Cover remaining cross-edges with 2-cliques
        # Iterate all v in I
        for v in range(k, n):
            N_v = neighbors[v]
            for u in N_v:
                e = (u, v)
                if e not in used_cross_edges:
                    cliques.append([u, v])
                    used_cross_edges.add(e)
        
        # Step 3: Cover remaining K-edges
        # Greedy clique cover on the remaining subgraph of K
        remaining_k_edges = list(available_k_edges)
        
        while remaining_k_edges:
            if time.time() > deadline:
                # Fallback: cover each remaining edge as a 2-clique
                for e in remaining_k_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximal clique in the remaining graph
            # Pick the first edge
            u, w = remaining_k_edges[0]
            
            # Find common neighbors of u and w in the remaining graph
            # A node x is a candidate if (u, x) and (w, x) are in remaining_k_edges
            candidates = set()
            for e in remaining_k_edges:
                a, b = e
                if a == u and b != w:
                    candidates.add(b)
                elif b == u and a != w:
                    candidates.add(a)
                if a == w and b != u:
                    candidates.add(b)
                elif b == w and a != u:
                    candidates.add(a)
            
            valid_c = []
            for c in candidates:
                e_uc = (min(u, c), max(u, c))
                e_wc = (min(w, c), max(w, c))
                if e_uc in remaining_k_edges and e_wc in remaining_k_edges:
                    valid_c.append(c)
            
            if valid_c:
                c = valid_c[0]
                clique = [u, w, c]
                # Remove edges covered by this clique
                edges_to_remove = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        edges_to_remove.add((min(a, b), max(a, b)))
                remaining_k_edges = [e for e in remaining_k_edges if e not in edges_to_remove]
                cliques.append(clique)
            else:
                cliques.append([u, w])
                remaining_k_edges.remove((u, w))
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Emergency: return empty lists? No, must be valid.
            # But if we are this late, we probably already have results or should have finished.
            # Let's just break and return what we have? No, must return 3 lists.
            # If we haven't processed an instance, we can't really do much.
            # But the algorithm is fast.
            pass
        cliques = solve_instance(*spec)
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results
