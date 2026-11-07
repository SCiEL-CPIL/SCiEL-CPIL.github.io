import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges within K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Edges between K and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def max_clique(vertices, adj_matrix):
        # vertices: list of ints
        # adj_matrix: dict of dict or set of sets
        # Returns list of vertices in max clique
        if not vertices:
            return []
        
        # Simple backtracking for max clique
        # adj_matrix[u] is set of neighbors in the current graph
        best = []
        
        def expand(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = R.copy()
                return
            
            # Pivot selection
            # Choose u in P union X with max degree in P
            all_nodes = P.union(X)
            if not all_nodes:
                return
            
            # Heuristic: pick node with max neighbors in P
            max_deg = -1
            pivot = None
            for u in all_nodes:
                deg = len(P.intersection(adj_matrix.get(u, set())))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # Iterate over P not adjacent to pivot
            # If pivot is in P, P \ N(pivot)
            # If pivot is in X, P (since N(pivot) in P is empty if pivot not in P? No, adj_matrix is static for the subgraph? 
            # Wait, adj_matrix should represent the current graph P.
            # Let's simplify: just iterate all in P.
            
            for u in list(P):
                expand(R.union({u}), P.intersection(adj_matrix.get(u, set())), X.intersection(adj_matrix.get(u, set())))
                P.remove(u)
                X.add(u)
                
        initial_P = set(vertices)
        initial_X = set()
        # Build local adj for the subgraph
        local_adj = {u: set() for u in vertices}
        for u in vertices:
            for v in vertices:
                if u != v and v in adj_matrix.get(u, set()):
                    local_adj[u].add(v)
        
        expand(set(), initial_P, initial_X)
        return best

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate K and I
        K = list(range(k))
        I = list(range(k, n))
        
        # Determine N(v) for v in I
        N = {v: set() for v in I}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # available_edges: set of pairs in K
        available_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                available_edges.add((u, v))
                
        cliques = []
        
        # Sort I by degree descending
        I_sorted = sorted(I, key=lambda v: len(N[v]), reverse=True)
        
        # Helper to check if a set of vertices forms a clique in available_edges
        def is_clique_in_available(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) not in available_edges:
                        return False
            return True
            
        # Helper to remove edges of a clique S from available_edges
        def remove_clique_edges(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    if u > v:
                        u, v = v, u
                    available_edges.discard((u, v))
        
        for v in I_sorted:
            targets = set(N[v])
            while targets:
                # Find largest clique in targets that is a clique in available_edges
                # Greedy: try to find a large clique
                # Since k is small, we can try to find max clique in the induced subgraph on 'targets'
                
                # Build adjacency for the subgraph induced by 'targets' based on available_edges
                sub_adj = {u: set() for u in targets}
                for u in targets:
                    for w in targets:
                        if u == w: continue
                        pair = (u, w) if u < w else (w, u)
                        if pair in available_edges:
                            sub_adj[u].add(w)
                            
                # Find Max Clique in this subgraph
                # Using a simple greedy approximation or exact if small
                # For k=30, targets size <= 30.
                # Let's use a simple heuristic: pick vertex with max degree, add to clique, remove non-neighbors, repeat.
                # This finds a maximal clique, not necessarily maximum, but usually good.
                # To be safer, we can try a few random orderings or just the greedy one.
                
                remaining_targets = set(targets)
                current_clique = set()
                
                # Greedy Max Clique
                while remaining_targets:
                    # Pick node with max degree in remaining_targets
                    best_node = None
                    max_deg = -1
                    for u in remaining_targets:
                        deg = len(sub_adj[u].intersection(remaining_targets))
                        if deg > max_deg:
                            max_deg = deg
                            best_node = u
                            
                    if max_deg == 0:
                        # No edges left in remaining_targets
                        break
                        
                    current_clique.add(best_node)
                    # Next candidates must be neighbors of best_node
                    remaining_targets = remaining_targets.intersection(sub_adj[best_node])
                    
                if len(current_clique) >= 2:
                    S = current_clique
                    cliques.append([v] + list(S))
                    remove_clique_edges(S)
                    targets.difference_update(S)
                else:
                    # No clique of size >= 2 found in targets (with available edges)
                    # Take a single vertex
                    u = next(iter(targets))
                    cliques.append([v, u])
                    targets.discard(u)
                    # Note: (v, u) is a cross edge, no K-edge removed.
        
        # Cover remaining edges in K
        while available_edges:
            # Find any edge
            # To minimize count, find largest clique in available_edges
            # Vertices involved in available_edges
            verts_involved = set()
            for u, v in available_edges:
                verts_involved.add(u)
                verts_involved.add(v)
                
            # Build subgraph
            sub_adj = {u: set() for u in verts_involved}
            for u, v in available_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
                
            # Greedy Max Clique
            remaining_verts = set(verts_involved)
            current_clique = set()
            
            while remaining_verts:
                best_node = None
                max_deg = -1
                for u in remaining_verts:
                    deg = len(sub_adj[u].intersection(remaining_verts))
                    if deg > max_deg:
                        max_deg = deg
                        best_node = u
                        
                if max_deg == 0:
                    break
                
                current_clique.add(best_node)
                remaining_verts = remaining_verts.intersection(sub_adj[best_node])
                
            if len(current_clique) >= 2:
                S = current_clique
                cliques.append(list(S))
                remove_clique_edges(S)
            else:
                # Pick any edge
                u, v = next(iter(available_edges))
                cliques.append([u, v])
                available_edges.discard((u, v))
                
        # Convert to lists of ints
        # Ensure types are int
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
        
    return results
