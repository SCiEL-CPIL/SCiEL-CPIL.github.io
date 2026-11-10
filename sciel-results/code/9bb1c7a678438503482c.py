"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    deadline = time.time() + 140.0  # Safety margin before the 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build the exact same split graph as the seed generator.
        # Clique on first k vertices, independent set on the rest.
        # Cross pairs joined with probability p.
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def bron_kerbosch_max_clique(R, P, X, adj, max_size):
        """
        Finds a maximum clique in the graph defined by adj (adjacency sets)
        using Bron-Kerbosch with pivot.
        R: current clique (list)
        P: candidates (set of ints)
        X: excluded (set of ints)
        adj: dict of int -> set of ints (neighbors)
        max_size: global best size found so far
        Returns (best_clique, best_size)
        """
        best_clique = []
        best_size = max_size
        
        # If P and X are both empty, R is a maximal clique
        if not P and not X:
            if len(R) > best_size:
                best_clique = R[:]
                best_size = len(R)
            return best_clique, best_size

        # Choose a pivot u from P union X to minimize branching
        # Heuristic: pick u with max degree in P
        pivot_candidates = P.union(X)
        if not pivot_candidates:
            return best_clique, best_size
            
        u = None
        max_deg = -1
        for v in pivot_candidates:
            deg = len(adj.get(v, set()) & P)
            if deg > max_deg:
                max_deg = deg
                u = v
                
        # Branch on P \ N(u)
        # If u is not in P, P \ N(u) is just P
        # If u is in P, P \ N(u) is P minus neighbors of u in P
        if u in P:
            to_branch = P.difference(adj.get(u, set()))
        else:
            to_branch = P.copy()
            
        for v in list(to_branch):
            new_R = R + [v]
            new_P = P.intersection(adj.get(v, set()))
            new_X = X.intersection(adj.get(v, set()))
            
            # Prune if even adding all of new_P to new_R won't beat best_size
            if len(new_R) + len(new_P) <= best_size:
                continue
                
            clique, size = bron_kerbosch_max_clique(new_R, new_P, new_X, adj, best_size)
            if size > best_size:
                best_clique = clique
                best_size = size
                
            P.remove(v)
            X.add(v)
            
        return best_clique, best_size

    def find_max_clique_in_subset(vertices, available_edges):
        """
        Finds the maximum clique in the subgraph induced by 'vertices' 
        using only edges in 'available_edges'.
        Returns a list of vertices forming the max clique.
        """
        if not vertices:
            return []
            
        # Build adjacency list for the subgraph
        adj = {v: set() for v in vertices}
        v_list = list(vertices)
        for i in range(len(v_list)):
            for j in range(i + 1, len(v_list)):
                u, w = v_list[i], v_list[j]
                e = (min(u, w), max(u, w))
                if e in available_edges:
                    adj[u].add(w)
                    adj[w].add(u)
                    
        # Run Bron-Kerbosch
        # Initial P is all vertices, X is empty
        P = set(vertices)
        X = set()
        R = []
        
        best_clique, _ = bron_kerbosch_max_clique(R, P, X, adj, 0)
        
        return best_clique

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Available C-edges (edges within the clique part that need to be covered)
        # Using a set of tuples (min, max) for O(1) lookup
        available = set()
        for u in range(k):
            for w in range(u + 1, k):
                available.add((u, w))
                
        mixed_cliques = []
        
        # Sort I vertices by degree descending to handle "harder" ones first
        I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=True)
        
        for v in I_vertices:
            if not N[v]:
                continue
                
            remaining_neighbors = set(N[v])
            
            while remaining_neighbors:
                if time.time() > deadline:
                    # Fallback: just add 2-cliques for remaining
                    for u in remaining_neighbors:
                        mixed_cliques.append([int(v), int(u)])
                    break
                
                # Find a maximum clique in the graph induced by remaining_neighbors
                # using only edges in 'available'.
                # This maximizes the number of C-edges covered per mixed clique.
                
                S = find_max_clique_in_subset(remaining_neighbors, available)
                
                if S:
                    # The clique is [v] + S
                    # This covers cross-edges (v, u) for all u in S
                    # And clique-edges (u, w) for all u, w in S
                    mixed_cliques.append([int(v)] + [int(u) for u in S])
                    
                    # Remove the clique-edges from available
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            available.discard(e)
                    
                    # Remove S from remaining_neighbors
                    remaining_neighbors.difference_update(S)
                else:
                    # If S is empty, it means no edges available? 
                    # But remaining_neighbors is not empty.
                    # If no edges in available connect any two vertices in remaining_neighbors,
                    # then the max clique size is 1.
                    # Bron-Kerbosch should return a clique of size 1 if there are no edges.
                    # Let's ensure we handle size 1.
                    # If find_max_clique_in_subset returns empty, it might be due to implementation detail.
                    # Let's fallback to picking one vertex.
                    u = next(iter(remaining_neighbors))
                    S = [u]
                    mixed_cliques.append([int(v), int(u)])
                    remaining_neighbors.discard(u)

        # Now cover remaining available edges with pure cliques (within C)
        pure_cliques = []
        rem_edges = set(available)
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: add edges as 2-cliques
                for e in rem_edges:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break
                
            # Find vertices involved in rem_edges
            vertices = set()
            for u, w in rem_edges:
                vertices.add(u)
                vertices.add(w)
                
            # Find maximum clique in this subgraph
            S = find_max_clique_in_subset(vertices, rem_edges)
            
            if S:
                pure_cliques.append([int(u) for u in S])
                # Remove edges in S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                # Should not happen if rem_edges is not empty, but just in case
                break

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback: just use the big clique + all cross edges as 2-cliques
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
        else:
            result.append(solve_instance(n, k, p, seed))

    # Ensure all types are Python int and lists
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
# EVOLVE-BLOCK-END
