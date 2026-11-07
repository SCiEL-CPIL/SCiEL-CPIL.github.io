import random
import itertools

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

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        K = list(range(k))
        I = list(range(k, n))
        
        # Adjacency lists for the original graph
        # We will maintain a set of remaining edges
        rem_edges = set(edges)
        
        # Helper to get neighbors in K for a vertex v in I, based on rem_edges
        def get_I_neighbors(v):
            neighbors = set()
            for u in K:
                pair = (u, v) if u < v else (v, u)
                if pair in rem_edges:
                    neighbors.add(u)
            return neighbors

        # Helper to check if a set S (subset of K) forms a clique in rem_edges
        def is_clique_in_rem(S):
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    pair = (u, v) if u < v else (v, u)
                    if pair not in rem_edges:
                        return False
            return True

        # Helper to find max clique in a subgraph defined by vertices V and edge set E_sub
        # V is a subset of K
        # E_sub is a set of edges among V that are in rem_edges
        def find_max_clique_in_K(V):
            if len(V) <= 1:
                return list(V)
            
            # Convert V to a list for indexing
            V_list = list(V)
            # Build local adjacency
            local_adj = {u: set() for u in V_list}
            for u in V_list:
                for v in V_list:
                    if u == v: continue
                    pair = (u, v) if u < v else (v, u)
                    if pair in rem_edges:
                        local_adj[u].add(v)
            
            best = []
            
            def expand(R, P, X):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = R.copy()
                    return
                
                all_nodes = P.union(X)
                if not all_nodes:
                    return
                
                # Pivot selection
                max_deg = -1
                pivot = None
                for u in all_nodes:
                    deg = len(P.intersection(local_adj.get(u, set())))
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                
                if pivot is not None and pivot in P:
                    candidates = P - local_adj.get(pivot, set())
                else:
                    candidates = P.copy()
                    
                for u in list(candidates):
                    expand(R.union({u}), P.intersection(local_adj.get(u, set())), X.intersection(local_adj.get(u, set())))
                    P.remove(u)
                    X.add(u)
                    
            expand(set(), set(V_list), set())
            return best

        cliques = []
        
        while rem_edges:
            best_clique = []
            best_size = 0
            
            # Check cliques involving I vertices
            for v in I:
                N_v = get_I_neighbors(v)
                if not N_v:
                    continue
                
                # Find max clique S in N_v such that S is a clique in rem_edges
                S = find_max_clique_in_K(N_v)
                if len(S) >= 1:
                    candidate = [v] + S
                    # Verify candidate is a clique in rem_edges
                    # Edges (v, s) are in rem_edges by construction of N_v
                    # Edges (s1, s2) are in rem_edges by construction of S
                    if len(candidate) > best_size:
                        best_size = len(candidate)
                        best_clique = candidate
            
            # Check cliques involving only K vertices
            # We need to find max clique in the subgraph of K induced by rem_edges
            # Vertices in K that have at least one remaining edge
            active_K = set()
            for u, v in rem_edges:
                if u < k and v < k:
                    active_K.add(u)
                    active_K.add(v)
            
            if active_K:
                S = find_max_clique_in_K(active_K)
                if len(S) > best_size:
                    best_size = len(S)
                    best_clique = S
            
            if not best_clique:
                # Should not happen if rem_edges is not empty
                # Fallback: pick any edge
                u, v = next(iter(rem_edges))
                best_clique = [u, v]
            
            # Add clique
            cliques.append(best_clique)
            
            # Remove edges from rem_edges
            C = best_clique
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    pair = (u, v) if u < v else (v, u)
                    rem_edges.discard(pair)
        
        # Convert to lists of ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
        
    return results
