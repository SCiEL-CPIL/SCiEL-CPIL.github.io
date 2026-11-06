import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_maximal_cliques(adj, n, time_limit):
        """
        Find all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of cliques (each as a set of vertices).
        """
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > time_limit:
                return False
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return True
            
            # Pivot selection
            if P or X:
                # Choose pivot u in P ∪ X with maximum degree in P
                # To save time, just pick the first one or a heuristic
                # For dense graphs, degree in P is similar for many vertices
                # Let's pick the vertex in P with max degree in P if P is not empty, else from X
                # Actually, standard BK: u in P ∪ X
                # We want to minimize |P \ N(u)|
                # So we want u with max |P ∩ N(u)|
                
                # Heuristic: pick vertex with highest degree in P
                # If P is empty, pick any from X
                candidates = P if P else X
                if not candidates:
                    return True
                u = max(candidates, key=lambda v: len(P & adj[v]))
                
                # Iterate over P \ N(u)
                candidates_to_process = P - adj[u]
            else:
                return True

            for v in list(candidates_to_process):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                if not bronk(new_R, new_P, new_X):
                    return False
                
                P.remove(v)
                X.add(v)
                
            return True

        bronk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = build_adj(edges, n)
        
        if not edges:
            return []
        
        # Phase 1: Find all maximal cliques
        # Allocate 50% of time to finding cliques
        clique_find_time = time.time() + time_budget * 0.5
        maximal_cliques = find_maximal_cliques(adj, n, clique_find_time)
        
        # Phase 2: Greedy Edge Partition
        # We want to select a set of cliques (subsets of maximal cliques) such that
        # every edge is in exactly one clique.
        # Strategy:
        # 1. Start with all edges uncovered.
        # 2. Repeatedly pick a clique from the remaining edges.
        # 3. To maximize efficiency, we want large cliques.
        # 4. However, we must ensure we don't "waste" edges that could form larger cliques later.
        #    But since we are partitioning, once an edge is used, it's gone.
        #    So greedy: pick the largest possible clique from the remaining edges.
        
        remaining_edges = set(edges)
        result_cliques = []
        
        # Pre-compute adjacency for remaining graph?
        # It changes as we remove edges.
        # For n=32, 64, we can afford to recompute or maintain a set of edges.
        
        # To find the largest clique in the remaining graph, we can use a greedy heuristic
        # or a limited search.
        # Given the time constraint, a greedy approach is safer.
        
        # Heuristic for finding a large clique in remaining graph:
        # 1. Pick a vertex with high degree in remaining graph.
        # 2. Start clique with it.
        # 3. Add neighbors that connect to all in clique.
        # 4. Repeat until no more can be added.
        
        # To improve, we can try multiple starting vertices and pick the best clique found.
        
        # But first, let's just use the maximal cliques we found?
        # No, maximal cliques in the original graph might contain edges that are now removed.
        # We need to find cliques in the *remaining* graph.
        
        # Let's implement a function to find a large clique in a set of edges.
        
        def find_large_clique_in_edges(edge_set, adj_orig, n, time_limit):
            if not edge_set:
                return []
            
            # Build adjacency for the subgraph defined by edge_set
            # This is O(E) which is fine.
            sub_adj = [set() for _ in range(n)]
            for u, v in edge_set:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            best_clique = []
            
            # Try starting from each vertex (or a subset of high-degree vertices)
            # For n=64, trying all 64 is fine.
            for start_v in range(n):
                if not sub_adj[start_v]:
                    continue
                
                current_clique = [start_v]
                candidates = set(sub_adj[start_v])
                
                while candidates:
                    # Pick candidate with max degree in candidates? Or just first?
                    # Heuristic: pick candidate that connects to most other candidates
                    # This is O(|candidates|^2) per step, which might be slow if candidates is large.
                    # For n=64, max candidates is 63. 63^2 is ~4000. 64 starts * 64 steps * 4000 ops = ~16M ops.
                    # This might be too slow if done repeatedly.
                    
                    # Simpler heuristic: pick candidate with max degree in sub_adj
                    # Or just pick the first one.
                    # Let's try picking the one with max degree in current candidates.
                    
                    if time.time() > time_limit:
                        break
                    
                    # Score candidates
                    best_cand = None
                    best_score = -1
                    for c in candidates:
                        # Count how many other candidates c is connected to
                        # And maybe how many vertices in current_clique it's connected to (it is, by definition)
                        score = len(candidates & sub_adj[c])
                        if score > best_score:
                            best_score = score
                            best_cand = c
                    
                    if best_cand is None:
                        break
                        
                    current_clique.append(best_cand)
                    candidates = candidates & sub_adj[best_cand]
                
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique
                    
            return best_clique

        # Main loop
        while remaining_edges:
            if time.time() > time.time() + time_budget * 0.4: # Leave some time for safety
                break
            
            # Find a large clique in the remaining edges
            # We use a time limit for this search
            search_limit = time.time() + 0.5 # 0.5s per clique search?
            # For n=32, this is very fast. For n=64, might be slower.
            # Let's scale it.
            search_limit = time.time() + 0.1
            
            clique = find_large_clique_in_edges(remaining_edges, adj, n, search_limit)
            
            if not clique:
                # Fallback: just pick an edge
                e = next(iter(remaining_edges))
                clique = list(e)
            
            result_cliques.append(clique)
            
            # Remove edges in this clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    remaining_edges.discard((u, v))
        
        return result_cliques

    # Solve both instances
    t1 = time.time()
    budget1 = min(60, DEADLINE - (t1 - start_time) - 10)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(60, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
