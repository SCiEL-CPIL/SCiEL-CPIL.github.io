import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 140  # 10s safety margin
        
        def random_graph(n, p, seed):
            rng = random.Random(seed)
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        edges.append((u, v))
            return edges

        def solve_instance(n, p, seed):
            edges = random_graph(n, p, seed)
            if not edges:
                return []
            
            # Adjacency list for fast neighbor lookup
            adj = {u: set() for u in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            remaining_edges = set(edges)
            partition = []
            
            while remaining_edges:
                if time.time() > deadline:
                    break
                
                # Find a large clique in the graph formed by remaining_edges
                # We can use Bron-Kerbosch to find max clique
                # But we need to restrict it to edges in remaining_edges
                
                # Helper to check if a set of vertices forms a clique using only remaining edges
                def is_clique(vertices):
                    v_list = list(vertices)
                    for i in range(len(v_list)):
                        for j in range(i+1, len(v_list)):
                            if (v_list[i], v_list[j]) not in remaining_edges and (v_list[j], v_list[i]) not in remaining_edges:
                                return False
                    return True

                # Greedy Max Clique Heuristic
                # 1. Get vertices involved in remaining edges
                active_vertices = set()
                for u, v in remaining_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                
                if not active_vertices:
                    break
                
                # Simple Greedy: Pick a vertex, then its neighbors, etc.
                # Or use a specific Max Clique algorithm on the subgraph.
                
                # Let's implement a quick Bron-Kerbosch for Max Clique on the subgraph defined by remaining_edges
                # Since n is small (32, 64), this is feasible.
                
                # Subgraph adjacency based on remaining_edges
                sub_adj = {u: set() for u in active_vertices}
                for u, v in remaining_edges:
                    if u in sub_adj and v in sub_adj:
                        sub_adj[u].add(v)
                        sub_adj[v].add(u)
                
                max_clique = find_max_clique_bk(list(active_vertices), sub_adj)
                
                if not max_clique:
                    # Fallback: pick any edge
                    u, v = next(iter(remaining_edges))
                    max_clique = [u, v]
                
                # Add clique to partition
                partition.append(max_clique)
                
                # Remove edges of this clique from remaining_edges
                clique_set = set(max_clique)
                for u in max_clique:
                    for v in max_clique:
                        if u < v:
                            remaining_edges.discard((u, v))
                        elif u > v:
                            remaining_edges.discard((v, u))
                        
            return partition

        def find_max_clique_bk(vertices, adj):
            # Bron-Kerbosch with pivot to find MAXIMUM clique
            best_clique = []
            
            def bk(R, P, X):
                nonlocal best_clique
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = R.copy()
                    return
                
                if not P:
                    return
                    
                # Choose pivot u from P union X
                # Maximize |N(u) ∩ P|
                pivot = None
                max_count = -1
                for u in P:
                    count = len(adj[u] & P)
                    if count > max_count:
                        max_count = count
                        pivot = u
                for u in X:
                    count = len(adj[u] & P)
                    if count > max_count:
                        max_count = count
                        pivot = u
                        
                if pivot is None:
                    return
                    
                # Recurse on P \ N(pivot)
                # If pivot is in P, N(pivot) is adj[pivot]
                # We iterate over P \ N(pivot)
                for v in list(P - adj[pivot]):
                    bk(R + [v], P & adj[v], X & adj[v])
                    P.remove(v)
                    X.add(v)
                    
            bk([], set(vertices), set())
            return best_clique

        result1 = solve_instance(32, 0.6, 601)
        result2 = solve_instance(64, 0.55, 602)
        
        return [result1, result2]
