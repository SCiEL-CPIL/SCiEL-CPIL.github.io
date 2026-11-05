import time
import sys
from itertools import combinations
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        if not edges_set:
            return []
        
        # Uncovered edges
        uncovered = set(edges_set)
        
        # Helper to check if edge is in uncovered
        def in_unc(u, v):
            if u == v: return False
            a, b = (u, v) if u < v else (v, u)
            return (a, b) in uncovered

        # Helper to check if a set of vertices forms a clique in the UNCOVERED graph
        def is_clique_in_unc(vertices):
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if not in_unc(vertices[i], vertices[j]):
                        return False
            return True

        # Helper to compute edges covered by a clique
        def clique_edges(vertices):
            es = []
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in uncovered:
                        es.append((a, b))
            return es

        # Strategy: Global Greedy Set Cover
        # 1. Generate a candidate pool of cliques.
        # 2. Iteratively pick the clique with the best "efficiency" (new_edges_covered / 1).
        #    Since all cliques cost 1 "unit" in the objective (minimize count), 
        #    we actually want to maximize the number of NEW edges covered by each selected clique.
        
        # To speed up candidate generation and selection:
        # We will maintain a list of candidate cliques.
        # Initially, generate candidates based on edges.
        
        # Optimization: 
        # Generating "maximal" cliques for every edge is expensive.
        # Instead, we can generate a set of "seed" cliques and then refine or just use them.
        
        # Let's try a "Batch" approach:
        # 1. While uncovered:
        #    a. Generate a batch of candidate cliques from current uncovered edges.
        #    b. Select the best one (max new edges covered).
        #    c. Remove its edges.
        
        # To make candidate generation fast:
        # For each edge (u,v) in uncovered, try to extend it to a clique using a greedy method.
        # We limit the number of edges we process in each batch to keep time down.
        
        # However, a better structural change:
        # Pre-compute a list of "potential" cliques from the ORIGINAL graph?
        # No, edges get removed.
        
        # Let's stick to the loop but improve the selection criterion.
        # Current: Max size.
        # New: Max new edges covered.
        
        # In dense graphs, size ~ new edges covered.
        # But as graph gets sparse (uncovered set shrinks), a large clique in original graph
        # might have many edges already covered.
        # So "new edges covered" is the correct metric.
        
        # To find good candidates efficiently:
        # We can sample edges.
        
        cliques = []
        
        # We'll keep a list of "active" candidate cliques? No, just regenerate.
        # But regenerating from scratch is O(E * n^2).
        
        # Let's try to cache some structure?
        # Dynamic adjacency list for uncovered graph.
        
        # Initialize adjacency
        adj = [set() for _ in range(n)]
        for u, v in uncovered:
            adj[u].add(v)
            adj[v].add(u)
            
        def remove_edges_from_adj(clique):
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if v in adj[u]:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        uncovered.discard((min(u,v), max(u,v)))

        while uncovered:
            if time.time() > DEADLINE:
                # Fallback
                for u, v in list(uncovered):
                    cliques.append([u, v])
                    remove_edges_from_adj([u, v])
                break
            
            best_clique = None
            best_new_edges = 0
            best_efficiency = 0 # Not needed if we just maximize new edges
            
            # Candidate generation
            # We don't want to iterate ALL edges if it's too slow.
            # But n=64, E~2000. 
            # Let's iterate a subset or all?
            # Let's iterate all edges in uncovered to find the best candidate.
            
            # To speed up "find clique extending (u,v)":
            # Candidates for extension must be in adj[u] INTERSECT adj[v].
            
            for u, v in list(uncovered):
                # If time is tight, break early?
                # Let's add a time check inside the loop if it gets too slow.
                # But 2000 edges is small.
                
                # Common neighbors in uncovered graph
                common = adj[u].intersection(adj[v])
                
                if not common:
                    # Clique is just [u, v]
                    # New edges covered: 1 (just u-v)
                    # But we only care if it's better than current best.
                    # If best_new_edges is 0, this is a candidate.
                    if best_new_edges < 1:
                        best_new_edges = 1
                        best_clique = [u, v]
                    continue
                
                # Greedy extend from [u, v]
                # Start with S = {u, v}
                # Candidates: common
                # We want to add vertices from common to S if they connect to all in S.
                
                # Sort candidates by degree in the induced subgraph of common?
                # Or just by degree in the whole graph?
                # Let's sort by degree in `common` to find dense clusters.
                
                # Compute degrees within common
                # This is O(|common|^2). |common| <= n-2 = 62. 62^2 = 3844.
                # For 2000 edges, 2000 * 3844 = 7.6M ops.
                # This is feasible.
                
                cand_list = list(common)
                
                # Optimization: If len(cand_list) is small, just check all.
                # Sort by degree in the subgraph induced by cand_list
                deg_in_cand = {w: 0 for w in cand_list}
                cand_set = set(cand_list)
                for w1 in cand_list:
                    for w2 in cand_list:
                        if w1 != w2 and w2 in adj[w1]:
                            deg_in_cand[w1] += 1
                            
                sorted_cands = sorted(cand_list, key=lambda x: deg_in_cand[x], reverse=True)
                
                current_clique = [u, v]
                # We need to check if w connects to ALL vertices in current_clique.
                # current_clique starts with u, v.
                # w is in common, so it connects to u and v.
                # As we add vertices to current_clique, we must check against them.
                
                for w in sorted_cands:
                    # Check if w connects to all in current_clique
                    ok = True
                    for x in current_clique:
                        if x == w: continue
                        if w not in adj[x]:
                            ok = False
                            break
                    if ok:
                        current_clique.append(w)
                
                # Now current_clique is a clique in the uncovered graph.
                # Calculate new edges covered.
                # Since it's a clique in the uncovered graph, ALL its edges are in uncovered.
                # So new_edges = len(current_clique) * (len(current_clique) - 1) / 2
                size = len(current_clique)
                new_edges = size * (size - 1) // 2
                
                if new_edges > best_new_edges:
                    best_new_edges = new_edges
                    best_clique = current_clique
            
            if best_clique is None:
                # Should not happen
                u, v = next(iter(uncovered))
                best_clique = [u, v]
            
            cliques.append(best_clique)
            remove_edges_from_adj(best_clique)
            
        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
