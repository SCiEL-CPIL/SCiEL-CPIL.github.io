"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    def solve_split_graph(n, k, edges_ci):
        """
        Solve the edge clique partition for a split graph.
        
        Strategy:
        1. For each vertex v in I, let S_v = N_C(v). The clique K_v = {v} U S_v is forced 
           (must be in any partition) because the edges (v, u) for u in S_v must be covered, 
           and v is only adjacent to S_v in C, so any clique containing v and an edge (v,u) 
           must be subset of {v} U S_v. Actually, it's even stronger: if we put v in a clique, 
           that clique is {v} U T where T subset S_v and T is a clique in C. But since C is 
           complete, any T subset S_v is a clique. The edges (v,u) for u in S_v must all be 
           in the same clique containing v? No. 
           
           Wait: v is in I. The neighbors of v are S_v in C. 
           The edges incident to v are (v, u) for u in S_v.
           In an edge partition, each edge (v, u) must be in exactly one clique.
           Consider the clique containing edge (v, u_1). It must be of the form {v} U T_1 
           where T_1 subset S_v and u_1 in T_1. 
           If there's another edge (v, u_2), it could be in the same clique if u_2 in T_1, 
           or in a different clique {v} U T_2 with u_2 in T_2, T_1 ∩ T_2 = ∅ (to avoid 
           double covering the vertex? No, vertices can be in multiple cliques. But edges 
           must be partitioned).
           
           However, if we split the edges incident to v into multiple cliques, say 
           K_1 = {v} U T_1 and K_2 = {v} U T_2 with T_1 ∩ T_2 = ∅, 
           we use 2 cliques for v instead of 1. This is strictly worse in terms of clique count.
           The only downside is if splitting allows better coverage of C-C edges.
           But covering C-C edges with cliques in C is independent of how we cover v's edges, 
           EXCEPT that the C-C edges covered by K_1 are E(T_1) and by K_2 are E(T_2).
           The total C-C edges covered by v's cliques is E(T_1) ∪ E(T_2) = E(T_1 ∪ T_2) 
           if T_1 ∩ T_2 = ∅. This is the same as if we had one clique {v} U (T_1 ∪ T_2).
           And using one clique is fewer cliques.
           
           So it's always optimal to use a single clique per v in I: K_v = {v} U S_v.
        """
        # Build neighborhood for each vertex in I
        N_C = [set() for _ in range(n)]
        for u, v in edges_ci:
            N_C[v].add(u)
        
        # Fixed cliques for non-isolated vertices in I
        fixed_cliques = []
        for v in range(k, n):
            if N_C[v]:
                fixed_cliques.append([v] + list(N_C[v]))
        
        # Determine which C-C edges are covered by these fixed cliques.
        covered_cc = [[False]*k for _ in range(k)]
        
        for v in range(k, n):
            if N_C[v]:
                verts = list(N_C[v])
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        covered_cc[verts[i]][verts[j]] = True
                        covered_cc[verts[j]][verts[i]] = True
                        
        # Remaining C-C edges
        edges_rem = set()
        for u in range(k):
            for w in range(u+1, k):
                if not covered_cc[u][w]:
                    edges_rem.add((u, w))
                    
        if not edges_rem:
            return fixed_cliques
        
        # We need to partition edges_rem into cliques in G_rem = (C, edges_rem).
        # This is the minimum edge clique cover of G_rem.
        
        # G_rem is a graph on k vertices.
        # We will use a heuristic to find a good partition.
        
        # Precompute adjacency list for G_rem
        adj_rem = [set() for _ in range(k)]
        for u, w in edges_rem:
            adj_rem[u].add(w)
            adj_rem[w].add(u)
            
        def get_max_clique_containing_edge(u, w, current_edges):
            """Find a maximal clique in current_edges containing edge (u, w)."""
            # Candidates must be adjacent to both u and w in current_edges
            candidates = set()
            for x in range(k):
                if x == u or x == w:
                    continue
                e_ux = (u, x) if u < x else (x, u)
                e_wx = (w, x) if w < x else (x, w)
                if e_ux in current_edges and e_wx in current_edges:
                    candidates.add(x)
                    
            current_clique = {u, w}
            avail = list(candidates)
            
            while avail:
                # Pick vertex with max degree in the induced subgraph on avail
                best_node = -1
                best_deg = -1
                for node in avail:
                    deg = 0
                    for other in avail:
                        if node == other:
                            continue
                        e = (node, other) if node < other else (other, node)
                        if e in current_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_node = node
                        
                if best_node == -1:
                    break
                    
                current_clique.add(best_node)
                
                # Filter avail: keep only neighbors of best_node in current_edges
                new_avail = []
                for node in avail:
                    if node == best_node:
                        continue
                    e = (node, best_node) if node < best_node else (best_node, node)
                    if e in current_edges:
                        new_avail.append(node)
                avail = new_avail
                
            return list(current_clique)

        # Heuristic: Randomized greedy with multiple restarts
        best_solution = None
        best_count = float('inf')
        
        # Number of restarts: adapt based on k and time
        # For k=30, we can afford many restarts
        max_restarts = 200
        if k <= 15:
            max_restarts = 500
        elif k <= 20:
            max_restarts = 300
        elif k <= 25:
            max_restarts = 200
        else:
            max_restarts = 100
            
        rng_solver = Random(12345)
        
        for _ in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            current_edges = set(edges_rem)
            sol_cliques = []
            
            while current_edges:
                # Pick a random edge
                edge = rng_solver.choice(list(current_edges))
                u, w = edge
                
                # Find maximal clique containing this edge
                clique = get_max_clique_containing_edge(u, w, current_edges)
                
                # Remove all edges in this clique from current_edges
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e in current_edges:
                            current_edges.remove(e)
                            
                sol_cliques.append(clique)
                
            total = len(sol_cliques)
            
            if total < best_count:
                best_count = total
                best_solution = sol_cliques
                # If we've found a solution with only a few cliques, we might stop early
                # but let's keep searching a bit more for better solutions
                
        # Combine fixed cliques and best solution
        final_cliques = list(fixed_cliques)
        if best_solution:
            final_cliques.extend(best_solution)
        else:
            # Fallback: cover edges one by one
            for u, w in edges_rem:
                final_cliques.append([u, w])
                
        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback: simple valid solution
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = solve_split_graph(n, k, edges_ci)
        
        # Validate and convert types
        valid_cliques = []
        for c in cliques:
            if len(c) < 2:
                continue
            if len(set(c)) != len(c):
                continue
            valid_cliques.append([int(x) for x in c])
            
        results.append(valid_cliques)
        
    return results
# EVOLVE-BLOCK-END
