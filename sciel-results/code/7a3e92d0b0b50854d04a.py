"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

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
        # Build neighborhood for each vertex in I (independent set)
        N_C = [set() for _ in range(n)]
        for u, v in edges_ci:
            N_C[v].add(u)
            
        # Identify fixed cliques for non-isolated vertices in I
        # K_v = {v} U N_C(v)
        # These are forced.
        
        # Determine which C-C edges are covered by these fixed cliques.
        # An edge (u, w) in C is covered if there exists v in I such that u, w in N_C(v).
        covered_cc = [[False]*k for _ in range(k)]
        
        # For each v in I, mark all pairs in N_C(v) as covered.
        for v in range(k, n):
            if N_C[v]:
                verts = list(N_C[v])
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        covered_cc[verts[i]][verts[j]] = True
                        covered_cc[verts[j]][verts[i]] = True
                        
        # E_rem is the set of C-C edges not covered by fixed cliques.
        # We need to partition E_rem into cliques (subsets of C).
        # This is equivalent to finding a minimum edge clique cover of the graph G_rem = (C, E_rem).
        
        # Since C is a complete graph K_k, any subset is a clique in the original graph.
        # The condition is that the cliques we pick must only use edges from E_rem.
        # This means that if we pick a clique S, all pairs in S must be in E_rem.
        # So S must be a clique in G_rem.
        # And the sets of edges induced by our chosen cliques must partition E_rem.
        
        # Note: If we pick a clique S, it covers ALL edges in E_rem that are within S.
        # We cannot cover an edge twice.
        # So we are looking for a collection of cliques in G_rem such that their induced edge sets form a partition of E(G_rem).
        
        # Let's model this as a set cover / packing problem?
        # No, it's a partition.
        # Since any subset of vertices in G_rem that is a clique covers a specific set of edges.
        # We want to pick a set of such cliques such that every edge in E_rem is covered exactly once.
        
        # Observation:
        # In G_rem, if we pick a maximal clique S, it covers all edges in S.
        # Are there any edges in S that are NOT in E_rem? No, by definition S is a clique in G_rem, so all edges in S are in E_rem.
        # So picking a clique S in G_rem covers exactly E(S) ∩ E_rem, which is just E(S).
        
        # So the problem reduces to: Partition the edge set E(G_rem) into edge sets induced by cliques in G_rem.
        # This is the "Edge Clique Cover" of G_rem.
        # Wait, standard edge clique cover usually allows covering edges with cliques, not necessarily a partition (overlap allowed? No, partition means disjoint).
        # "Every edge must lie in exactly one listed clique" -> Disjoint edge sets.
        
        # Since G_rem is a subgraph of a complete graph, this is a specific problem.
        # Let's analyze the structure of G_rem.
        # E_rem contains (u, w) if u, w are NOT both in N_C(v) for ANY v in I.
        # Let S_v = N_C(v).
        # (u, w) in E_rem iff for all v, NOT (u in S_v AND w in S_v).
        # This means the set of vertices {u, w} is NOT a subset of any S_v.
        
        # Let's try to solve the edge clique cover of G_rem using a greedy algorithm or local search.
        # Given k is small (<=30), we can afford some computation.
        
        # Heuristic:
        # 1. Find a maximal clique in G_rem.
        # 2. Add it to the solution.
        # 3. Remove its edges from G_rem.
        # 4. Repeat.
        
        # To improve, we can try multiple maximal cliques and pick the best (e.g., largest), or use local search.
        # Since we have a time budget, we can try a simple local search on the set of cliques.
        
        # Let's start with a greedy approach.
        # We represent G_rem with an adjacency matrix or set of edges.
        # edges_rem = set((u, w) for u < w if not covered_cc[u][w])
        
        edges_rem = set()
        for u in range(k):
            for w in range(u+1, k):
                if not covered_cc[u][w]:
                    edges_rem.add((u, w))
                    
        if not edges_rem:
            # All C-C edges covered by fixed cliques.
            fixed_cliques = []
            for v in range(k, n):
                if N_C[v]:
                    fixed_cliques.append([v] + list(N_C[v]))
            return fixed_cliques
            
        # Greedy Edge Clique Cover
        # We want to cover edges_rem with cliques in G_rem.
        
        # Helper to get neighbors in current G_rem
        # We'll maintain a set of edges and update it.
        
        def get_clique_from_edge(u, w, current_edges):
            # Find a clique in G_rem containing (u, w)
            # Start with {u, w}
            # Candidates: vertices x such that (u, x) and (w, x) are in current_edges.
            # We want to extend this to a maximal clique.
            
            # To maximize the number of edges covered, we want a large clique.
            # But a large clique covers a specific set of edges.
            # Actually, any clique works.
            # Let's try to find a MAXIMAL clique containing (u, w).
            
            # Initialize candidate set
            # Candidates must be adjacent to both u and w in current_edges.
            candidates = set()
            for x in range(k):
                if x == u or x == w:
                    continue
                e_ux = (u, x) if u < x else (x, u)
                e_wx = (w, x) if w < x else (x, w)
                if e_ux in current_edges and e_wx in current_edges:
                    candidates.add(x)
                    
            # Greedy expansion
            current_clique = {u, w}
            # We can pick candidates in an order to maximize clique size?
            # Let's just pick arbitrary or by degree.
            # To make it deterministic and fast, pick by index or random.
            # Random might help escape local optima if we restart, but let's stick to greedy for now.
            
            # Sort candidates by degree in the induced subgraph of candidates?
            # Or just iterate.
            
            # Let's try to build a maximal clique.
            # While candidates:
            #   Pick x with max degree in candidates (induced by current_edges)
            #   Add x to current_clique
            #   Remove from candidates any vertex not connected to x
            #   Update degrees
            
            # This is O(k^3) per clique.
            
            # Let's implement a simple version.
            avail = list(candidates)
            while avail:
                # Find best node
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
                
                # Filter avail
                new_avail = []
                for node in avail:
                    if node == best_node:
                        continue
                    e = (node, best_node) if node < best_node else (best_node, node)
                    if e in current_edges:
                        new_avail.append(node)
                avail = new_avail
                
            return list(current_clique)

        # Main greedy loop
        # To improve, we can choose the "hardest" edge first (one with fewest clique extensions).
        # Or just iterate through edges.
        
        # Let's iterate through edges in a fixed order, but pick the one that yields the largest clique?
        # That might be expensive to compute for all edges.
        
        # Alternative: Randomized greedy with restarts?
        # Given time, let's try a single greedy pass first.
        
        current_edges = set(edges_rem)
        cliques = []
        
        # We process edges in a specific order.
        # Let's sort edges by some heuristic?
        # For now, arbitrary order (u, w).
        # We need to ensure we process each edge exactly once.
        # We can just pick an edge, form a clique, remove its edges.
        
        # To make it deterministic, we pick the lexicographically smallest edge remaining.
        # Or we can try multiple random orders and keep the best result within a time limit.
        
        # Let's implement a few random restarts of the greedy algorithm.
        
        best_solution = None
        best_count = float('inf')
        
        # Number of restarts depends on time.
        # Let's say 100 restarts if fast enough.
        
        max_restarts = 500 # Adjust based on k and time
        # If k is small, we can do more.
        
        for _ in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            current_edges = set(edges_rem)
            sol_cliques = []
            
            # Shuffle the order of edge processing?
            # We don't have a list of edges to shuffle directly if we remove them.
            # We can shuffle the vertex order or use random edge selection.
            
            # Let's pick a random edge from current_edges at each step.
            # This introduces randomness into the greedy choice.
            
            while current_edges:
                # Pick random edge
                edge = random.choice(list(current_edges))
                u, w = edge
                
                clique = get_clique_from_edge(u, w, current_edges)
                
                # Remove all edges in this clique from current_edges
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (a, b) if a < b else (b, a)
                        if e in current_edges:
                            current_edges.remove(e)
                            
                sol_cliques.append(clique)
                
            # Count total cliques
            # Fixed cliques + these cliques
            num_fixed = sum(1 for v in range(k, n) if N_C[v])
            total = num_fixed + len(sol_cliques)
            
            if total < best_count:
                best_count = total
                best_solution = sol_cliques
                # Early stop if we hit a good number?
                # Theoretical lower bound?
                # If best_count is close to num_fixed + 1, we can stop.
                
        # Combine fixed cliques and best_solution
        final_cliques = []
        for v in range(k, n):
            if N_C[v]:
                final_cliques.append([v] + list(N_C[v]))
                
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
        if time.time() > deadline:
            # Fallback
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        # Use a fixed seed for reproducibility of the random restarts?
        # The instance itself is fixed. The solver uses random.
        # Let's seed the solver's random with a fixed value to be deterministic?
        # Or just use default random.
        # To ensure the same output every time, we should seed it.
        random.seed(42)
        
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
