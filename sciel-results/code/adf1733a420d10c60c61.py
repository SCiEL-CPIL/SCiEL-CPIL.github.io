import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

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

    def find_maximal_independent_sets(complement_adj, n, time_limit):
        """
        Find maximal independent sets in the complement graph.
        This is equivalent to finding maximal cliques in the original graph.
        We use a greedy heuristic with some backtracking or just greedy if time is tight.
        For dense original graphs (sparse complement), greedy is often good.
        """
        cliques = []
        # Simple greedy: repeatedly pick vertex with min degree in remaining complement
        # This finds *a* maximal independent set. To get *all* or *many*, we need more.
        # Given time constraints, we will use a limited Bron-Kerbosch on the complement.
        
        def bron_kerbosch_iso(R, P, X, adj, cliques, time_limit):
            if time.time() > time_limit:
                return False
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return True
            
            if not P:
                return True

            # Pivot selection: max degree in P
            pivot = max(P, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                success = bron_kerbosch_iso(new_R, new_P, new_X, adj, cliques, time_limit)
                if not success:
                    return False
                P.remove(v)
                X.add(v)
            return True

        initial_P = set(range(n))
        initial_X = set()
        initial_R = set()
        bron_kerbosch_iso(initial_R, initial_P, initial_X, complement_adj, cliques, time_limit)
        return cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        adj = build_adj(edges, n)
        
        # Build complement adjacency
        complement_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edges:
                    complement_adj[u].add(v)
                    complement_adj[v].add(u)
        
        # Find candidate cliques (maximal independent sets in complement)
        # Allocate 50% of time to finding cliques
        clique_find_time = time_budget * 0.5
        candidate_cliques = find_maximal_independent_sets(complement_adj, n, time.time() + clique_find_time)
        
        # Ensure all edges are covered by candidates.
        # If a maximal clique doesn't cover an edge, we might need smaller cliques.
        # However, any edge is part of at least one maximal clique.
        # So the set of maximal cliques covers all edges.
        
        # If no cliques found (e.g. timeout or trivial), fallback to edges
        if not candidate_cliques:
            return [(u, v) for u, v in edges]
            
        # Deduplicate cliques
        unique_cliques = list(set(tuple(sorted(c)) for c in candidate_cliques))
        
        # If too many cliques, ILP might be too slow or memory intensive.
        # For n=64, E~2000. If K > 5000, it's risky.
        # Let's cap it. If too many, we just use greedy.
        if len(unique_cliques) > 2000:
             # Fallback to greedy
             return greedy_fallback(edges, adj, n, time_budget)

        # Formulate ILP
        # Variables: x_i for each clique i
        # Minimize sum(x_i)
        # Subject to: sum(x_i for i where e in C_i) == 1 for all e in E
        
        edge_list = list(edges)
        num_edges = len(edge_list)
        num_cliques = len(unique_cliques)
        
        # Build sparse matrix A
        # A[e_idx, c_idx] = 1 if edge e is in clique c
        A = lil_matrix((num_edges, num_cliques), dtype=np.int8)
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        for c_idx, clique in enumerate(unique_cliques):
            clique_set = set(clique)
            # Iterate over edges in the clique
            # Since clique is a clique in G, all pairs are edges.
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        A[edge_to_idx[e], c_idx] = 1
        
        A = csr_matrix(A)
        
        # Objective: minimize x
        c = np.ones(num_cliques)
        
        # Constraints: A @ x == 1
        constraints = LinearConstraint(A, 1, 1)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: x is integer
        integrality = np.ones(num_cliques)
        
        # Solve
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0 and res.x is not None:
                # Extract solution
                selected_indices = np.where(res.x > 0.5)[0]
                result = [list(unique_cliques[i]) for i in selected_indices]
                # Verify validity just in case
                if verify_partition(result, edges):
                    return result
        except Exception as e:
            pass
            
        # Fallback
        return greedy_fallback(edges, adj, n, time_budget)

    def greedy_fallback(edges, adj, n, time_budget):
        """
        Greedy fallback: repeatedly pick the largest clique from remaining edges.
        """
        remaining_edges = set(edges)
        result = []
        start = time.time()
        
        while remaining_edges:
            if time.time() > start + time_budget:
                break
            # Find largest clique in the subgraph induced by remaining_edges
            # Heuristic: Pick a random edge, try to grow it.
            # Better: Iterate through vertices, find max clique containing that vertex?
            # Simple: Pick the edge with highest "clique potential" (degree sum).
            
            # To ensure we finish, we just pick edges if we can't find big cliques quickly.
            # Let's try a simple greedy:
            # 1. Find a vertex with max degree in remaining graph.
            # 2. Find max clique in its neighborhood?
            
            # For speed, let's just pick a random edge and grow it greedily.
            e = next(iter(remaining_edges))
            u, v = e
            clique = [u, v]
            
            # Candidates that connect to all in clique
            candidates = set(adj[u]) & set(adj[v])
            candidates -= {u, v}
            
            # Filter candidates to those connected to all current clique members
            # and having edges to them in remaining_edges
            valid_candidates = []
            for w in candidates:
                if all((min(c, w), max(c, w)) in remaining_edges for c in clique):
                    valid_candidates.append(w)
            
            # Sort by degree in remaining graph
            valid_candidates.sort(key=lambda x: sum(1 for y in remaining_edges if x in y), reverse=True)
            
            for w in valid_candidates:
                # Check if adding w is still valid (it is, by construction, but check remaining)
                # Actually, we need to update remaining edges logic carefully.
                # Let's just add it.
                clique.append(w)
                # Update candidates?
                # This is getting complex. Let's simplify:
                # Just add w if it connects to all existing.
                # We already checked that.
                # But adding w might make other candidates invalid.
                # We'll just break after the first valid one? No, try to grow more.
                # Re-evaluate candidates for next step?
                # For simplicity and speed in fallback, let's just take the first valid candidate?
                # No, that's bad.
                
                # Let's restart candidate filtering for each addition.
                # It's O(K^2) but K is small.
                pass
            
            # Actually, the previous greedy in the seed was okay. Let's reuse a simple version.
            # Pick edge (u,v).
            # Candidates = N(u) ∩ N(v) ∩ RemainingGraph.
            # Sort by degree.
            # Add best candidate if it connects to all.
            
            clique = [u, v]
            candidates = (set(adj[u]) & set(adj[v])) - {u, v}
            
            while True:
                # Filter candidates
                valid = []
                for w in candidates:
                    if all((min(c, w), max(c, w)) in remaining_edges for c in clique):
                        # Check if w is connected to all in clique in the ORIGINAL graph
                        # (It is, because it's in N(u)∩N(v) etc, but let's be safe)
                        if all(w in adj[c] for c in clique):
                            valid.append(w)
                
                if not valid:
                    break
                    
                # Pick best
                best = max(valid, key=lambda w: sum(1 for e in remaining_edges if w in e))
                clique.append(best)
                candidates = (candidates - {best})
                
            # Add clique to result
            result.append(clique)
            
            # Remove edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    remaining_edges.discard(e)
                    
        return result

    def verify_partition(cliques, edges):
        covered = set()
        for c in cliques:
            if len(c) < 2:
                return False
            # Check clique validity
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    if e not in edges:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edges

    # Solve both instances
    t1 = time.time()
    budget1 = min(70, DEADLINE - (t1 - start_time) - 5)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(70, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
