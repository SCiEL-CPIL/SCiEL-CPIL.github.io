"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        """
        Solve minimum clique partition for a split graph.
        
        Key structural insight:
        - C = {0,...,k-1} is a clique, I = {k,...,n-1} is an independent set.
        - Each clique in the partition contains at most one vertex from I.
        - C-C edges must be covered. They can be covered by:
          (a) pure C-cliques (subsets of C), or
          (b) I-vertex cliques {v} ∪ S where S ⊆ N_C(v) and a,b ∈ S.
        
        Strategy:
        1. For each I-vertex v, we must cover all edges (v,u) for u in N_C(v).
           We can split N_C(v) into multiple parts, each forming a clique with v.
           But since C is a clique, any subset of N_C(v) works.
           So the minimum for v is 1 clique if N_C(v) is non-empty.
        
        2. C-C edges: total k(k-1)/2. 
           If we use one big C-clique, it covers all C-C edges. Total = 1 + |non-isolated I|.
           
           Alternative: don't use one big C-clique. Instead, let I-vertex cliques cover
           some C-C edges (the pairs within their C-subset). Then we need fewer/smaller
           pure C-cliques for the remaining C-C edges.
           
           But this doesn't help: if an I-vertex v has clique {v} ∪ S_v, the C-C edges
           within S_v are covered by this clique. But we could have covered them with the
           big C-clique instead. The number of cliques is the same or worse.
        
        3. Actually, the real optimization: Can we reduce the number of I-vertex cliques?
           No - each non-isolated I-vertex needs at least one clique.
        
        4. Can we reduce the pure C-cliques? If we use 1 big C-clique, that's minimal 
           for C-C edges (1 clique covers all k(k-1)/2 edges).
        
        So the optimal is: 1 (big C) + |{v in I : N_C(v) ≠ ∅}|
        
        BUT WAIT: What if some I-vertex has N_C(v) = C (all of C)? Then its clique is
        {v} ∪ C, which covers all C-C edges too! In that case, we don't need the big C-clique.
        
        More generally: if the union of C-subsets from I-vertex cliques covers all C-C edges,
        we don't need any pure C-cliques.
        
        Let's think: I-vertex cliques cover C-C edge (a,b) if there exists an I-vertex v
        with a,b ∈ S_v (where S_v is the C-subset in v's clique).
        
        If we assign S_v = N_C(v) for each v, then C-C edge (a,b) is covered if 
        ∃v: a,b ∈ N_C(v), i.e., both a and b are neighbors of v.
        
        So: C-C edges NOT covered by I-vertex cliques must be covered by pure C-cliques.
        The uncovered C-C edges form a graph G' on C. We need to partition G' into cliques.
        Since G' is a subgraph of a complete graph, we need to cover its edges with cliques.
        
        Actually, G' has edge (a,b) iff NO I-vertex is adjacent to both a and b.
        Equivalently, (a,b) ∈ G' iff N_C^{-1}(a) ∩ N_C^{-1}(b) = ∅ (no common I-neighbor).
        
        We need to partition the edges of G' into the minimum number of cliques.
        This is still a hard problem in general, but G' has a special structure.
        
        For small k (10, 20, 30), we can use a greedy approach for covering G'.
        
        Let me implement:
        1. Build the graph.
        2. For each I-vertex v, create clique {v} ∪ N_C(v).
        3. Determine which C-C edges are covered by these I-vertex cliques.
        4. The remaining C-C edges form G'. Cover G' with minimum cliques (greedy).
        5. Total = (n-k non-isolated) + (cliques to cover G').
        
        Compare with: 1 + (n-k non-isolated) [using big C-clique].
        Take the better one.
        """
        rng = Random(seed)
        # Build cross adjacency: for each I-vertex, which C-vertices it's connected to
        cross_adj = {}
        for v in range(k, n):
            neighbors_in_C = set()
            for u in range(k):
                if rng.random() < p:
                    neighbors_in_C.add(u)
            cross_adj[v] = neighbors_in_C

        non_isolated_I = [v for v in range(k, n) if len(cross_adj[v]) > 0]
        num_non_iso = len(non_isolated_I)

        # Option A: Big C-clique + one clique per non-isolated I-vertex
        # Total: 1 + num_non_iso
        # But only if k >= 2 for the C-clique to be valid (at least 2 vertices)
        
        # Option B: No big C-clique. I-vertex cliques cover some C-C edges.
        # Remaining C-C edges need to be covered by pure C-cliques.
        
        # Determine which C-C edges are covered by I-vertex cliques
        # Edge (a,b) in C is covered if ∃v in non_isolated_I: a ∈ cross_adj[v] and b ∈ cross_adj[v]
        covered_cc = set()
        for v in non_isolated_I:
            neighbors = cross_adj[v]
            if len(neighbors) >= 2:
                neighbors_list = sorted(neighbors)
                for i in range(len(neighbors_list)):
                    for j in range(i+1, len(neighbors_list)):
                        covered_cc.add((neighbors_list[i], neighbors_list[j]))

        # All C-C edges
        all_cc_edges = set()
        for a in range(k):
            for b in range(a+1, k):
                all_cc_edges.add((a, b))

        # Uncovered C-C edges
        uncovered_cc = all_cc_edges - covered_cc

        # Option A cost
        cost_A = 1 + num_non_iso if k >= 2 else num_non_iso

        # Option B: cover uncovered_cc with pure C-cliques
        # Greedy: repeatedly find max clique in the uncovered_cc graph
        # The uncovered_cc graph has vertices 0..k-1 and edges = uncovered_cc
        # We need to partition its edges into cliques.
        
        # For a graph H, minimum edge clique cover can be approximated by:
        # Greedy: find largest clique, remove its edges, repeat.
        # Or: since H is a subgraph of K_k, we can use the complement.
        # Actually, let's just do greedy max clique repeatedly.
        
        def greedy_clique_cover(edges_set, vertices):
            """Cover edges of a graph (given as edge set) with minimum cliques (greedy)."""
            if not edges_set:
                return []
            
            uncovered = set(edges_set)
            cliques = []
            
            while uncovered:
                # Find a maximal clique in the uncovered subgraph
                # Use greedy: pick vertex with max degree, then greedily add neighbors
                deg = {v: 0 for v in vertices}
                for (u, v) in uncovered:
                    deg[u] += 1
                    deg[v] += 1
                
                if max(deg.values()) == 0:
                    break
                
                # Pick vertex with max degree
                start = max(vertices, key=lambda x: deg[x])
                
                # Build clique greedily
                clique = [start]
                candidates = set()
                for (u, v) in uncovered:
                    if u == start:
                        candidates.add(v)
                    elif v == start:
                        candidates.add(u)
                
                # Sort candidates by degree in current candidate set
                def candidate_degree(v):
                    count = 0
                    for w in clique:
                        if (min(v,w), max(v,w)) in uncovered:
                            count += 1
                    return count
                
                for v in sorted(candidates, key=candidate_degree, reverse=True):
                    if all((min(v, w), max(v, w)) in uncovered for w in clique):
                        clique.append(v)
                
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        uncovered.discard(e)
                
                cliques.append(clique)
            
            return cliques

        cc_cliques = greedy_clique_cover(uncovered_cc, list(range(k)))
        cost_B = num_non_iso + len(cc_cliques)

        # Choose the better option
        if cost_A <= cost_B:
            # Option A: big C-clique + I-vertex cliques
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for v in non_isolated_I:
                clique = [v] + sorted(cross_adj[v])
                cliques.append(clique)
        else:
            # Option B: I-vertex cliques + pure C-cliques for uncovered C-C edges
            cliques = []
            for v in non_isolated_I:
                clique = [v] + sorted(cross_adj[v])
                cliques.append(clique)
            for cc_clique in cc_cliques:
                if len(cc_clique) >= 2:
                    cliques.append(cc_clique)

        # Validate the solution
        # Build full adjacency
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        rng2 = Random(seed)
        for v in range(k, n):
            for u in range(k):
                if rng2.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))

        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            if len(clique) < 2:
                return None
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    # Check it's actually an edge in the graph
                    if v not in adj[u]:
                        return None
                    edge = (min(u,v), max(u,v))
                    if edge in edge_to_clique:
                        return None
                    edge_to_clique[edge] = i

        if len(edge_to_clique) != len(all_edges):
            return None

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if (time.time() - t0) > DEADLINE:
            # Emergency: use simple fallback
            n, k, p, seed = spec
            rng = Random(seed)
            cross_adj = {}
            for v in range(k, n):
                neighbors_in_C = set()
                for u in range(k):
                    if rng.random() < p:
                        neighbors_in_C.add(u)
                cross_adj[v] = neighbors_in_C
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for v in range(k, n):
                if len(cross_adj[v]) > 0:
                    cliques.append([v] + sorted(cross_adj[v]))
            clean_result = [[int(v) for v in clique] for clique in cliques]
            results.append(clean_result)
            continue
        
        n, k, p, seed = spec
        result = solve_split_graph(n, k, p, seed)
        if result is None:
            # Fallback: big C + one per non-isolated I
            rng = Random(seed)
            cross_adj = {}
            for v in range(k, n):
                neighbors_in_C = set()
                for u in range(k):
                    if rng.random() < p:
                        neighbors_in_C.add(u)
                cross_adj[v] = neighbors_in_C
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for v in range(k, n):
                if len(cross_adj[v]) > 0:
                    cliques.append([v] + sorted(cross_adj[v]))
            result = cliques

        # Ensure types are correct
        clean_result = []
        for clique in result:
            clean_clique = [int(v) for v in clique]
            clean_result.append(clean_clique)
        results.append(clean_result)

    return results
# EVOLVE-BLOCK-END
