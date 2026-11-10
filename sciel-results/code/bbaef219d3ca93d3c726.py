"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """Find a clique partition minimizing the number of cliques."""
        # Generate the graph
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Separate clique and independent set
        C = list(range(k))       # clique vertices
        I = list(range(k, n))    # independent set vertices
        
        # Cross edges
        cross_edges = [(u, v) for u, v in edges if u < k and v >= k]
        cross_edge_set = set(cross_edges)
        
        # The clique edges within C must be covered by cliques that are subsets of C
        # But wait - every edge must be in exactly one clique.
        # Edges within C: these form a complete graph on k vertices.
        # We need to partition these edges into cliques (which are subsets of C).
        # Since C is a clique, any subset of C is also a clique.
        # We want to minimize the total number of cliques.
        
        # Strategy: 
        # 1. Cover cross-edges with cliques of the form {i} ∪ S where i ∈ C, S ⊆ N(i) ∩ I
        #    Actually, cliques can also include multiple C vertices.
        #    A clique in a split graph is of the form A ∪ B where A ⊆ C, B ⊆ I,
        #    and every vertex in B is adjacent to every vertex in A.
        #    So for a clique with C-part A and I-part B, we need B ⊆ ⋂_{a∈A} N(a) ∩ I.
        
        # 2. Cover clique-edges (edges within C) with cliques that are subsets of C.
        
        # Let's think about this differently.
        # Each edge (u,v) with u,v ∈ C must be covered by some clique containing both u and v.
        # Each cross-edge (u,v) with u ∈ C, v ∈ I must be covered by some clique containing both.
        
        # Key insight: If we use a clique {u} ∪ S (single C-vertex u, multiple I-vertices S),
        # it covers all cross-edges (u, s) for s ∈ S. It does NOT cover any C-C edges.
        # If we use a clique {u, v} ∪ S (two C-vertices), it covers C-C edge (u,v) and 
        # cross-edges (u, s) and (v, s) for s ∈ S.
        
        # To minimize cliques, we should try to cover as many edges per clique as possible.
        # A clique with |A| C-vertices and |B| I-vertices covers C(|A|,2) C-C edges and |A|*|B| cross-edges.
        
        # Greedy approach:
        # - Maintain set of uncovered C-C edges and uncovered cross-edges.
        # - At each step, find the clique that covers the most uncovered edges.
        # - Add it, remove covered edges, repeat.
        
        # For efficiency, let's precompute for each subset of C (up to size 2 or 3) 
        # the set of I-vertices adjacent to all of them.
        
        # Actually, let's use a simpler greedy:
        # 1. First handle cross-edges: for each i in C, find all j in I adjacent to i.
        #    Group them. Use cliques {i} ∪ S_i to cover cross-edges.
        # 2. Then handle C-C edges: partition the complete graph on C into cliques.
        #    Since C is a clique, we can cover all C-C edges with just one clique: C itself.
        #    But wait - if we use C as a clique, it covers ALL C-C edges in one clique.
        #    That's optimal for C-C edges!
        
        # So the problem reduces to:
        # - Use ONE clique for all C-C edges: the clique C itself.
        # - Cover all cross-edges with additional cliques.
        #   Each such clique is of the form A ∪ B where A ⊆ C, B ⊆ I, B ⊆ ⋂_{a∈A} N(a).
        #   The clique C already covers no cross-edges (since it has no I-vertices).
        #   Wait, actually the clique C = {0,...,k-1} covers only C-C edges.
        #   Cross-edges need separate cliques.
        
        # Hmm, but we could also incorporate some I-vertices into the main clique.
        # If we make the main clique C ∪ B where B ⊆ ⋂_{c∈C} N(c) (I-vertices adjacent to ALL of C),
        # then it covers all C-C edges AND all cross-edges (c, b) for c ∈ C, b ∈ B.
        
        # Let's compute B_all = {j ∈ I : j adjacent to all c ∈ C}
        B_all = [j for j in I if all(j in adj[c] for c in C)]
        
        # Option 1: Use clique C ∪ B_all as one clique.
        # This covers all C-C edges and cross-edges to B_all.
        # Remaining cross-edges: those (c, j) where j ∉ B_all.
        
        # For remaining cross-edges, use greedy set cover.
        remaining_cross = set((c, j) for c, j in cross_edges if j not in B_all)
        
        # Build candidate cliques for remaining cross-edges.
        # A candidate is (A, B) where A ⊆ C, |A| ≥ 1, B ⊆ I \ B_all, B ⊆ ⋂_{a∈A} N(a),
        # and B is non-empty. The clique is A ∪ B.
        # It covers cross-edges (a, b) for a ∈ A, b ∈ B.
        
        # For efficiency, for each a ∈ C, let N_I(a) = {j ∈ I \ B_all : j adjacent to a}
        N_I = {a: set(j for j in I if j not in B_all and j in adj[a]) for a in C}
        
        # Greedy set cover on cross-edges
        cliques = []
        if B_all:
            cliques.append(sorted(C + B_all))
        else:
            # Need to cover C-C edges separately
            pass
        
        # If B_all is empty, we still need to cover C-C edges.
        # Let's just add C as a clique for C-C edges if needed.
        c_c_edges_covered = len(B_all) > 0  # if we used C ∪ B_all, C-C edges are covered
        
        if not c_c_edges_covered:
            cliques.append(sorted(C[:]))
            c_c_edges_covered = True
        
        # Now cover remaining cross-edges
        uncovered = set(remaining_cross)
        
        # Precompute for each a in C, the neighbors in I\B_all
        # For pairs (a1, a2), compute common neighbors
        # For triples, etc.
        
        # Greedy: at each step, try all possible cliques and pick the one covering most uncovered edges.
        # To limit search space, consider cliques of form {a} ∪ S, {a,b} ∪ S, {a,b,c} ∪ S
        # where S is a subset of common neighbors.
        
        # Actually, for a given A ⊆ C, the best S is the full common neighbor set 
        # (restricted to uncovered edges). So for each A, the clique A ∪ N_common(A) 
        # covers all cross-edges from A to N_common(A).
        
        # Let's enumerate A as subsets of C of size 1, 2, 3 (maybe 4).
        # For each A, compute N_common(A) = ⋂_{a∈A} N_I(a)
        # The clique A ∪ N_common(A) covers edges (a, b) for a ∈ A, b ∈ N_common(A).
        
        # But we might not want to use ALL of N_common(A) - maybe a subset is better?
        # No, using all of N_common(A) is always at least as good since it covers more edges
        # and the clique is still valid.
        
        max_A_size = min(4, k)
        
        while uncovered:
            if time.time() - start_time > DEADLINE:
                # Emergency: cover remaining edges one by one
                for (a, b) in list(uncovered):
                    cliques.append(sorted([a, b]))
                    uncovered.discard((a, b))
                break
            
            best_clique = None
            best_covered = 0
            best_A = None
            best_S = None
            
            # Try all subsets A of C of size 1 to max_A_size
            for size in range(1, max_A_size + 1):
                for A in combinations(C, size):
                    # Compute common neighbors
                    common = set(N_I[A[0]])
                    for a in A[1:]:
                        common &= N_I[a]
                    if not common:
                        continue
                    # Count how many uncovered edges this would cover
                    covered = set()
                    for a in A:
                        for b in common:
                            e = (a, b) if a < b else (b, a)
                            if e in uncovered:
                                covered.add(e)
                    if len(covered) > best_covered:
                        best_covered = len(covered)
                        best_A = A
                        best_S = common
            
            if best_A is None or best_covered == 0:
                # Should not happen if uncovered is non-empty
                # Fallback: cover one edge at a time
                for (a, b) in list(uncovered):
                    cliques.append(sorted([a, b]))
                    uncovered.discard((a, b))
                break
            
            # Add the clique
            clique = sorted(list(best_A) + list(best_S))
            if len(clique) >= 2:
                cliques.append(clique)
            
            # Remove covered edges
            for a in best_A:
                for b in best_S:
                    e = (a, b) if a < b else (b, a)
                    uncovered.discard(e)
        
        # Verify and clean up
        # Ensure all cliques have at least 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        # Final verification
        # Check every edge is in exactly one clique
        edge_to_clique = {}
        for idx, clique in enumerate(cliques):
            for u, v in combinations(clique, 2):
                e = (min(u, v), max(u, v))
                if e in edge_to_clique:
                    # Edge in two cliques - problem
                    pass
                edge_to_clique[e] = idx
        
        # Check all edges are covered
        for e in edge_set:
            if e not in edge_to_clique:
                # Add a clique for this edge
                cliques.append(sorted([e[0], e[1]]))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        # Convert to ensure proper types
        cliques = [[int(v) for v in clique] for clique in cliques]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
