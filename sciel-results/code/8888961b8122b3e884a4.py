import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Split vertices: C = [0, k), I = [k, n)
        C_edges = set()
        cross_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                C_edges.add((u, v))
            elif (u < k and v >= k) or (u >= k and v < k):
                cross_edges.add((u, v))
            else:
                # I-I edges should not exist in this generator
                pass
        
        # 1. Construct I-cliques: {i} U S_i
        # S_i should be a maximal clique in the subgraph induced by N(i) on C.
        # Since C is a clique, any subset of N(i) is a clique in G.
        # To minimize the number of cliques, we want S_i to be as large as possible?
        # Actually, each I-clique covers the cross-edges incident to i and within S_i.
        # If we pick S_i = N(i), we cover ALL cross-edges incident to i.
        # This leaves only C-edges to be covered.
        # Is it better to pick smaller S_i?
        # If we pick smaller S_i, we leave some cross-edges incident to i uncovered.
        # Those cross-edges must be covered by other cliques.
        # Since I vertices are independent, they cannot be in a clique with each other.
        # So cross-edges (i, c) can ONLY be covered by a clique containing i.
        # The clique must be {i} U S where c in S.
        # If we don't cover (i, c) in the clique for i, we must cover it in another clique containing i.
        # But we only make one clique per i? No, we can make multiple.
        # However, the problem asks for a partition.
        # If we split N(i) into multiple cliques, we increase the count.
        # So for each i, the best is to make exactly one clique {i} U N(i).
        # This covers all cross-edges incident to i.
        # Total I-cliques = |I| = n - k.
        # Remaining edges to cover: C_edges.
        
        # Wait, is it possible that NOT covering all cross edges in one clique per i allows for a better C-cover?
        # No, because C-edges are disjoint from cross-edges.
        # The C-edges must be covered by cliques that are subsets of C (since I vertices can't be in C-cliques).
        # So the C-edges form a subgraph on C. We need to cover this subgraph with cliques.
        # The choice of S_i (subset of N(i)) does NOT affect which C-edges remain.
        # All C-edges remain regardless of how we cover cross-edges.
        # So the optimal strategy is:
        # 1. For each i in I, add clique {i} U N(i).
        # 2. Cover the remaining graph (which is just C with C_edges) with minimum cliques.
        
        # Let's verify this logic.
        # Constraint: Every edge must lie in EXACTLY one listed clique.
        # If I add {i} U N(i), it covers edges (i, c) for c in N(i).
        # It also covers edges (c1, c2) for c1, c2 in N(i).
        # BUT, if (c1, c2) is a C-edge, and it is covered by {i} U N(i), it CANNOT be covered by a pure C-clique later.
        # So, covering cross-edges "eats" C-edges!
        # This is the key.
        # If I include c1, c2 in S_i, the edge (c1, c2) is consumed.
        # If I leave c1, c2 out of S_i, the edge (c1, c2) remains for the C-cover.
        # So I should choose S_i to consume as many C-edges as possible?
        # Yes! Consuming a C-edge saves 1 unit of "C-cover cost" (roughly).
        # But adding a vertex c to S_i costs 0 extra "I-clique count" (since we already have the clique for i).
        # So we should include ALL neighbors in S_i?
        # If I include c in S_i, I consume all edges (c, c') for c' in S_i.
        # So yes, S_i = N(i) is optimal for maximizing consumption of C-edges?
        # Let's check.
        # If S_i = N(i), we consume all edges within N(i).
        # If S_i is smaller, we consume fewer edges.
        # So S_i = N(i) is the greedy choice to minimize remaining C-edges.
        
        # Let's proceed with S_i = N(i).
        
        result_cliques = []
        
        # Build N(i)
        N_i = [set() for _ in range(n)]
        for u, v in cross_edges:
            if u < k:
                i, c = u, v
            else:
                i, c = v, u
            N_i[i].add(c)
        
        for i in range(k, n):
            if N_i[i]:
                clique = [i] + list(N_i[i])
                result_cliques.append(clique)
                # Mark C-edges within N_i[i] as covered
                members = list(N_i[i])
                for idx in range(len(members)):
                    for jdx in range(idx + 1, len(members)):
                        c1, c2 = members[idx], members[jdx]
                        if c1 < c2:
                            C_edges.discard((c1, c2))
                        else:
                            C_edges.discard((c2, c1))
        
        # Now cover remaining C_edges with minimum cliques.
        # This is the Minimum Clique Cover problem on the graph G_C = (C, C_edges).
        # Equivalently, Maximum Independent Set on the complement graph G_C_bar.
        # Vertices: C = {0, ..., k-1}
        # Edges in G_C_bar: (u, v) if u != v and (u, v) NOT in C_edges.
        
        # Since k is small (10, 20, 30), we can use a fast Branch and Bound for Max Independent Set.
        # Or simply, since C is a clique in the original graph, the complement G_C_bar is the graph of NON-edges in C.
        # Wait, C is a clique in G. So ALL pairs (u, v) in C are edges in G.
        # Therefore, the complement G_C_bar has NO edges?
        # If G_C_bar has no edges, the Max Independent Set is the whole set C.
        # Which means the Min Clique Cover of G_C is 1 clique (the whole set C)?
        # NO.
        # G_C is the subgraph of G induced by C.
        # G is a split graph. C is a clique.
        # So G_C is a complete graph K_k?
        # Let's check the generator.
        # "edges = [(u, v) for u in range(k) for v in range(u + 1, k)]"
        # Yes, all pairs in C are edges.
        # So G_C is K_k.
        # The remaining C_edges are a subset of E(K_k).
        # We need to cover these remaining edges with cliques.
        # Since G_C is a subgraph of K_k, any subset of vertices forms a clique in G.
        # So we just need to partition the vertices involved in remaining C_edges into cliques?
        # No, we just need to cover the EDGES.
        # Since any subset is a clique, we can cover the remaining edges by grouping vertices.
        # Actually, if we have a set of remaining edges, we can cover them all with 1 clique if we take the union of all vertices involved?
        # Yes! If we take V_rem = union of endpoints of C_edges, then {V_rem} is a clique in G (since C is a clique).
        # And it covers all edges in C_edges.
        # So we can always cover all remaining C-edges with 1 clique!
        # Unless... we want to minimize the number of cliques.
        # 1 is the minimum possible (if there are any edges).
        # So the total number of cliques is |I| + 1?
        
        # Let's re-read carefully.
        # "Every listed set must be a clique... and every edge must lie in exactly one listed clique."
        # If I add {V_rem}, it covers all remaining C-edges.
        # Does it cover any cross-edges? No, V_rem is subset of C.
        # Does it cover any I-I edges? No.
        # So yes, 1 clique suffices for all remaining C-edges.
        
        # So the total count is (n - k) + 1?
        # Let's check the seed program's output.
        # The seed program does a complex greedy.
        # If my logic is correct, the seed program is suboptimal and I can do much better.
        # Why would the seed program do a complex greedy?
        # Maybe I am missing a constraint.
        # "Clique partitions of three fixed split (chordal) graphs"
        # "Minimize the total number of cliques"
        
        # Is it possible that {V_rem} is not a valid clique?
        # "Clique on the first k vertices" -> Yes, C is a clique.
        # So any subset of C is a clique.
        
        # So the optimal solution is:
        # 1. For each i in I, clique {i} U N(i).
        # 2. One clique for all remaining C-edges.
        
        # Let's test this hypothesis.
        # If this is true, the score should be very high (low value).
        # The Erdős-Ordman-Zalcstein question asks if n^2/6 + O(n) suffice.
        # My solution: (n-k) + 1.
        # For n=90, k=30: 60 + 1 = 61.
        # n^2/6 = 8100/6 = 1350.
        # 61 is much smaller than 1350.
        
        # Wait, is it that easy?
        # Why is this a research question then?
        # "Clique partitions" usually refers to partitioning the VERTEX SET into cliques?
        # No, "every edge must lie in exactly one listed clique". This is an EDGE PARTITION (or edge coloring with cliques).
        # This is known as "Clique Covering of Edges" or "Clique Partition of Edges".
        
        # If C is a clique, then any subset of C is a clique.
        # So yes, we can cover any subset of C-edges with 1 clique.
        
        # So the algorithm is trivial?
        # 1. Generate graph.
        # 2. For each i in I, add {i} + N(i).
        # 3. Add 1 clique for remaining C-edges.
        
        # Let's double check if "Clique on the first k vertices" means the induced subgraph is a clique.
        # Yes, "clique on the first k vertices" implies all edges exist.
        
        # So I will implement this simple strategy.
        
        if C_edges:
            # Collect all vertices involved in remaining C_edges
            v_set = set()
            for u, v in C_edges:
                v_set.add(u)
                v_set.add(v)
            if v_set:
                result_cliques.append(list(v_set))
        
        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
