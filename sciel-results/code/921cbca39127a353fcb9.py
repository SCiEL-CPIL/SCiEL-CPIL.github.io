"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # 1. Partition the clique part (0..k-1) into minimum number of cliques (layers)
        # Since 0..k-1 is a clique, we can just put all in one layer, 
        # but to minimize total cliques, we might want to split if it helps?
        # Actually, for split graphs, the optimal partition often uses large cliques.
        # However, if we use one big clique for C, we can only add vertices from I 
        # that are connected to ALL of C.
        # If we split C into smaller cliques, we can add more vertices to each.
        
        # Let's try to find a good partition of C into cliques.
        # Since C is a clique, any subset is a clique.
        # The constraint is on the I vertices.
        
        # Strategy:
        # 1. Start with C as a single clique.
        # 2. Identify I vertices connected to all of C. Add them to this clique.
        # 3. The remaining I vertices and C vertices form a subgraph?
        # No, C vertices are already covered in the first clique.
        # Wait, the "Clique Partition" problem requires covering all edges.
        # Edges within C are covered by the first clique.
        # Edges between C and I:
        #   - If v in I is in the first clique, edges (u,v) for all u in C are covered.
        #   - If v in I is NOT in the first clique, edges (u,v) must be covered by other cliques.
        #     But other cliques cannot contain vertices from C that are not in the first clique?
        #     No, a vertex can be in multiple cliques.
        #     However, if we put u in C into another clique, it must be with a subset of I vertices.
        
        # Let's re-evaluate.
        # A clique partition is a set of cliques such that every edge is in exactly one.
        # This is equivalent to a "clique edge partition".
        
        # For split graphs, there is a known structure.
        # Let C be the clique, I be the independent set.
        # Edges are E(C), E(C,I).
        # E(C) must be covered by cliques containing vertices from C.
        # E(C,I) must be covered by cliques containing one from C and one from I (or more).
        
        # Observation:
        # Any clique in a split graph is either a subset of C, or a subset of C plus some vertices from I that are adjacent to all vertices in the C-subset.
        
        # Let's try a greedy approach that builds cliques:
        # 1. Find a maximal clique.
        # 2. Remove its edges.
        # 3. Repeat.
        
        # But we want to MINIMIZE the number of cliques.
        # This is the "Clique Cover" problem restricted to edge partition (which is the same as clique cover for chordal graphs? No, edge partition is stricter).
        # Actually, for chordal graphs, the minimum clique partition can be found in polynomial time?
        # Yes, for chordal graphs, the minimum clique partition is related to the clique tree.
        # However, implementing the exact chordal graph clique partition algorithm is complex.
        
        # Let's use a heuristic that is likely to be very good for split graphs.
        # 1. Sort I vertices by degree (descending).
        # 2. Try to form large cliques involving high-degree I vertices.
        
        # Better heuristic for split graphs:
        # 1. Partition C into "groups" G1, G2, ..., Gm.
        # 2. For each group Gi, find the set of I vertices adjacent to all of Gi. Let this be N(Gi).
        # 3. Assign each I vertex to the "best" group it is adjacent to.
        #    "Best" means we want to cover its edges.
        #    If we assign v to Gi, we cover edges (u,v) for u in Gi.
        #    We must ensure that no edge is covered twice.
        #    If v is assigned to Gi, we CANNOT assign it to Gj (j != i) if it shares edges?
        #    No, if v is in Gi, edges (u,v) for u in Gi are covered.
        #    Edges (u,v) for u NOT in Gi must be covered by other cliques.
        #    This implies v must also be in cliques covering those edges.
        #    But if v is in Gj, edges (u,v) for u in Gj are covered.
        #    So v can be in multiple cliques, as long as the C-vertices in those cliques are disjoint?
        #    Yes! If the C-vertices in the cliques containing v are disjoint, the edges are disjoint.
        
        # So the problem reduces to:
        # Partition C into disjoint subsets G1, ..., Gm.
        # For each v in I, let S_v be the set of groups Gi such that v is adjacent to ALL vertices in Gi.
        # We must assign v to a subset of S_v such that:
        #   - For every edge (u,v), u is in exactly one group assigned to v.
        #   - This is automatically satisfied if we assign v to ALL groups Gi in S_v?
        #     No, if we assign v to ALL groups in S_v, then for any u in C, if u is in Gi, and v is adjacent to u, then v is adjacent to ALL of Gi?
        #     Not necessarily. v might be adjacent to u but not to another u' in Gi.
        #     So v can only be in Gi if v is adjacent to ALL of Gi.
        
        # If we partition C into G1...Gm, and for each v, we put v in every Gi where v is adjacent to all of Gi,
        # then every edge (u,v) is covered exactly once (in the group containing u, provided v is adjacent to all of that group).
        # What if v is NOT adjacent to all of the group containing u?
        # Then the edge (u,v) is NOT covered by that group's clique.
        # It must be covered by another clique.
        # But the only other cliques containing v are those for groups Gj where v is adjacent to all of Gj.
        # If u is in Gi, and v is not adjacent to all of Gi, then v cannot be in the clique for Gi.
        # So edge (u,v) is uncovered!
        
        # Therefore, the partition of C must be such that for every edge (u,v), v is adjacent to ALL vertices in the group containing u.
        # This means: If u and u' are in the same group Gi, then N(v) must contain both u and u' or neither?
        # No, if (u,v) is an edge, u is in Gi. Then v must be adjacent to all of Gi.
        # So if (u,v) is an edge and (u',v) is NOT an edge, then u and u' CANNOT be in the same group.
        
        # Thus, the groups G1...Gm must be subsets of C such that for any v in I, the set of neighbors N(v) is a union of some of the groups.
        # In other words, the partition of C must be "compatible" with the neighborhoods of all v in I.
        
        # This is equivalent to finding a partition of C such that each part is a subset of the intersection of complements of non-neighborhoods?
        # Specifically, for any two vertices u, u' in C, they can be in the same group ONLY IF for every v in I, (u,v) is an edge iff (u',v) is an edge?
        # No, that's too strong.
        # They can be in the same group if there is NO v in I such that exactly one of (u,v) and (u',v) is an edge.
        # Because if such a v exists, and u, u' are in the same group Gi, then:
        #   If (u,v) is edge, v must be adjacent to all of Gi, so (u',v) must be edge. Contradiction.
        #   If (u,v) is not edge, v is not adjacent to all of Gi, so v cannot be in Gi's clique. Then (u',v) (which is an edge) is uncovered. Contradiction.
        
        # So, u and u' must be in the SAME group if and only if they have the SAME neighborhood in I?
        # No, they MUST be in the SAME group if they have the same neighborhood?
        # If N(u) = N(u'), they can be in the same group.
        # If N(u) != N(u'), they MUST be in different groups?
        # Yes, as shown above.
        
        # Therefore, the minimum number of groups is the number of DISTINCT neighborhoods in I for vertices in C.
        # Let the distinct neighborhoods be H1, H2, ..., Hm.
        # Group Gi contains all u in C such that N(u) = Hi.
        
        # Now, for each group Gi, the clique is Gi union {v in I | v is adjacent to all of Gi}.
        # Since all u in Gi have the same neighborhood Hi, "v is adjacent to all of Gi" is equivalent to "Gi subset of N(v)"?
        # No, "v is adjacent to all u in Gi" means Gi is a subset of N(v) (where N(v) is neighbors of v in C).
        # Wait, N(u) is neighbors of u in I.
        # Let's denote N_I(u) = {v in I | (u,v) in E}.
        # Condition: v is in clique for Gi iff for all u in Gi, (u,v) in E.
        # This is equivalent to: Gi subset of N_C(v), where N_C(v) = {u in C | (u,v) in E}.
        
        # So the algorithm is:
        # 1. Compute N_C(v) for each v in I.
        # 2. Compute N_I(u) for each u in C.
        # 3. Group vertices in C by their N_I(u). Let the groups be G1, ..., Gm.
        # 4. For each group Gi, the clique is Gi union {v in I | Gi subset of N_C(v)}.
        # 5. Return these m cliques.
        
        # Let's verify this covers all edges exactly once.
        # Edges in C: Covered by the group they belong to. Since groups are disjoint, each edge is in exactly one group.
        # Edges in C-I: Let (u,v) be an edge. u is in some group Gi.
        # Since (u,v) is an edge, u is in N_C(v).
        # Since all u' in Gi have the same N_I, and u is in Gi, does v have the same relation to all u' in Gi?
        # Yes, because N_I(u) = N_I(u') for all u' in Gi.
        # Wait, N_I(u) is the set of v's.
        # If v is in N_I(u), then v is adjacent to u.
        # Since N_I(u) = N_I(u'), v is also in N_I(u'), so v is adjacent to u'.
        # So v is adjacent to ALL u' in Gi.
        # Therefore, Gi subset of N_C(v).
        # So v is included in the clique for Gi.
        # Thus edge (u,v) is covered by the clique for Gi.
        # Is it covered by any other clique Gk?
        # v is in Gk's clique iff Gk subset of N_C(v).
        # If v is in Gk's clique, then all u in Gk are adjacent to v.
        # But u is in Gi. If Gi != Gk, then N_I(u) != N_I(u') for some u' in Gk?
        # Not necessarily. But u is in Gi, so u is not in Gk.
        # The edge (u,v) has u in Gi. It is covered by Gi's clique.
        # Could it be covered by Gk's clique? Only if u is in Gk, which is false.
        # So each edge is covered exactly once.
        
        # This yields a valid clique partition.
        # Is it minimal?
        # The number of cliques is the number of distinct neighborhoods in I for vertices in C.
        # This is a lower bound because vertices in C with different neighborhoods must be in different cliques (as proven earlier).
        # So this is the OPTIMAL solution for split graphs!
        
        # Implementation:
        # 1. Build adjacency lists.
        # 2. For each u in C, compute tuple(N_I(u)).
        # 3. Group u's by this tuple.
        # 4. For each group, find v's in I such that group is subset of N_C(v).
        # 5. Form cliques.
        
        # Complexity:
        # n=90, k=30.
        # C has 30 vertices. I has 60 vertices.
        # Edges are ~ 30*29/2 + 30*60*0.7 = 435 + 1260 = ~1700.
        # Very small.
        
        # Build N_C and N_I
        N_C = [set() for _ in range(n)]
        N_I = [set() for _ in range(n)]
        
        for u, v in edges:
            N_C[u].add(v)
            N_C[v].add(u)
            N_I[u].add(v)
            N_I[v].add(u)
        
        # Note: N_C[v] for v in I contains only vertices in C (since I is independent).
        # N_I[u] for u in C contains only vertices in I.
        
        groups = {}
        for u in range(k):
            key = frozenset(N_I[u])
            if key not in groups:
                groups[key] = []
            groups[key].append(u)
        
        cliques = []
        for key, group_vertices in groups.items():
            clique = list(group_vertices)
            # Find v in I such that group_vertices is subset of N_C[v]
            for v in range(k, n):
                # Check if all u in group_vertices are in N_C[v]
                # Since group_vertices is small, this is fast
                if all(u in N_C[v] for u in group_vertices):
                    clique.append(v)
            if len(clique) >= 2:
                cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: return current best or empty? Must return valid.
            # If we haven't finished, we can't return invalid.
            # But this algorithm is O(n^2) which is tiny. It will never timeout.
            pass
        res = solve_split(*spec)
        # Ensure types are Python int and list
        res = [[int(v) for v in clique] for clique in res]
        results.append(res)
    
    return results
# EVOLVE-BLOCK-END
