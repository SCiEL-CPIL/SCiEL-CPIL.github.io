import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)

        # All edges within K (the first k vertices form a clique)
        k_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                k_edges.add((u, v))

        # For each v in I, get its neighbor set in K
        cross_edges = []
        for v in range(k, n):
            for u in neighbors[v]:
                cross_edges.append((u, v))

        # Build the shadow graph: a K-edge (u,w) is "shadowed" by v if both u,w in N(v)
        # We need to assign each K-edge to at most one v (or to "K" for standalone coverage)
        # 
        # Strategy: For each v, we create one clique N(v) ∪ {v}. 
        # The K-edges within N(v) are covered by this clique.
        # But if a K-edge is in N(v1) and N(v2), it appears in both cliques → invalid.
        #
        # Solution: For each K-edge, assign it to exactly one v that shadows it, or to "K".
        # For v, the clique is N(v) ∪ {v}. The K-edges within N(v) that are assigned to v
        # are "covered" by v's clique. But the clique still contains ALL K-edges within N(v).
        # So we can't avoid the conflict by assignment alone.
        #
        # Correct approach: We must ensure that for any K-edge (u,w), there is at most one
        # clique that contains both u and w. Since K is a clique, any subset of K is a clique.
        # If we create clique N(v)∪{v}, it contains all K-edges within N(v).
        #
        # To avoid conflicts: we can only create clique N(v)∪{v} if no other v' has a clique
        # that also contains any K-edge within N(v). This is complex.
        #
        # Simpler correct approach: 
        # - For each v, create clique N(v) ∪ {v}. This covers all cross edges for v.
        # - For K-edges: each K-edge (u,w) is in clique C_v if u,w ∈ N(v). 
        # - If a K-edge is in multiple C_v's, we have a problem.
        #
        # Fix: After forming all C_v, for each K-edge, if it's in multiple C_v's, we need
        # to "remove" it from all but one. But we can't remove edges from a clique.
        #
        # Alternative: Don't form C_v for v's that cause conflicts. Instead, use edge-cliques
        # for v's cross edges.
        #
        # Let me think differently. The key insight from the cell finding:
        # "one big clique on the first k vertices covers all clique-edges, and each cross edge
        # must be its own 2-clique" giving 1 + |cross| per instance.
        #
        # Wait, that's not right either. If we use one big clique K for all K-edges, and each
        # cross edge as a 2-clique, then:
        # - K-edges: all in the big clique K. ✓
        # - Cross edges: each in its own 2-clique. ✓
        # - No edge is in two cliques. ✓
        # Total: 1 + |cross_edges|
        #
        # But can we do better? If we group cross edges: for v, if N(v) = {u1, u2, ..., um},
        # we can make clique {u1, u2, ..., um, v}. This covers m cross edges AND all K-edges
        # within {u1,...,um}. But those K-edges are already in the big clique K! Conflict.
        #
        # So we can't use the big clique K AND grouped cross cliques. We must choose:
        # Option A: Big clique K + individual cross 2-cliques. Total: 1 + |cross|.
        # Option B: Grouped cross cliques + cover remaining K-edges separately.
        #
        # For Option B: For each v, create clique N(v)∪{v}. K-edges within N(v) are covered.
        # K-edges not in any N(v) need separate coverage. But K-edges in multiple N(v) are
        # covered multiple times → invalid.
        #
        # To fix Option B: assign each K-edge to at most one v. For v, create clique N(v)∪{v},
        # but this clique contains K-edges not assigned to v. Those K-edges are "in" the clique
        # but not "covered" by it? No, the rule is about which cliques an edge lies in.
        #
        # The edge (u,w) lies in clique C_v if u,w ∈ N(v), regardless of assignment.
        # So if (u,w) ∈ N(v1) ∩ N(v2), it lies in both C_v1 and C_v2. Invalid.
        #
        # Therefore, Option B requires that for any K-edge, it's in at most one N(v).
        # This means we can only use grouped cliques for v's where N(v) doesn't share K-edges
        # with other v's.
        #
        # Practical approach: 
        # 1. Use the big clique K for all K-edges.
        # 2. For cross edges, use individual 2-cliques.
        # Total: 1 + |cross|.
        #
        # Can we improve? If for some v, N(v) = {u} (single neighbor), the 2-clique {u,v}
        # doesn't add any K-edge. If N(v) = {u,w}, the 2-cliques {u,v} and {w,v} don't add
        # K-edges either. The big clique K already covers (u,w).
        #
        # So Option A is always valid and gives 1 + |cross|.
        #
        # Is there a better option? What if we don't use the big clique K, and instead
        # cover K-edges via grouped cross cliques where possible?
        #
        # For a K-edge (u,w), it's covered by v's clique if u,w ∈ N(v). If we create cliques
        # for all v, and a K-edge is in multiple v's N(v), it's in multiple cliques.
        #
        # To avoid this: for each K-edge, pick at most one v to "claim" it. But the clique
        # N(v)∪{v} still contains all K-edges within N(v), not just claimed ones.
        #
        # I think the cleanest optimal approach is:
        # - For each v in I, create clique N(v) ∪ {v} ONLY IF the K-edges within N(v) are
        #   not in any other v's N(v'). Otherwise, use 2-cliques for v's cross edges.
        # - Cover remaining K-edges (not in any N(v) of a v that got a grouped clique) with
        #   a greedy clique cover.
        #
        # This is complex. Let me just go with Option A: 1 + |cross|.
        # For n=90, k=30, p=0.7: |cross| ≈ 60*30*0.7 = 1260. Total ≈ 1261.
        # The trivial edge-by-edge would be 435 + 1260 = 1695. So 1261 is better.
        #
        # Actually, can we do even better than 1 + |cross|? 
        # If we group cross edges for v's where it doesn't conflict, we save cliques.
        # For v with |N(v)| = m, grouping saves (m-1) cliques (m 2-cliques → 1 m+1 clique).
        # But we lose the K-edge coverage from the big clique for edges within N(v).
        #
        # Net: if we group for v, we save (m-1) cross cliques but need to cover K-edges
        # within N(v) separately (unless covered by other grouped v's or a K-clique).
        #
        # This is a tradeoff. For now, let me implement Option A and see.

        # Option A: big clique K + individual cross 2-cliques
        cliques = []
        
        # Big clique for K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Individual cross 2-cliques
        for v in range(k, n):
            for u in neighbors[v]:
                cliques.append([u, v])
        
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
