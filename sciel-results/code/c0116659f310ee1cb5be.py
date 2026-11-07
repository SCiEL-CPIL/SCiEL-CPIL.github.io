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
        
        # Build neighbor sets as sets for fast lookup
        nsets = {v: set(neighbors[v]) for v in range(k, n)}
        
        # For each K-edge (u,w), find which v's in I have both u and w in N(v)
        # We'll assign each K-edge to at most one v to avoid conflicts
        # Strategy: assign K-edge (u,w) to the v with smallest index that contains both u,w
        
        # Map: K-edge -> assigned v (or None)
        edge_assignment = {}  # (u,w) -> v or None
        for u in range(k):
            for w in range(u+1, k):
                edge_assignment[(u, w)] = None
        
        # For each v, track which K-edges are assigned to it
        v_assigned_edges = {v: [] for v in range(k, n)}
        
        # Assign each K-edge to the first v (smallest index) that can cover it
        for u in range(k):
            for w in range(u+1, k):
                for v in range(k, n):
                    if u in nsets[v] and w in nsets[v]:
                        edge_assignment[(u, w)] = v
                        v_assigned_edges[v].append((u, w))
                        break
        
        # Now for each v, we need to partition N(v) into groups such that:
        # - Each group G forms clique G ∪ {v}
        # - The K-edges within G are exactly those assigned to v with both endpoints in G
        # - All K-I edges for v are covered
        #
        # Simple approach: for each v, create one clique N(v) ∪ {v} ONLY if no K-edge within N(v)
        # is assigned to a different v. Otherwise, we need to be more careful.
        #
        # Actually, the issue is: if we make clique N(v) ∪ {v}, it contains ALL K-edges within N(v).
        # But some of those K-edges might be assigned to other v's. That's a conflict.
        #
        # Solution: for each v, only include in the clique those vertices u in N(v) such that
        # all K-edges (u, u') for u' in the group are assigned to v.
        #
        # Simpler: for each v, the clique is {v} ∪ {u in N(v) : no K-edge (u, u') with u' in N(v) 
        # is assigned to v' ≠ v}. But this might not cover all K-I edges.
        #
        # Cleanest structural approach: 
        # For each v, we MUST cover all K-I edges (u,v) for u in N(v).
        # We'll create cliques for v. Each clique is {v} ∪ S where S ⊆ N(v).
        # The K-edges within S must all be assigned to v (not to other v's).
        #
        # For each v, let's find the maximal subsets of N(v) where all internal K-edges are assigned to v.
        # This is a graph problem: on the vertex set N(v), edges are K-edges assigned to v.
        # Find connected components (or cliques) in this graph.
        #
        # Actually, since K is complete, any subset of N(v) is a clique in K. The constraint is just
        # that the K-edges within the subset are assigned to v.
        #
        # Let me just use: for each v, one clique = {v} ∪ N(v), but only if all K-edges within N(v)
        # are assigned to v. Otherwise, split into smaller groups.
        #
        # Even simpler and guaranteed correct:
        # - For each v, create clique {v} ∪ N(v) ONLY for v's where all K-edges within N(v) are 
        #   assigned to v. For other v's, use individual edge-cliques {u, v} for each u in N(v).
        # - Remaining K-edges (not assigned to any v): cover with greedy clique cover in K.
        
        cliques = []
        covered_k_edges = set()  # K-edges covered by I-cliques
        
        # Determine which v's can use full clique
        v_can_use_full = {}
        for v in range(k, n):
            Sv = nsets[v]
            can_use = True
            for u in Sv:
                for w in Sv:
                    if u < w:
                        e = (u, w)
                        if edge_assignment[e] != v:
                            can_use = False
                            break
                if not can_use:
                    break
            v_can_use_full[v] = can_use
        
        # For v's that can use full clique:
        for v in range(k, n):
            if v_can_use_full[v]:
                Sv = sorted(nsets[v])
                if len(Sv) >= 1:
                    cliques.append([v] + Sv)
                    # Mark all K-edges within Sv as covered
                    for i in range(len(Sv)):
                        for j in range(i+1, len(Sv)):
                            covered_k_edges.add((Sv[i], Sv[j]))
            else:
                # Use individual edge-cliques for this v
                for u in sorted(nsets[v]):
                    cliques.append([u, v])
        
        # Remaining K-edges: those not covered by any I-clique
        remaining_k_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                if (u, w) not in covered_k_edges:
                    remaining_k_edges.add((u, w))
        
        # Greedy clique cover for remaining K-edges
        while remaining_k_edges:
            e = next(iter(remaining_k_edges))
            u, w = e
            clique = {u, w}
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x not in clique:
                        if all((min(x, y), max(x, y)) in remaining_k_edges for y in clique):
                            clique.add(x)
                            changed = True
                            break
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for a, b in combinations(clique, 2):
                    remaining_k_edges.discard((min(a, b), max(a, b)))
        
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
