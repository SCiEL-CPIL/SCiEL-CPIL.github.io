"""Clique partition for three fixed split graphs using a max-flow-based matching approach."""
import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # Build adjacency
        # K = clique vertices 0..k-1
        # I = independent set vertices k..n-1
        # For each I vertex, store which K vertices it's connected to
        cross_adj = [set() for _ in range(n - k)]  # cross_adj[i] = set of K vertices connected to I vertex (k+i)

        # For each K vertex, store which I vertices it's connected to
        k_to_i = [set() for _ in range(k)]

        for u, v in edges:
            if u < k and v >= k:
                cross_adj[v - k].add(u)
                k_to_i[u].add(v - k)
            elif v < k and u >= k:
                cross_adj[u - k].add(v)
                k_to_i[v].add(u - k)

        # K-K edges: all pairs in K are edges
        kk_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]

        # Strategy: 
        # Type-B cliques: For each I vertex i, we can form a clique {i+k} ∪ S where S ⊆ N_K(i)
        # and S must be such that all pairs in S form K-K edges (which they always do since K is a clique)
        # The key constraint: each K-K edge can be in at most one Type-B clique
        # Each cross edge (u, i+k) must be in exactly one clique
        # 
        # For a Type-B clique {i+k} ∪ S:
        # - Covers cross edges (u, i+k) for u in S
        # - Covers K-K edges (u, v) for u, v in S
        #
        # We want to maximize the number of K-K edges covered by Type-B cliques (to minimize Type-A cliques)
        # while ensuring all cross edges are covered.
        #
        # Since each cross edge (u, i+k) must be covered by exactly one clique, and it can only be 
        # covered by a Type-B clique involving i+k (since I is independent, no other clique can contain i+k
        # and u), we must have u in S for the Type-B clique of i+k.
        # Wait, that's not right. The cross edge (u, i+k) can be covered by:
        # 1. A Type-B clique {i+k} ∪ S where u ∈ S
        # 2. A Type-A clique that contains both u and i+k? No, Type-A only uses K vertices.
        # Actually, i+k is in I, so it can only appear in Type-B cliques.
        # So cross edge (u, i+k) MUST be covered by the Type-B clique for i+k, meaning u must be in S_i.
        #
        # Therefore, for each active I vertex i (with at least one K neighbor), S_i must contain ALL of N_K(i).
        # But then S_i = N_K(i), and the Type-B clique is {i+k} ∪ N_K(i).
        # This means the K-K edges covered by Type-B clique for i are all edges within N_K(i).
        #
        # Wait, but if two I vertices share a K neighbor, that's fine - the cross edges are different.
        # The issue is: if N_K(i) contains vertices a and b, then the K-K edge (a,b) is covered by the
        # Type-B clique for i. If N_K(j) also contains a and b, then (a,b) would be covered twice.
        # So we cannot have both i and j forming Type-B cliques that both include {a,b}.
        #
        # So the constraint is: for any K-K edge (a,b), at most one Type-B clique can have both a and b in its S.
        #
        # To minimize total cliques, we want to:
        # 1. Form Type-B cliques for as many I vertices as possible (each Type-B clique covers one I vertex)
        # 2. Maximize the number of K-K edges covered by Type-B cliques
        #
        # But every active I vertex MUST have a Type-B clique (since its cross edges must be covered).
        # So we form Type-B cliques for all active I vertices.
        # The question is: can we reduce S_i to avoid conflicts? No, because S_i must contain all N_K(i).
        # 
        # Wait, I'm confusing myself. Let me re-read the problem.
        # "Every edge must lie in exactly one listed clique."
        # So each cross edge (u, i+k) must be in exactly one clique. The only clique that can contain i+k
        # is a Type-B clique for i. So yes, S_i must contain all of N_K(i).
        #
        # Therefore, Type-B cliques are forced: for each active I vertex i, the clique is {i+k} ∪ N_K(i).
        # The K-K edges covered by Type-B are: for each i, all edges within N_K(i).
        # If a K-K edge is covered by multiple Type-B cliques, that's a violation.
        #
        # So we need to check: is there a K-K edge covered by two different Type-B cliques?
        # If so, we have a problem. But the problem statement says split graphs need at most 3n^2/16 + O(n),
        # implying a solution exists. Let me reconsider.
        #
        # Actually, I think the key insight is: we don't HAVE to put all of N_K(i) in S_i. 
        # Wait, but the cross edge (u, i+k) for u in N_K(i) must be covered. If u is not in S_i,
        # then (u, i+k) is not covered by the Type-B clique for i. Can it be covered by another clique?
        # The other clique would need to contain both u and i+k. Since i+k is in I, and I is independent,
        # the only clique containing i+k is the Type-B clique for i. So no, (u, i+k) can only be covered
        # by the Type-B clique for i. Therefore, u must be in S_i.
        #
        # So S_i = N_K(i) is forced. And if this leads to double coverage of K-K edges, we have a problem.
        # But the problem guarantees a solution exists, so maybe I'm wrong about the structure.
        #
        # Let me reconsider: maybe a K vertex can appear in multiple Type-B cliques, and a K-K edge
        # can be in multiple Type-B cliques? No, "every edge must lie in exactly one listed clique."
        #
        # OK so I think the issue is that my assumption about forced Type-B cliques is correct,
        # and the double-coverage issue means that for these specific graphs, no K-K edge is shared
        # by two N_K(i) sets in a way that creates a conflict. Or perhaps I need to be more careful.
        #
        # Let me just implement it and check for conflicts. If there's a conflict, I'll need a different approach.

        # Active I vertices
        active_I = [i for i in range(n - k) if len(cross_adj[i]) > 0]

        # Form Type-B cliques
        type_b_cliques = []
        covered_kk_edges = set()
        conflict = False

        for i in active_I:
            S = sorted(cross_adj[i])
            clique = [i + k] + S
            # Check K-K edges within S
            for a in range(len(S)):
                for b in range(a + 1, len(S)):
                    e = (min(S[a], S[b]), max(S[a], S[b]))
                    if e in covered_kk_edges:
                        conflict = True
                        break
                    covered_kk_edges.add(e)
                if conflict:
                    break
            if conflict:
                break
            type_b_cliques.append(clique)

        if conflict:
            # Fallback: use a simpler approach
            # Just use edge cliques
            return [[u, v] for u, v in edges]

        # Remaining K-K edges
        remaining_kk = []
        for u, v in kk_edges:
            if (u, v) not in covered_kk_edges:
                remaining_kk.append((u, v))

        # Cover remaining K-K edges with Type-A cliques
        # This is a minimum clique cover of the graph (K, remaining_kk)
        # Which is equivalent to minimum coloring of the complement graph
        
        if remaining_kk:
            involved = set()
            for u, v in remaining_kk:
                involved.add(u)
                involved.add(v)
            
            if involved:
                v_list = sorted(involved)
                v_map = {v: i for i, v in enumerate(v_list)}
                m = len(v_list)
                
                # Build complement graph for coloring
                # Edge in complement if NOT in remaining_kk
                rem_set = set(remaining_kk)
                comp_adj = [set() for _ in range(m)]
                for i in range(m):
                    for j in range(i + 1, m):
                        u, v = v_list[i], v_list[j]
                        e = (min(u, v), max(u, v))
                        if e not in rem_set:
                            comp_adj[i].add(j)
                            comp_adj[j].add(i)
                
                # Greedy coloring
                color_assign = [-1] * m
                for i in range(m):
                    # Find available colors
                    used_colors = set()
                    for nb in comp_adj[i]:
                        if color_assign[nb] >= 0:
                            used_colors.add(color_assign[nb])
                    c = 0
                    while c in used_colors:
                        c += 1
                    color_assign[i] = c
                
                # Group by color
                cliques_by_color = {}
                for i in range(m):
                    c = color_assign[i]
                    if c not in cliques_by_color:
                        cliques_by_color[c] = []
                    cliques_by_color[c].append(v_list[i])
                
                type_a_cliques = []
                for c, verts in cliques_by_color.items():
                    if len(verts) >= 2:
                        type_a_cliques.append(sorted(verts))

        else:
            type_a_cliques = []

        all_cliques = type_b_cliques + type_a_cliques

        # Validate and clean up
        valid_cliques = []
        for c in all_cliques:
            clean_c = sorted(set(int(x) for x in c))
            if len(clean_c) < 2:
                continue
            # Check it's a clique
            is_clique = True
            for i in range(len(clean_c)):
                for j in range(i + 1, len(clean_c)):
                    e = (min(clean_c[i], clean_c[j]), max(clean_c[i], clean_c[j]))
                    if e not in edge_set:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                valid_cliques.append(clean_c)

        if not valid_cliques:
            return [[u, v] for u, v in edges]

        return valid_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        cliques = solve_one(*spec)
        clean_cliques = []
        for c in cliques:
            if len(c) >= 2:
                clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)

    return results
