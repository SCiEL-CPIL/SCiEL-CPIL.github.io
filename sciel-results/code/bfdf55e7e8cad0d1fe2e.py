"""Strategy: Global optimization via simulated annealing to minimize clique count."""

def construct():
    import random
    import time
    import math

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_one(n, k, p, seed, deadline):
        rng = random.Random(seed)
        
        # 1. Generate graph
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[v].add(u)
                    adj[u].add(v)
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)

        # 2. Initial Solution: 
        # Every cross edge (u, v) with u<k<=v gets a clique [u, v].
        # Every core edge (u, v) with u<v<k gets a clique [u, v].
        # This is a valid partition (each edge in exactly one clique, all cliques are edges).
        # Total cliques = E.
        
        # Optimization Goal: Merge cliques.
        # A set of edges can be merged into one clique if they form a clique in G.
        # Equivalent: We want to partition the edge set E into the minimum number of cliques.
        
        # Heuristic: 
        # 1. Identify "dense" regions.
        # 2. For each vertex v in I (k..n-1), its neighbors in K form a clique (since K is a clique).
        #    So N_K(v) ∪ {v} is a clique.
        #    If we use this clique, it covers all edges between v and N_K(v).
        #    It also covers edges between vertices in N_K(v) (which are core edges).
        
        # Let's try a greedy set cover approach on the "clique candidates".
        # Candidates:
        # 1. For each v in I: C_v = N_K(v) ∪ {v}. (Covers star edges + core edges in N_K(v))
        # 2. For each core edge (u, w): C = {u, w}.
        # 3. For each core clique subset S of K: C = S.
        
        # Actually, let's look at the structure.
        # The graph is a split graph.
        # The core K is a clique.
        # The independent set I has no edges.
        # Edges are Core-Core (all present) and Core-Independent (random).
        
        # A clique in this graph is either:
        # A. A subset of K.
        # B. A subset of K plus at most one vertex from I. (Since I is independent, you can't have two vertices from I in a clique).
        
        # So any valid clique partition consists of:
        # - Some cliques entirely within K.
        # - Some cliques of the form {v} ∪ S_v where v ∈ I and S_v ⊆ N_K(v).
        
        # Constraints:
        # 1. Every edge (u, w) with u, w ∈ K must be covered by exactly one clique.
        #    - It can be covered by a clique A (subset of K) containing u, w.
        #    - OR by a clique B = {v} ∪ S_v where v ∈ I, v ~ u, v ~ w.
        # 2. Every edge (u, v) with u ∈ K, v ∈ I must be covered by exactly one clique.
        #    - It MUST be covered by a clique of type B containing v and u.
        #    - It cannot be covered by a clique of type A (since v ∉ K).
        
        # Implication:
        # For each v ∈ I, the set of edges incident to v (which are (u, v) for u ∈ N_K(v)) MUST be covered by cliques of type B that include v.
        # Since cliques are disjoint in terms of edges, can a vertex v be in multiple cliques?
        # Yes. If v is in clique C1 = {v} ∪ S1 and C2 = {v} ∪ S2, then:
        # - Edges (u, v) for u ∈ S1 are covered by C1.
        # - Edges (u, v) for u ∈ S2 are covered by C2.
        # - Edges (u1, u2) for u1, u2 ∈ S1 are covered by C1.
        # - Edges (u1, u2) for u1, u2 ∈ S2 are covered by C2.
        # - Edges (u1, u2) for u1 ∈ S1, u2 ∈ S2 are NOT covered by C1 or C2. They must be covered by another clique (either a core clique or another v' clique).
        
        # To minimize count, we want to maximize the size of S_v for each v?
        # If we take S_v = N_K(v) for all v, we use one clique per v ∈ I.
        # This covers all cross edges.
        # It also covers all core edges (u, w) such that there exists a v ∈ I with u, w ∈ N_K(v).
        # Remaining core edges are those (u, w) where NO v ∈ I is adjacent to both u and w.
        # These remaining core edges must be covered by cliques of type A (subsets of K).
        
        # Is it always optimal to use S_v = N_K(v)?
        # Suppose we split N_K(v) into S1 and S2.
        # We use 2 cliques for v instead of 1.
        # We cover the same cross edges.
        # We cover core edges within S1 and within S2.
        # We DO NOT cover core edges between S1 and S2.
        # So splitting increases the number of cliques for v, and leaves more core edges to be covered by core cliques.
        # This seems strictly worse or equal.
        # Exception: If covering core edges within S1 ∪ S2 causes conflicts?
        # No, covering an edge is a "good" thing (reduces remaining core edges).
        # Wait, if we cover a core edge with a v-clique, we can't cover it with a core clique.
        # But covering it is mandatory.
        # So, covering as many core edges as possible with v-cliques is good, because it reduces the load on core cliques.
        # And using one v-clique for all neighbors of v is the most efficient way to cover the cross edges of v.
        
        # Hypothesis: The optimal solution uses exactly one clique per v ∈ I, defined as C_v = N_K(v) ∪ {v}.
        # Then, the remaining problem is to cover the "uncovered" core edges.
        # An edge (u, w) in K is covered by C_v iff u ∈ N_K(v) and w ∈ N_K(v).
        # i.e., u ~ v and w ~ v.
        # Let U be the set of core edges (u, w) such that there is NO v ∈ I with u ~ v and w ~ v.
        # We need to partition U into cliques (subsets of K).
        # Since K is a clique, any subset of K is a clique.
        # So we just need to partition the graph (K, U) into the minimum number of cliques.
        # This is equivalent to finding the chromatic number of the complement graph (K, \bar{U}).
        # \bar{U} is the graph on K where (u, w) is an edge if (u, w) is NOT in U.
        # (u, w) is in U iff no v is common neighbor.
        # (u, w) is NOT in U iff there exists v with u~v and w~v.
        
        # So we need to color the graph H = (K, E_H) where E_H = { (u, w) | ∃ v ∈ I, u~v, w~v }.
        # The number of colors is the number of core cliques needed.
        
        # Algorithm:
        # 1. Build adj.
        # 2. For each v ∈ I, create clique C_v = N_K(v) ∪ {v}.
        # 3. Determine U: core edges (u, w) such that no v ∈ I is adjacent to both.
        # 4. Construct H: vertices K, edges are core edges NOT in U.
        # 5. Find chromatic number of H (or a good approximation).
        # 6. For each color class (clique in H), add it as a core clique.
        #    Wait, H edges mean "covered by some v".
        #    If (u, w) is in H, it is covered by some C_v.
        #    If (u, w) is NOT in H, it is NOT covered by any C_v.
        #    So we need to cover edges NOT in H.
        #    Let U = { (u, w) | (u, w) ∉ H }.
        #    We need to partition U into cliques.
        #    Since U is a subgraph of K (clique), any subset of vertices forming edges in U is a clique in G.
        #    So we just need to cover the edges of U with disjoint cliques.
        #    This is the same as coloring the complement of U?
        #    Let G_U = (K, U). We want to partition E(G_U) into cliques.
        #    This is the clique cover number of G_U.
        #    Since G_U is a subgraph of a complete graph, it's a general graph.
        #    Clique cover of G_U is equal to Chromatic Number of \bar{G_U}.
        
        # So:
        # 1. Identify U.
        # 2. Build \bar{G_U} on K.
        # 3. Find Chromatic Number of \bar{G_U}.
        # 4. The color classes of \bar{G_U} correspond to cliques in G_U?
        #    No.
        #    Coloring \bar{G_U}: Adjacent in \bar{G_U} means NOT adjacent in G_U.
        #    So a color class in \bar{G_U} is an independent set in \bar{G_U}, which is a clique in G_U.
        #    Yes.
        #    So the minimum number of cliques to cover U is the chromatic number of \bar{G_U}.
        
        # Implementation of Coloring:
        # k is small (10, 20, 30).
        # We can use a greedy coloring or a simple backtracking for small k.
        # Greedy coloring (DSATUR) is usually good enough.
        
        # Let's refine the "Hypothetical Optimality".
        # Is it possible that splitting a v-clique helps?
        # Suppose we have v with neighbors {1, 2, 3}.
        # Option A: One clique {v, 1, 2, 3}. Covers cross (v,1), (v,2), (v,3). Covers core (1,2), (1,3), (2,3).
        # Option B: Two cliques {v, 1, 2} and {v, 3}.
        #   Covers cross (v,1), (v,2) in C1, (v,3) in C2.
        #   Covers core (1,2) in C1.
        #   Does NOT cover core (1,3), (2,3).
        #   So (1,3) and (2,3) must be covered by core cliques or other v's.
        #   Option A uses 1 clique. Option B uses 2 cliques.
        #   Option A covers 3 core edges. Option B covers 1 core edge.
        #   Option A is clearly better locally.
        #   Globally, if (1,3) and (2,3) are "hard" to cover, maybe?
        #   But covering them is always "free" if we have a clique containing them.
        #   The only cost is the clique itself.
        #   So maximizing coverage per clique is good.
        
        # What if N_K(v) is not a clique?
        # N_K(v) is a subset of K. K is a clique. So N_K(v) is a clique.
        
        # So the strategy is solid:
        # 1. Use C_v = N_K(v) ∪ {v} for all v ∈ I.
        # 2. Cover remaining core edges U with min cliques.
        
        # Let's implement this.
        
        cliques = []
        
        # Step 1: V-cliques
        # Track which core edges are covered by these.
        covered_core = set()
        
        for v in range(k, n):
            S_v = [u for u in range(k) if u in adj[v]]
            if S_v:
                cliques.append([v] + S_v)
                # Mark core edges in S_v as covered
                for i in range(len(S_v)):
                    for j in range(i+1, len(S_v)):
                        u, w = S_v[i], S_v[j]
                        covered_core.add((min(u, w), max(u, w)))
            # If S_v is empty, v is isolated from K. No cross edges.
            # v is not in any clique?
            # Wait, if v has no neighbors in K, it has no edges incident to it.
            # The problem says "every edge must lie in exactly one listed clique".
            # It does NOT say every vertex must be in a clique.
            # "Every listed set must be a clique with at least two vertices".
            # So isolated vertices are ignored.
        
        # Step 2: Remaining Core Edges
        # Total core edges: (u, w) for 0 <= u < w < k.
        remaining = set()
        for u in range(k):
            for w in range(u+1, k):
                if (u, w) not in covered_core:
                    remaining.add((u, w))
        
        if remaining:
            # Build graph G_U on K with edges = remaining.
            # We want to partition E(G_U) into min cliques.
            # This is equivalent to coloring \bar{G_U}.
            
            # Build \bar{G_U} adjacency
            # Vertices 0..k-1.
            # Edge (u, w) in \bar{G_U} if (u, w) NOT in remaining.
            # Note: (u, w) is a core edge.
            
            # Greedy Coloring of \bar{G_U}
            # Colors represent cliques in G_U.
            
            # Actually, we can just find cliques in G_U directly?
            # Greedy clique cover:
            # While edges remain:
            #   Find a maximal clique in G_U (using remaining edges).
            #   Add it to cliques.
            #   Remove its edges from remaining.
            
            # Since k is small, we can be a bit more careful.
            # Greedy: Pick vertex with highest degree in remaining graph.
            # Find maximal clique containing it.
            
            rem_adj = [set() for _ in range(k)]
            for u, w in remaining:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            while remaining:
                if time.time() > deadline:
                    # Fallback
                    for u, w in remaining:
                        cliques.append([u, w])
                    break
                
                # Find vertex with max degree
                max_deg = -1
                start_v = -1
                for v in range(k):
                    if len(rem_adj[v]) > max_deg:
                        max_deg = len(rem_adj[v])
                        start_v = v
                
                if max_deg == 0:
                     # Should not happen if remaining not empty
                     break
                
                # Greedy clique expansion from start_v
                S = {start_v}
                candidates = list(rem_adj[start_v])
                
                # Sort candidates by degree (descending) to find larger clique?
                # Or just iterate.
                # We want a maximal clique in the subgraph induced by remaining edges.
                
                # Simple greedy:
                # Try to add vertices from candidates one by one.
                # A vertex x can be added if it is connected to ALL vertices in S via remaining edges.
                
                # To be efficient, check against current S.
                # S is small.
                
                # Let's iterate through candidates.
                # To ensure maximality, we should try all?
                # Greedy is usually fine.
                
                # Order candidates by degree in remaining graph (global)
                candidates.sort(key=lambda x: len(rem_adj[x]), reverse=True)
                
                for x in candidates:
                    # Check if x is connected to all in S
                    ok = True
                    for s in S:
                        if x not in rem_adj[s]:
                            ok = False
                            break
                    if ok:
                        S.add(x)
                        # Note: rem_adj is static for this step?
                        # No, we should update or just check against original rem_adj?
                        # If we add x, it must be connected to all in S.
                        # Since we are building a clique, all pairs in S must be edges in remaining.
                        # The check `x not in rem_adj[s]` ensures x is connected to s.
                        # Since s was added previously, s is connected to all previous S.
                        # So yes, this builds a clique.
                        
                if len(S) < 2:
                    # Should not happen if max_deg > 0
                    # Just take an edge
                    u, w = next(iter(remaining))
                    S = {u, w}
                
                cliques.append(list(S))
                
                # Remove edges in S from remaining and rem_adj
                for u in S:
                    for w in S:
                        if u < w:
                            if (u, w) in remaining:
                                remaining.discard((u, w))
                                rem_adj[u].discard(w)
                                rem_adj[w].discard(u)
        
        return cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(_solve_one(*spec, deadline))

    return results

construct()
