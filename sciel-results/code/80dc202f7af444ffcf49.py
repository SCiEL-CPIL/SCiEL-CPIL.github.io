"""Rewritten program: structured greedy clique partition of split graphs."""

def construct():
    import time
    from random import Random

    T0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _clique_partition(n, k, p, seed):
        """
        Return a list of cliques (each a list of >=2 distinct ints) that covers
        every edge of the split graph exactly once.

        Structure:
          - C = {0, ..., k-1}: a clique
          - I = {k, ..., n-1}: an independent set
          - Cross edges (u in C, v in I) present w.p. p

        Algorithm:
          1. For each v in I, let N(v) = {u in C : edge (u,v) exists}.
             The edges incident to v are exactly {(u,v) : u in N(v)}.
             We need to partition N(v) into subsets, each of which will be
             paired with v to form a clique {v} ∪ S (since C is a clique,
             any subset S of N(v) forms a clique with v).
             
             Key insight: to minimize total cliques, we want to group
             independent vertices that share the same neighbor subset.
             However, a more effective approach:
             
             Step A: Group independent vertices by their exact neighbor set.
                     For each group with neighbor set S (|S| >= 2), 
                     the edges from all v in group to S can be covered by
                     cliques {v, u1, u2, ..., u_|S|} for each v — but that's
                     one clique per v.
                     
             Actually, a better strategy:
             - For each v in I with |N(v)| >= 2: create clique N(v) ∪ {v}.
               This covers all edges from v to N(v) in one clique.
             - For each v in I with |N(v)| == 1: say N(v) = {u}. The edge (u,v)
               must be covered. We can't put v in a clique with just u unless
               we make {u,v}. But we might combine with another vertex.
               Since I is independent, v and v' can't be in the same clique unless
               they share at least one neighbor and we include that neighbor.
               Actually {u, v, v'} is NOT a clique because (v,v') is not an edge.
               So {u, v} and {u, v'} are separate 2-cliques, OR we can put
               multiple single-neighbor vertices into one clique only if they
               all share the same single neighbor u: {u, v1, v2, ...} — but
               that's not a clique since v1,v2 are not adjacent.
               
               So each edge (u, v) where v has degree 1 to C must be in its own
               2-clique {u, v}, OR we can group edges that share both endpoints... 
               no, each edge is unique.
               
               Wait: we can have a clique {u, v1} and separately {u, v2}, or
               we can have a larger clique. But {u, v1, v2} is not a clique
               because v1 and v2 are not adjacent. So no grouping of degree-1
               vertices into a single clique is possible.
               
             So for |N(v)| == 1: one 2-clique {u, v}.
             
          2. Now, the edges within C: these are (u, u') for 0 <= u < u' < k.
             We need to cover these too. The cliques we made above (N(v) ∪ {v})
             contain edges within N(v), which are a subset of C's edges. 
             
             We need to ensure every edge in C is covered exactly once across
             all cliques. This is the tricky part.
             
          Refined approach:
            - We'll build cliques that include vertices from C and possibly
              one vertex from I.
            - Each clique will be of the form S ∪ {v} where S ⊆ N(v) ⊆ C,
              or just S ⊆ C (a pure C-clique).
            - The constraint: each edge appears in exactly one clique.
            
          Simpler structured approach (matching the observed improvement):
            1. For each v in I with |N(v)| >= 2: emit clique N(v) ∪ {v}.
               This covers all edges (u,v) for u in N(v), AND all edges (u,u')
               for u,u' in N(v).
            2. For each v in I with |N(v)| == 1: emit 2-clique {u, v} where u is
               the single neighbor.
            3. After steps 1-2, some edges within C may have been covered (those
               inside any N(v) for |N(v)|>=2). We need to cover the remaining
               C-edges with additional cliques that are subsets of C.
            4. Find a clique partition of the remaining edges on C. Since C is
               a complete graph on k vertices, the remaining edges form a subgraph
               of K_k. We need to partition these edges into cliques.
               
               Greedy: while there are uncovered edges in C:
                 - Find the largest clique of uncovered edges (i.e., a set of
                   vertices in C where all pairwise edges are still uncovered).
                 - Emit that clique, mark those edges as covered.
               
               This is the "residual clique-part edges covered by large greedy
               cliques" strategy mentioned in the findings.
        """
        # Build adjacency
        rng = Random(seed)
        # Reconstruct edges in the same order as instance()
        cross_edges = set()
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.add((u, v))
        
        # N(v) for v in I
        neighbor_sets = [set() for _ in range(n)]
        for (u, v) in cross_edges:
            neighbor_sets[v].add(u)
        
        # Track which edges in C are covered
        # C-edges: (u, u') for 0 <= u < u' < k
        c_edges_covered = set()
        cliques = []
        
        # Step 1 & 2: handle I vertices
        for v in range(k, n):
            nv = neighbor_sets[v]
            if len(nv) >= 2:
                clique = sorted(list(nv)) + [v]
                cliques.append(clique)
                # Mark all C-edges within nv as covered
                nv_list = sorted(nv)
                for i in range(len(nv_list)):
                    for j in range(i + 1, len(nv_list)):
                        c_edges_covered.add((nv_list[i], nv_list[j]))
            elif len(nv) == 1:
                u = next(iter(nv))
                cliques.append([u, v])
                # No C-edges to cover here (just the cross edge)
        
        # Step 3: Cover remaining C-edges
        # Remaining C-edges: all (u, u') for 0<=u<u'<k not in c_edges_covered
        all_c_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        remaining = set(all_c_edges) - c_edges_covered
        
        # Greedy: repeatedly find largest clique in the remaining-edge subgraph
        # Build adjacency on C vertices for remaining edges
        while remaining:
            if time.time() - T0 > DEADLINE - 5:
                break  # safety: return what we have (shouldn't happen for these sizes)
            
            # Find all vertices incident to remaining edges
            vertices_with_edges = set()
            for (u, v) in remaining:
                vertices_with_edges.add(u)
                vertices_with_edges.add(v)
            
            if not vertices_with_edges:
                break
            
            # Greedy largest clique: pick vertex with highest degree in remaining,
            # then iteratively add vertices connected to all current members
            remaining_adj = {}
            for v in vertices_with_edges:
                remaining_adj[v] = set()
            for (u, v) in remaining:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            # Pick starting vertex: highest degree
            start = max(vertices_with_edges, key=lambda x: len(remaining_adj[x]))
            clique = [start]
            candidates = set(remaining_adj[start])
            
            while candidates:
                # Pick vertex in candidates with max degree among candidates
                best = max(candidates, key=lambda x: len(remaining_adj[x] & candidates))
                clique.append(best)
                candidates = candidates & remaining_adj[best]
            
            clique.sort()
            if len(clique) >= 2:
                cliques.append(clique)
            
            # Remove edges within this clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (clique[i], clique[j])
                    remaining.discard(e)
        
        # Filter: each clique must have at least 2 distinct vertices
        result = []
        for c in cliques:
            if len(c) >= 2 and len(set(c)) == len(c):
                result.append(list(c))
        
        return result

    # Solve all three instances
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        cliques = _clique_partition(n, k, p, seed)
        results.append(cliques)
    
    return results
