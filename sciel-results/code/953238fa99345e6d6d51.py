import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        # Build edge set
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in range(k):
                if u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))

        # We need to cover all edges with cliques.
        # Strategy: Greedy set cover on edges, where each candidate is a maximal clique.
        # But that's expensive. Instead, use a smarter approach for split graphs.
        
        # For split graphs, the clique C (first k vertices) is a clique.
        # I vertices (k to n-1) form an independent set.
        # Edges are: all within C, and between C and I with prob p.
        
        # Key insight: Each clique in the partition can contain at most one I vertex
        # (since I is independent). So each clique is either:
        #   - A subset of C (a clique within C)
        #   - A subset of C plus one I vertex v, where all vertices in the C subset
        #     are neighbors of v.
        
        # To minimize cliques, we want to maximize the size of each clique.
        # For an I vertex v, the best clique containing v is {v} ∪ N(v) ∩ C.
        # But we also need to cover edges within C.
        
        # Approach: 
        # 1. For each I vertex v, consider the clique {v} ∪ S where S ⊆ N(v) ∩ C.
        #    This covers all edges (v, u) for u in S, and all edges within S.
        # 2. We need to choose S for each v such that:
        #    - Every edge (v, u) is covered by exactly one clique (the one for v that includes u)
        #    - Every edge within C is covered by exactly one clique
        #    - No edge is covered by two cliques
        
        # This is equivalent to: partition the edges into cliques.
        # For I vertex v, the edges incident to v are (v, u) for u in N(v) ∩ C.
        # These must all be in the same clique (since v can only be in one clique per edge... 
        # wait, no. v can be in multiple cliques, but each edge must be in exactly one clique).
        # Actually, v can appear in multiple cliques. Each clique containing v covers 
        # some subset of edges (v, u). But v can only be in cliques where all other vertices
        # are neighbors of v.
        
        # Simpler approach: greedy.
        # Process edges one by one, assign each to a clique.
        # Use a priority: prefer to put edges into existing cliques if possible.
        
        # Better: Since C is a clique, any subset of C is a clique.
        # For I vertex v, {v} ∪ S is a clique iff S ⊆ N(v) ∩ C.
        
        # Let's use a different strategy:
        # 1. For each I vertex v, create a clique {v} ∪ N(v) ∩ C. This covers all edges 
        #    incident to v and all edges within N(v) ∩ C.
        # 2. The remaining edges are those within C that are not in any N(v) ∩ C for any v.
        #    Wait, that's not right. Edges within C are covered if both endpoints are in 
        #    the same clique. If we put all of N(v) ∩ C in one clique with v, then edges 
        #    within N(v) ∩ C are covered by that clique.
        # 3. But edges within C that involve vertices not in any N(v) ∩ C need separate cliques.
        #    Also, if two I vertices v1, v2 share a neighbor u in C, the edge (u, v1) is in 
        #    clique1 and (u, v2) is in clique2, which is fine. But the edge (u, w) for w in C 
        #    might be covered by clique1 if w is in N(v1) ∩ C, and by clique2 if w is in N(v2) ∩ C.
        #    That would be a conflict!
        
        # So we need to ensure that each edge within C is covered by at most one clique.
        # This means: for any two I vertices v1, v2, N(v1) ∩ C and N(v2) ∩ C should not 
        # share any edge, i.e., |N(v1) ∩ N(v2) ∩ C| ≤ 1 (since an edge needs two endpoints).
        
        # This is getting complex. Let's use a greedy edge-covering approach.
        
        # Greedy: 
        # 1. Start with all edges uncovered.
        # 2. Repeatedly find a maximal clique in the remaining edge graph, add it to the partition,
        #    and remove all its edges from the remaining set.
        # 3. To find a good clique, use a heuristic: pick an edge, then try to extend it to a 
        #    maximal clique.
        
        # For split graphs, maximal cliques are of the form {v} ∪ S where v is an I vertex 
        # and S ⊆ N(v) ∩ C, or subsets of C.
        
        # Let's implement a greedy algorithm:
        # - Maintain a set of uncovered edges.
        # - While there are uncovered edges:
        #   - Pick an uncovered edge (u, v).
        #   - If both u, v are in C: try to extend to a maximal clique within C using only 
        #     uncovered edges.
        #   - If one is in I (say v) and one in C (say u): the clique must contain v and u, 
        #     and can be extended to {v} ∪ S where S ⊆ N(v) ∩ C and all edges in S are uncovered 
        #     and connect to u.
        
        # This is still complex. Let's try a simpler but effective approach:
        
        # For each I vertex v, the edges incident to v must be covered. Since v is in I, 
        # any clique containing v can only have other vertices from N(v) ∩ C.
        # So for v, we need to partition N(v) ∩ C into subsets S1, S2, ..., Sm such that 
        # each {v} ∪ Si is a clique (which it is, since Si ⊆ N(v) ∩ C ⊆ C which is a clique).
        # But we also need to cover edges within C.
        
        # Key: if we put all of N(v) ∩ C in one clique with v, we cover all edges (v, u) for u in N(v) ∩ C,
        # and all edges within N(v) ∩ C. But this might conflict with other I vertices.
        
        # Let's just use a greedy maximal clique extraction:
        
        remaining = set(all_edges)
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            # Pick an edge from remaining
            u, v = next(iter(remaining))
            
            # Try to build a maximal clique containing this edge
            # Start with {u, v}
            clique = {u, v}
            
            # Try to add more vertices
            # For a vertex x to be added, x must be connected to all vertices in clique
            # and all edges (x, y) for y in clique must be in remaining
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in clique:
                        continue
                    can_add = True
                    for y in clique:
                        if (min(x, y), max(x, y)) not in remaining:
                            can_add = False
                            break
                    if can_add:
                        clique.add(x)
                        changed = True
            
            # Add this clique
            cliques.append(sorted(clique))
            
            # Remove all edges within this clique from remaining
            for u1, u2 in combinations(sorted(clique), 2):
                remaining.discard((min(u1, u2), max(u1, u2)))
        
        # Safety: check for uncovered edges
        covered = set()
        for cl in cliques:
            for u1, u2 in combinations(cl, 2):
                covered.add((min(u1, u2), max(u1, u2)))
        
        uncovered = all_edges - covered
        while uncovered:
            u, v = next(iter(uncovered))
            clique = {u, v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in clique:
                        continue
                    can_add = True
                    for y in clique:
                        if (min(x, y), max(x, y)) not in uncovered:
                            can_add = False
                            break
                    if can_add:
                        clique.add(x)
                        changed = True
            cliques.append(sorted(clique))
            for u1, u2 in combinations(sorted(clique), 2):
                uncovered.discard((min(u1, u2), max(u1, u2)))
        
        # Ensure all cliques have at least 2 vertices and proper types
        final_cliques = []
        for cl in cliques:
            cl = [int(x) for x in cl]
            if len(cl) >= 2:
                final_cliques.append(cl)
        
        results.append(final_cliques)
    
    return results
