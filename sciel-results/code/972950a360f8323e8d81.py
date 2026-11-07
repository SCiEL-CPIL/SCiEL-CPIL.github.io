"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        if time.time() > deadline:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]

        edges = instance(n, k, p, seed)
        if not edges:
            return []

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Split graph structure: C = {0,...,k-1} is a clique, I = {k,...,n-1} is independent
        C = list(range(k))
        I = list(range(k, n))

        # For each v in I, N(v) = adj[v] ∩ C
        N = {v: adj[v] & set(C) for v in I}

        # Strategy:
        # 1. For each v in I, partition N(v) into blocks such that:
        #    - Each block B_j gives clique B_j ∪ {v}
        #    - C-edges within B_j are covered by this clique
        #    - No C-edge is covered by more than one clique (disjointness)
        # 2. Cover remaining C-edges with additional cliques (size-2 or larger)

        # Key insight: A C-edge (u,w) is "consumed" if there exists v in I such that
        # both u and w are in the same block of N(v).
        # We want to maximize consumed C-edges (each consumes one clique that would be needed separately)
        # while minimizing the number of blocks.

        # Greedy approach:
        # For each v in I, try to form large blocks from N(v) to consume C-edges.
        # Process I vertices in order of |N(v)| descending (more neighbors = more potential savings).
        # Use a conflict-aware greedy: when adding a vertex to a block, check if any C-edge
        # formed would conflict with an already-consumed edge.

        # Track which C-edges are consumed
        consumed = set()  # set of (u,w) pairs with u<w that are consumed by I-vertex cliques

        # blocks[v] = list of sets (each set is a subset of N(v))
        blocks = {v: [] for v in I}

        # Process I vertices in order of |N(v)| descending
        I_sorted = sorted(I, key=lambda v: len(N[v]), reverse=True)

        for v in I_sorted:
            if time.time() > deadline - 5:
                break
            neighbors = list(N[v])
            if not neighbors:
                continue

            # Greedily form blocks from neighbors
            # Try to make one large block first, then split if conflicts arise
            # Actually, let's use a simpler approach:
            # Try to include all neighbors in one block. If any C-edge conflicts, remove the offending vertex.
            
            # Start with all neighbors in one candidate block
            candidate = set(neighbors)
            
            # Check for conflicts: for each pair (u,w) in candidate, if (u,w) is already consumed, conflict
            # We need to remove vertices to resolve conflicts.
            # Greedy: iteratively remove the vertex that is in the most conflicting pairs.
            
            changed = True
            while changed and candidate:
                changed = False
                # Find all conflicting pairs
                conflict_vertices = {}  # vertex -> count of conflicts
                for u, w in combinations(sorted(candidate), 2):
                    e = (min(u,w), max(u,w))
                    if e in consumed:
                        # Conflict: this edge is already consumed
                        conflict_vertices[u] = conflict_vertices.get(u, 0) + 1
                        conflict_vertices[w] = conflict_vertices.get(w, 0) + 1
                
                if conflict_vertices:
                    # Remove the vertex with the most conflicts
                    worst = max(conflict_vertices, key=lambda x: conflict_vertices[x])
                    candidate.discard(worst)
                    changed = True

            if candidate:
                # Add this block and mark all C-edges within it as consumed
                for u, w in combinations(sorted(candidate), 2):
                    e = (min(u,w), max(u,w))
                    consumed.add(e)
                blocks[v].append(candidate)
            else:
                # No valid block possible with the full set, try singletons
                # Actually, singletons always work (no C-edges within a singleton)
                # But we should still try to group non-conflicting pairs
                # For simplicity, just use singletons if the full set failed
                for u in neighbors:
                    blocks[v].append({u})
                    # No C-edges to consume from singletons

        # Now, for vertices in I that we didn't process due to time, use singletons
        for v in I:
            if not blocks[v]:
                for u in N[v]:
                    blocks[v].append({u})

        # Build the list of cliques
        cliques = []

        # 1. Cliques involving I vertices
        for v in I:
            for S in blocks[v]:
                if S:
                    clique = list(S) + [v]
                    cliques.append(clique)

        # 2. Cover remaining C-edges (those not in `consumed`)
        # Find all C-edges
        all_cc_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_cc_edges.add((u, w))

        remaining = all_cc_edges - consumed

        # Cover remaining C-edges. Since C is a clique, any subset is a clique.
        # To minimize number of cliques, we want large cliques.
        # But each edge must be in exactly one clique.
        # Greedy: repeatedly pick the largest clique that covers the most remaining edges.
        # For a clique T ⊆ C, the number of remaining edges it covers is
        # |{(u,w) : u,w in T} ∩ remaining|.
        
        # Since we just need to cover edges with disjoint cliques, and C is a clique,
        # we can use a greedy set cover on the remaining edges.
        
        # Simple approach: just use size-2 cliques for remaining edges.
        # This is suboptimal but valid. Let's try to do slightly better.
        
        # Try to find large cliques that cover many remaining edges.
        # Greedy: while remaining is non-empty, find the largest subset T of C
        # such that all C-edges within T are in remaining (i.e., no edge in T is already covered
        # by another clique we're about to add). Wait, that's not right.
        
        # Actually, for the remaining C-edges, we just need to partition them into cliques.
        # Since C is a clique, any subset is a valid clique. The constraint is that each edge
        # appears in exactly one clique.
        
        # Greedy: pick a vertex u with the most remaining edges, form a clique with all its
        # remaining neighbors, remove those edges, repeat.
        
        remaining_set = set(remaining)
        while remaining_set:
            if time.time() > deadline - 2:
                # Fallback: size-2 cliques
                for e in remaining_set:
                    cliques.append([e[0], e[1]])
                break
            
            # Find vertex with most remaining edges
            edge_count = {}
            for u, w in remaining_set:
                edge_count[u] = edge_count.get(u, 0) + 1
                edge_count[w] = edge_count.get(w, 0) + 1
            
            if not edge_count:
                break
            
            u = max(edge_count, key=lambda x: edge_count[x])
            
            # Find all neighbors of u in the remaining graph
            neighbors_of_u = set()
            for e in remaining_set:
                if u in e:
                    v = e[0] if e[1] == u else e[1]
                    neighbors_of_u.add(v)
            
            # Form clique {u} ∪ neighbors_of_u
            clique = list(neighbors_of_u) + [u]
            cliques.append(clique)
            
            # Remove all edges within this clique from remaining_set
            clique_set = set(clique)
            new_remaining = set()
            for e in remaining_set:
                if e[0] not in clique_set or e[1] not in clique_set:
                    new_remaining.add(e)
            remaining_set = new_remaining

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)

    return results
# EVOLVE-BLOCK-END
