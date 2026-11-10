"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build cross adjacency: for each I-vertex, which C-vertices it connects to
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # All C-edges
        C_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        
        # For each C-edge, find which I-vertices are adjacent to both endpoints
        edge_to_is = defaultdict(set)
        for i in range(k, n):
            if len(N[i]) < 2:
                continue
            Ni = N[i]
            for u in Ni:
                for v in Ni:
                    if u < v:
                        edge_to_is[(u, v)].add(i)
        
        # Strategy: For each I-vertex, we must cover all cross edges from it.
        # A clique {i} ∪ S covers cross edges (i,s) for s in S, and C-edges within S.
        # We want to minimize total cliques.
        
        # Approach: Greedy with matching-based C-edge packing.
        # 1. For each I-vertex i, we need at least 1 clique if N[i] non-empty.
        # 2. The clique {i} ∪ N[i] covers all cross edges from i and all C-edges within N[i].
        # 3. Remaining C-edges (not in any N[i]) must be covered by pure-C cliques.
        # 4. But we can do better: if a C-edge is in multiple N[i]'s, we can choose
        #    which i's clique covers it, potentially allowing smaller cliques for others.
        
        # Better approach: Use maximum matching to assign C-edges to I-vertex cliques.
        # Each C-edge (u,v) can be "covered" by any i where u,v ∈ N[i].
        # We want to maximize the number of C-edges covered by I-vertex cliques
        # to minimize pure-C cliques needed.
        
        # But actually, every C-edge within N[i] is automatically covered if we
        # use clique {i} ∪ N[i]. The question is whether we can use smaller cliques
        # for some i's to leave more C-edges for other i's or pure-C cliques.
        
        # Key insight: The number of cliques containing i must be at least the
        # minimum number of subsets of N[i] that cover N[i], which is 1.
        # So each non-isolated I-vertex needs exactly 1 clique: {i} ∪ N[i].
        # This covers all C-edges within N[i].
        # Remaining C-edges are those not contained in any N[i].
        
        # Wait, that's not quite right. We could split N[i] into multiple cliques
        # if it helps cover more C-edges overall, but that increases clique count.
        # So it's always optimal to use {i} ∪ N[i] for each i? Not necessarily,
        # because a C-edge (u,v) might be in N[i1] and N[i2], and if we use
        # {i1} ∪ N[i1] and {i2} ∪ N[i2], the C-edge (u,v) is in two cliques,
        # which is invalid!
        
        # CRITICAL: Each edge must be in EXACTLY ONE clique. So we cannot have
        # a C-edge in two different I-vertex cliques.
        
        # This changes everything. We need to partition C-edges among:
        # - Cliques containing I-vertices
        # - Pure-C cliques
        # such that no C-edge appears in two cliques.
        
        # For each I-vertex i, if we use clique {i} ∪ S where S ⊆ N[i], it covers:
        # - Cross edges (i, s) for s in S
        # - C-edges within S
        # All cross edges from i must be covered, so ∪ S_j = N[i] for the cliques
        # containing i.
        
        # This is a complex set partition problem. Let's use a greedy approach:
        # Process I-vertices in order of decreasing |N[i]|.
        # For each i, pick S = N[i] (one clique), mark all C-edges within S as covered.
        # Remaining C-edges go to pure-C cliques.
        
        # But we can do better: if some C-edges in N[i] are also in N[j] for j > i,
        # we might want to exclude them from i's clique and let j cover them.
        
        # Let's try multiple strategies and pick the best.
        
        best_cliques = None
        best_count = float('inf')
        
        I_vertices = [i for i in range(k, n) if N[i]]
        
        # Strategy 1: Process in decreasing |N[i]|, use full N[i]
        # Strategy 2: Process in increasing |N[i]|, use full N[i]
        # Strategy 3: Random orders
        # Strategy 4: For each i, try to exclude C-edges that are "shared" with later vertices
        
        strategies = []
        strategies.append(sorted(I_vertices, key=lambda i: len(N[i]), reverse=True))
        strategies.append(sorted(I_vertices, key=lambda i: len(N[i])))
        
        rng = random.Random(seed)
        for _ in range(30):
            strategies.append(rng.sample(I_vertices, len(I_vertices)))
        
        for order in strategies:
            if time.time() > deadline:
                break
            
            covered_C_edges = set()
            cliques = []
            
            for i in order:
                if time.time() > deadline:
                    break
                Ni = set(N[i])
                if not Ni:
                    continue
                
                # Find C-edges within Ni that are already covered
                Ni_C_edges = set((u, v) for u in Ni for v in Ni if u < v)
                already_covered = Ni_C_edges & covered_C_edges
                
                # We must cover all cross edges from i, so we need cliques whose
                # S-parts union to Ni. Simplest: one clique {i} ∪ Ni, but this
                # covers C-edges within Ni that may already be covered (invalid!).
                
                # So we need to partition Ni into subsets S1, S2, ... such that:
                # - No C-edge within Sj is already covered
                # - Every vertex in Ni is in some Sj
                # Minimize number of subsets.
                
                # Greedy: Build subsets avoiding already-covered C-edges.
                remaining = set(Ni)
                subsets = []
                
                while remaining:
                    S = set()
                    for u in sorted(remaining):
                        # Check if adding u to S creates a covered C-edge
                        ok = True
                        for v in S:
                            e = (min(u, v), max(u, v))
                            if e in covered_C_edges:
                                ok = False
                                break
                        if ok:
                            S.add(u)
                    
                    if not S:
                        S = {next(iter(remaining))}
                    
                    subsets.append(S)
                    remaining -= S
                
                for S in subsets:
                    clique = sorted([i] + list(S))
                    cliques.append(clique)
                    S_list = sorted(S)
                    for a_idx in range(len(S_list)):
                        for b_idx in range(a_idx + 1, len(S_list)):
                            covered_C_edges.add((S_list[a_idx], S_list[b_idx]))
            
            # Cover remaining C-edges with pure-C cliques
            remaining_C_edges = C_edges - covered_C_edges
            
            if remaining_C_edges:
                # Build graph of remaining C-edges
                rem_adj = defaultdict(set)
                for u, v in remaining_C_edges:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                
                while remaining_C_edges:
                    if time.time() > deadline:
                        break
                    
                    # Find max clique in remaining graph (it's a subgraph of a clique,
                    # so the whole remaining graph is a clique! All vertices in rem_adj
                    # are mutually adjacent in the original C, and we only removed
                    # some edges. Wait, no - remaining_C_edges is a subset of C-edges,
                    # and C is a clique, so any subset of C-edges induces a subgraph
                    # that is NOT necessarily a clique. Two vertices u,v in rem_adj
                    # are adjacent in rem_adj iff (u,v) is in remaining_C_edges.
                    
                    # So we need to find cliques in this subgraph.
                    # Greedy: pick vertex with max degree, add all its neighbors.
                    if not rem_adj:
                        break
                    
                    # Find vertex with max degree
                    max_v = max(rem_adj.keys(), key=lambda v: len(rem_adj[v]))
                    S = {max_v} | rem_adj[max_v]
                    S_list = sorted(S)
                    cliques.append(S_list)
                    
                    for a_idx in range(len(S_list)):
                        for b_idx in range(a_idx + 1, len(S_list)):
                            remaining_C_edges.discard((S_list[a_idx], S_list[b_idx]))
                    
                    for node in S_list:
                        rem_adj.pop(node, None)
                        for w in list(rem_adj.keys()):
                            rem_adj[w].discard(node)
                    rem_adj = {u2: v2 for u2, v2 in rem_adj.items() if v2}
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
            
            if time.time() > deadline - 5:
                break
        
        if best_cliques is None:
            return [list(e) for e in edges]
        
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
