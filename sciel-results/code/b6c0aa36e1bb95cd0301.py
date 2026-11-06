"""
Clique partition via vertex-merging heuristic + local repair.
Strategy: greedily merge vertices into cliques by minimizing conflict with
already-assigned edges, then repair via edge-level local search.
"""
import time
import random
import sys
from itertools import combinations

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    if not cliques:
        return len(edge_list) == 0
    edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * len(edge_list)
    for clique in cliques:
        if len(clique) < 2:
            return False
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
                e = _edge_key(clique[a], clique[b])
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _greedy_vertex_merge(adj, edge_list, n):
    """
    Greedy vertex-merging: start with each vertex as its own group.
    Repeatedly merge the pair of vertices (u, v) that are adjacent and whose
    merge creates the fewest new "conflict edges" (edges from the merged group
    to vertices not in the group that are already assigned to other groups).
    We track which edges are "consumed" by which group.
    
    Actually, a cleaner formulation:
    - Maintain a partition of vertices into groups.
    - Each group must be a clique.
    - An edge (u,v) is "covered" if u and v are in the same group.
    - We need every edge to be covered by exactly one group (i.e., both endpoints in same group).
    - Non-edges must NOT have both endpoints in the same group.
    
    So this is: partition V into cliques such that every edge has both endpoints in the same clique.
    This means: if (u,v) is an edge, u and v MUST be in the same clique.
    This is actually a strong constraint! It means the partition is determined by connected components
    of the graph... wait no. If u-v is an edge, they must be in the same clique. By transitivity,
    all vertices in a connected component must be in the same clique. But then the clique must be
    the entire connected component, which must itself be a clique.
    
    Wait, let me re-read the problem. "Every edge must lie in exactly one listed clique."
    A clique is a set of vertices where every pair is connected. An edge "lies in" a clique if
    both endpoints are in that clique. So if edge (u,v) lies in clique C, then u,v ∈ C.
    
    So yes: for every edge (u,v), there exists exactly one clique C with u,v ∈ C.
    This means: if (u,v) is an edge, u and v must be in the same clique.
    By transitivity over edges, all vertices in the same connected component must be in the same clique.
    And that clique must be a complete subgraph (clique) on the connected component.
    
    But wait - the graph is dense random with p=0.6 or 0.55. For n=32, p=0.6, the graph is almost
    certainly connected and NOT a clique (since p<1). So there's no valid partition where every
    edge is in exactly one clique AND each clique is a complete graph...
    
    Unless I'm misreading. Let me re-read: "Every listed set must be a clique with at least two
    vertices and every edge must lie in exactly one listed clique."
    
    Hmm, "every edge must lie in exactly one listed clique" - this means for each edge e=(u,v),
    there is exactly one listed clique C such that u ∈ C and v ∈ C. It does NOT mean u and v
    are in the same clique as each other exclusively - they could be in other cliques too, as
    long as no other clique contains BOTH u and v.
    
    Wait, but if u is in clique C1 and v is in clique C2, and C1 ≠ C2, then edge (u,v) is not
    "in" any clique. So for edge (u,v) to be in clique C, we need u ∈ C AND v ∈ C.
    
    So the constraint is: for each edge (u,v), the set of cliques containing both u and v has
    size exactly 1.
    
    This does NOT require u and v to be in the same "group" in a partition sense. A vertex can
    belong to multiple cliques! The cliques don't need to form a partition of vertices; they
    form a partition of EDGES.
    
    OK so the correct framing is: we have a collection of cliques (each a set of ≥2 vertices,
    pairwise adjacent), and we need: for every edge (u,v) in the graph, exactly one clique
    contains both u and v.
    
    This is a different problem than what I was thinking. Let me reconsider.
    
    So the vertex-merging approach doesn't directly apply. Let me think differently.
    
    Actually, the original approach (enumerate maximal cliques, pick a subset that partitions edges)
    is the right framing. Let me try a different heuristic:
    
    New strategy: Instead of ILP, use a greedy + local search that works directly on the
    edge-to-clique assignment. Start with a greedy assignment, then use simulated annealing
    with moves that reassign individual edges to different cliques.
    """
    pass  # placeholder, we'll use a different approach below

def _all_maximal_cliques(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    # Use bitmasks for efficiency
    # adj_mask[v] = bitmask of neighbors of v
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    def _bk(R, P, X):
        if not P and not X:
            if R:
                verts = [i for i in range(n) if (R >> i) & 1]
                if len(verts) >= 2:
                    cliques.append(tuple(sorted(verts)))
            return
        
        # Choose pivot
        if P:
            pivot = max(P, key=lambda u: bin((P >> u) & adj_mask[u]).count('1'))
            candidates = [v for v in range(n) if (P >> v) & 1 and not ((adj_mask[v] >> pivot) & 1)]
        elif X:
            pivot = max(X, key=lambda u: bin((X >> u) & adj_mask[u]).count('1'))
            candidates = [v for v in range(n) if (P >> v) & 1]
        else:
            candidates = []
        
        for v in candidates:
            new_R = R | (1 << v)
            new_P = P & adj_mask[v]
            new_X = X & adj_mask[v]
            _bk(new_R, new_P, new_X)
            P &= ~(1 << v)
            X |= (1 << v)
    
    _bk(0, (1 << n) - 1, 0)
    return cliques

def _clique_edges(clique, adj, n):
    """Get all edges within a clique as a frozenset."""
    edges = set()
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            if adj[clique[i]][clique[j]]:
                edges.add(_edge_key(clique[i], clique[j]))
    return frozenset(edges)

def _greedy_edge_partition(adj, edge_list, all_cliques, n):
    """
    Greedy: repeatedly pick the clique that covers the most uncovered edges.
    """
    uncovered = set(edge_list)
    chosen = []
    
    # Precompute edges for each clique
    clique_edges = [_clique_edges(list(c), adj, n) for c in all_cliques]
    
    remaining = list(range(len(all_cliques)))
    
    while uncovered and remaining:
        best_idx = -1
        best_count = 0
        for idx in remaining:
            count = len(clique_edges[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx < 0 or best_count == 0:
            break
        chosen.append(list(all_cliques[best_idx]))
        uncovered -= clique_edges[best_idx]
        remaining.remove(best_idx)
    
    # Remaining edges as individual 2-cliques
    for e in sorted(uncovered):
        chosen.append([e[0], e[1]])
    
    return chosen

def _local_search(adj, edge_list, cliques, all_cliques, max_time=30.0):
    """
    Local search: try to reduce the number of cliques by:
    1. Merging two cliques if their union is a clique and doesn't create conflicts.
    2. Replacing a small clique with a larger one that covers its edges plus more.
    """
    start = time.time()
    current = [list(c) for c in cliques]
    
    def build_edge_to_clique(cliques_list):
        emap = {}
        for ci, c in enumerate(cliques_list):
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    e = _edge_key(c[a], c[b])
                    if e in emap:
                        return None
                    emap[e] = ci
        return emap
    
    def is_valid(cliques_list):
        emap = build_edge_to_clique(cliques_list)
        if emap is None:
            return False
        if len(emap) != len(edge_list):
            return False
        for c in cliques_list:
            if len(c) < 2:
                return False
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    if not adj[c[a]][c[b]]:
                        return False
        return True
    
    if not is_valid(current):
        return current
    
    rng = random.Random(42)
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        
        # Try merging pairs of cliques
        for i in range(len(current)):
            if time.time() - start > max_time:
                break
            for j in range(i + 1, len(current)):
                # Check if union is a clique
                merged = sorted(set(current[i]) | set(current[j]))
                valid = True
                for a in range(len(merged)):
                    for b in range(a + 1, len(merged)):
                        if not adj[merged[a]][merged[b]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check no edge conflict: merged clique's edges must be subset of
                # edges currently covered by cliques i and j
                merged_edges = set()
                for a in range(len(merged)):
                    for b in range(a + 1, len(merged)):
                        merged_edges.add(_edge_key(merged[a], merged[b]))
                
                old_edges = set()
                for c in [current[i], current[j]]:
                    for a in range(len(c)):
                        for b in range(a + 1, len(c)):
                            old_edges.add(_edge_key(c[a], c[b]))
                
                if merged_edges == old_edges:
                    # Safe merge
                    new_current = current[:i] + current[i+1:j] + current[j+1:] + [merged]
                    # Actually need to be careful with indices
                    new_current = []
                    for k, c in enumerate(current):
                        if k != i and k != j:
                            new_current.append(c)
                    new_current.append(merged)
                    
                    if is_valid(new_current):
                        current = new_current
                        improved = True
                        break
            if improved:
                break
        
        # Try replacing a 2-clique (edge) with a larger clique
        if not improved:
            for i in range(len(current)):
                if time.time() - start > max_time:
                    break
                if len(current[i]) != 2:
                    continue
                
                u, v = current[i]
                # Try to extend this edge to a larger clique
                # Find all maximal cliques containing edge (u,v)
                candidates = []
                for c in all_cliques:
                    if u in c and v in c and len(c) > 2:
                        candidates.append(c)
                
                for cand in candidates:
                    cand_set = set(cand)
                    cand_edges = set()
                    for a in range(len(cand)):
                        for b in range(a + 1, len(cand)):
                            cand_edges.add(_edge_key(cand[a], cand[b]))
                    
                    # Check: all edges in cand must be currently covered by clique i or other cliques
                    # that we're willing to "take over"
                    # Simple check: cand_edges must be subset of edges in current partition
                    # and no edge in cand_edges is covered by a clique other than i
                    # (unless that clique only has edges that are all in cand)
                    
                    # Simpler: just try replacing clique i with cand and check validity
                    new_current = current[:]
                    new_current[i] = list(cand)
                    if is_valid(new_current):
                        current = new_current
                        improved = True
                        break
                if improved:
                    break
    
    return current

def _solve_one_graph(adj, edge_list, time_budget=50.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    
    if not edge_list:
        return []
    
    # Enumerate maximal cliques
    t0 = time.time()
    all_cliques = _all_maximal_cliques(adj, n)
    t1 = time.time()
    
    if not all_cliques:
        return [[u, v] for u, v in edge_list]
    
    # Greedy initial solution
    greedy = _greedy_edge_partition(adj, edge_list, all_cliques, n)
    
    if not _verify_partition(greedy, edge_list, adj):
        # Fallback
        return [[u, v] for u, v in edge_list]
    
    # Local search
    remaining_time = time_budget - (time.time() - t0)
    if remaining_time > 5:
        ls_result = _local_search(adj, edge_list, greedy, all_cliques, max_time=min(remaining_time - 2, 35.0))
        if _verify_partition(ls_result, edge_list, adj):
            greedy = ls_result
    
    return greedy

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        # Allocate time proportionally
        time_budget = min(55.0, _remaining() - 10)
        
        cliques = _solve_one_graph(adj, edge_list, time_budget)
        
        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
