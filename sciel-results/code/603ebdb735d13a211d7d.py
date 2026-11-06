"""
Clique partition via Maximum Weight Clique decomposition.
Strategy:
1. Enumerate all maximal cliques using Bron-Kerbosch with pivot.
2. Greedily select the clique that covers the most uncovered edges.
3. Remove covered edges and repeat until all edges are covered.
4. Use multiple random orderings for the greedy phase to find good solutions.
"""
import time
import random
import sys
from collections import defaultdict

_START = time.time()
DEADLINE = 145.0  # safety margin from 150s

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

def _bron_kerbosch(adj, n):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    Returns a list of sets (each set is a maximal clique's vertex set).
    """
    maximal_cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                maximal_cliques.append(frozenset(R))
            return
        
        # Choose pivot: vertex in P union X with max degree in P
        pivot_candidates = list(P) + list(X)
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: sum(1 for u in P if adj[v][u]))
            candidates = [v for v in P if not adj[v][pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & {u for u in P if adj[v][u]}
            new_X = X & {u for u in X if adj[v][u]}
            bk(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    bk(set(), set(range(n)), set())
    return maximal_cliques

def _clique_edges(clique_verts, adj):
    """Return set of edges within the clique."""
    edges = set()
    verts = list(clique_verts)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if adj[verts[i]][verts[j]]:
                edges.add((min(verts[i], verts[j]), max(verts[i], verts[j])))
    return edges

def _greedy_mwc_decomposition(adj, edge_list, n, max_cliques_to_consider=2000, rng=None):
    """
    Greedy decomposition:
    1. Find all maximal cliques.
    2. Repeatedly pick the clique covering the most uncovered edges.
    3. Remove those edges from uncovered set.
    """
    if rng is None:
        rng = random.Random(42)
    
    # Enumerate maximal cliques
    maximal_cliques = _bron_kerbosch(adj, n)
    
    # Filter cliques with at least 2 vertices
    maximal_cliques = [c for c in maximal_cliques if len(c) >= 2]
    
    # Precompute edges for each maximal clique
    clique_edges_map = {}
    for c in maximal_cliques:
        edges = _clique_edges(c, adj)
        if edges:
            clique_edges_map[c] = edges
    
    uncovered = set(edge_list)
    selected_cliques = []
    
    while uncovered and _remaining() > 5:
        # Find the clique covering the most uncovered edges
        best_clique = None
        best_coverage = 0
        
        for c, edges in clique_edges_map.items():
            coverage = len(edges & uncovered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_clique = c
                if coverage == len(uncovered):
                    break
        
        if best_clique is None or best_coverage == 0:
            # Fallback: take a single edge
            e = next(iter(uncovered))
            selected_cliques.append(sorted([e[0], e[1]]))
            uncovered.discard(e)
        else:
            selected_cliques.append(sorted(best_clique))
            uncovered -= clique_edges_map[best_clique]
    
    # Handle any remaining uncovered edges
    for e in uncovered:
        selected_cliques.append(sorted([e[0], e[1]]))
    
    return selected_cliques

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    n_edges = len(edge_list)
    edge_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edge_list)}
    edge_cover = [0] * n_edges
    
    for clique in cliques:
        # Check clique validity
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
        # Mark edges
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                e = (min(clique[a], clique[b]), max(clique[a], clique[b]))
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    
    for c in edge_cover:
        if c != 1:
            return False
    return True

def _local_search_refine(cliques, edge_list, adj, n, max_time=10.0):
    """
    Refine the greedy solution by trying to merge cliques when possible.
    A merge is valid if the union of vertices forms a clique and the edge sets match.
    """
    start = time.time()
    current = [list(c) for c in cliques]
    best = [list(c) for c in current]
    best_count = len(current)
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        
        # Try merging any pair of cliques
        n_cliques = len(current)
        for i in range(n_cliques):
            if time.time() - start > max_time:
                break
            for j in range(i + 1, n_cliques):
                # Check if union forms a clique
                union_verts = set(current[i]) | set(current[j])
                verts = list(union_verts)
                valid = True
                for a in range(len(verts)):
                    for b in range(a + 1, len(verts)):
                        if not adj[verts[a]][verts[b]]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check edge coverage matches
                    new_edges = set()
                    for a in range(len(verts)):
                        for b in range(a + 1, len(verts)):
                            e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
                            new_edges.add(e)
                    
                    old_edges = set()
                    for c in (current[i], current[j]):
                        for a in range(len(c)):
                            for b in range(a + 1, len(c)):
                                e = (min(c[a], c[b]), max(c[a], c[b]))
                                old_edges.add(e)
                    
                    if new_edges == old_edges:
                        # Valid merge
                        new_list = [list(c) for c in current]
                        new_list.pop(j)
                        new_list.pop(i)
                        new_list.append(sorted(verts))
                        current = new_list
                        improved = True
                        if len(current) < best_count:
                            best = [list(c) for c in current]
                            best_count = len(current)
    
    return best

def _solve_clique_partition(adj, edge_list):
    """
    Solve the clique partition problem using MWC decomposition + local search.
    """
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    
    if not edge_list:
        return []
    
    start = time.time()
    max_time = min(100.0, _remaining() - 10)
    
    # Try multiple greedy runs with different random seeds for tie-breaking
    best_result = None
    best_count = float('inf')
    
    for attempt in range(5):
        if time.time() - start > max_time or _remaining() < 15:
            break
        
        rng = random.Random(attempt * 12345 + 67890)
        
        # Greedy MWC decomposition
        greedy_result = _greedy_mwc_decomposition(adj, edge_list, n, rng=rng)
        
        if not _verify_partition(greedy_result, edge_list, adj):
            continue
        
        # Local search refinement
        ls_result = _local_search_refine(greedy_result, edge_list, adj, n, 
                                          max_time=min(8.0, max_time - (time.time() - start)))
        
        if _verify_partition(ls_result, edge_list, adj):
            if len(ls_result) < best_count:
                best_count = len(ls_result)
                best_result = ls_result
    
    if best_result is None:
        # Fallback: each edge as its own clique
        best_result = [[u, v] for u, v in edge_list]
    
    return best_result

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            # Fallback: just output each edge as its own clique
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        # Solve
        cliques = _solve_clique_partition(adj, edge_list)
        
        # Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback: each edge its own clique
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
