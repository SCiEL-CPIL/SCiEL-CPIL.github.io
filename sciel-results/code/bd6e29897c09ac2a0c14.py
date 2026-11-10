"""Edge clique partition via greedy maximal-clique selection with time guard."""
import time
import random
from itertools import combinations

def _generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_cliques(n, adj):
    """Bron-Kerbosch with pivot, using sets. Returns list of frozensets."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Pivot selection: choose vertex in P∪X with max |P ∩ adj[u]|
        union_px = P | X
        if union_px:
            pivot = max(union_px, key=lambda u: len(P & adj[u]))
            candidates = P - adj[pivot]
        else:
            candidates = P
            
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            X.add(v)
            bk(new_R, new_P, new_X)
    
    bk(set(), set(range(n)), set())
    return cliques

def _greedy_clique_partition(n, adj, edges, deadline):
    """
    Greedy: repeatedly pick a maximal clique that covers the most uncovered edges,
    add it to partition, remove its edges. Repeat until all edges covered.
    Use local search at the end to try merging/optimizing.
    """
    edge_set = set(edges)
    uncovered = set(edges)
    partition = []
    
    # Precompute all maximal cliques
    cliques = _find_maximal_cliques(n, adj)
    
    # Sort cliques by size descending for initial preference
    cliques.sort(key=lambda c: len(c), reverse=True)
    
    # For efficiency, map each clique to its edge set
    clique_edges = {}
    for c in cliques:
        verts = list(c)
        es = frozenset((min(a,b), max(a,b)) for a, b in combinations(verts, 2))
        clique_edges[c] = es
    
    while uncovered and time.time() < deadline:
        best_clique = None
        best_count = 0
        
        for c in cliques:
            es = clique_edges[c]
            # Count how many edges of this clique are still uncovered
            cnt = len(es & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_clique = c
                if best_count == len(es):
                    # Can't do better
                    break
        
        if best_clique is None or best_count == 0:
            # No maximal clique covers any uncovered edge? This shouldn't happen
            # since every edge is in some maximal clique. Fallback: use single edge.
            # Find an uncovered edge and make it a clique of size 2
            remaining = list(uncovered)
            if remaining:
                e = remaining[0]
                partition.append(list(e))
                uncovered.discard(e)
            else:
                break
        else:
            partition.append(list(best_clique))
            uncovered -= clique_edges[best_clique]
    
    # Verify all edges are covered
    if uncovered:
        # Fallback: add remaining edges as 2-cliques
        for e in uncovered:
            partition.append(list(e))
        uncovered = set()
    
    return partition

def _verify_and_fix(n, adj, edges, partition):
    """Verify partition is valid; if not, fix by using individual edges."""
    edge_set = set(edges)
    covered = set()
    valid = True
    for c in partition:
        if len(c) < 2:
            valid = False
            break
        # Check it's a clique
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                u, v = c[i], c[j]
                if v not in adj[u]:
                    valid = False
                    break
                e = (min(u,v), max(u,v))
                if e in covered:
                    valid = False
                    break
                covered.add(e)
            if not valid:
                break
        if not valid:
            break
    
    if not valid or covered != edge_set:
        # Fallback: use individual edges
        return [list(e) for e in edges]
    
    return partition

def _local_improve(n, adj, partition, deadline):
    """
    Try to improve the partition by:
    1. Merging two cliques if their union is a clique (and no edge overlap).
    2. Replacing a set of cliques with a larger clique.
    Simple version: try merging pairs.
    """
    if time.time() > deadline:
        return partition
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_partition = list(partition)
        
        # Build a set of edges covered by each clique for quick lookup
        clique_edge_sets = []
        for c in new_partition:
            verts = list(c)
            es = frozenset((min(a,b), max(a,b)) for a, b in combinations(verts, 2))
            clique_edge_sets.append(es)
        
        # Try merging pairs i, j where i < j
        merged = False
        for i in range(len(new_partition)):
            if merged:
                break
            for j in range(i+1, len(new_partition)):
                if time.time() > deadline:
                    return new_partition
                ci = set(new_partition[i])
                cj = set(new_partition[j])
                if ci & cj:
                    continue  # share vertices, can't merge into a single clique easily
                # Check if ci ∪ cj is a clique
                union_verts = ci | cj
                is_clique = True
                for u in union_verts:
                    for v in union_verts:
                        if u >= v:
                            continue
                        if v not in adj[u]:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    # Also check no edge overlap (they shouldn't since disjoint vertex sets)
                    new_partition[i] = list(union_verts)
                    new_partition.pop(j)
                    improved = True
                    merged = True
                    break
        
        if not merged:
            # Try replacing two cliques with a bigger one
            # For each pair, see if there's a maximal clique containing both
            # This is expensive; skip for now
            break
        
        partition = new_partition
    
    return partition

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _generate_edges(n, p, seed)
        adj = _build_adj(n, edges)
        
        partition = _greedy_clique_partition(n, adj, edges, deadline)
        partition = _verify_and_fix(n, adj, edges, partition)
        
        if time.time() < deadline - 5:
            partition = _local_improve(n, adj, partition, deadline)
        
        results.append(partition)
    
    return results
