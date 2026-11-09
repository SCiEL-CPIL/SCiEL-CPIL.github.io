"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X with max |P ∩ N(u)|
        if P:
            pivot_candidates = list(P)
        elif X:
            pivot_candidates = list(X)
        else:
            pivot_candidates = []
        
        if pivot_candidates:
            # Pick pivot with max degree into P
            pivot = max(pivot_candidates, key=lambda u: len(P & adj[u]))
            # Only recurse on P \ N(pivot)
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            X.add(v)
            bk(new_R, new_P, new_X)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _min_hitting_set(cliques, n):
    """Find a minimum hitting set for a family of cliques (each clique is a set of vertices).
    Uses greedy + local search."""
    if not cliques:
        return []
    
    # Greedy: repeatedly pick vertex that covers most uncovered cliques
    uncovered = list(cliques)
    hit = set()
    
    while uncovered:
        # Count how many uncovered cliques each vertex is in
        counts = defaultdict(int)
        for c in uncovered:
            for v in c:
                counts[v] += 1
        
        if not counts:
            break
        
        # Pick vertex with max count
        best_v = max(counts, key=lambda v: counts[v])
        hit.add(best_v)
        
        # Remove all cliques containing best_v
        uncovered = [c for c in uncovered if best_v not in c]
    
    # Local search: try to remove vertices and re-cover
    # Try removing each vertex in hit and see if we can still cover
    improved = True
    while improved:
        improved = False
        for v in list(hit):
            trial_hit = hit - {v}
            # Check if trial_hit still hits all cliques
            ok = True
            for c in cliques:
                if not (trial_hit & c):
                    ok = False
                    break
            if ok:
                hit = trial_hit
                improved = True
    
    # Swap-based local search: try replacing one vertex with another
    improved = True
    while improved:
        improved = False
        for v in list(hit):
            trial_hit = hit - {v}
            # Find uncovered cliques
            uncovered = [c for c in cliques if not (trial_hit & c)]
            if not uncovered:
                # v was redundant
                hit = trial_hit
                improved = True
                break
            # Try adding each vertex w not in hit that covers most uncovered
            best_w = None
            best_count = 0
            for w in range(n):
                if w in hit:
                    continue
                cnt = sum(1 for c in uncovered if w in c)
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            if best_w is not None and best_count > 0:
                # Only swap if it doesn't increase size (it's same size)
                # But we want to maintain same size, so swap is neutral in size
                # Only do swap if it allows future improvements - skip for now
                pass
    
    return sorted(hit)

def construct():
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = _generate_graph(n, p, seed)
        all_cliques = _bron_kerbosch(adj, n)
        # Filter to cliques with at least 2 vertices
        cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        hit = _min_hitting_set(cliques, n)
        results.append(hit)
    
    return results
# EVOLVE-BLOCK-END
