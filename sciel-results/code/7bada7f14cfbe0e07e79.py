# EVOLVE-BLOCK-START
import time
from random import Random

def _build_adj(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _solve_instance(n, p, seed, deadline):
    adj = _build_adj(n, p, seed)
    # Build edge set
    edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    
    if not edges:
        return []
    
    # Greedy: repeatedly pick an uncovered edge, grow maximal clique containing it
    uncovered = set(edges)
    cliques = []
    
    while uncovered:
        if time.time() > deadline:
            break
        # Pick edge: choose one whose endpoints have fewest uncovered edges (hardest)
        # Approximate: pick edge with minimum degree in uncovered subgraph
        best_edge = None
        best_score = float('inf')
        # Sample if too many edges
        edge_list = list(uncovered)
        if len(edge_list) > 2000:
            import random as _r
            sample = _r.sample(edge_list, 2000)
        else:
            sample = edge_list
        
        for (u, v) in sample:
            # Count uncovered edges incident to u or v
            cnt = 0
            for w in adj[u]:
                if (min(u,w), max(u,w)) in uncovered:
                    cnt += 1
            for w in adj[v]:
                if (min(v,w), max(v,w)) in uncovered:
                    cnt += 1
            if cnt < best_score:
                best_score = cnt
                best_edge = (u, v)
        
        if best_edge is None:
            best_edge = edge_list[0]
        
        u0, v0 = best_edge
        # Grow clique starting from {u0, v0}
        clique = {u0, v0}
        # Candidate vertices: neighbors of all in clique
        candidates = set(adj[u0]) & set(adj[v0])
        candidates.discard(u0)
        candidates.discard(v0)
        
        # Greedily add vertices to clique
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Sort candidates by degree in current clique neighborhood (prefer high connectivity)
            cand_list = sorted(candidates, key=lambda x: -len(adj[x]))
            for w in cand_list:
                if time.time() > deadline:
                    break
                if w in clique:
                    continue
                # Check w is adjacent to all in clique
                if all(w in adj[x] for x in clique):
                    # Check w is adjacent to all other candidates we might add - just add it
                    clique.add(w)
                    candidates.discard(w)
                    changed = True
                    break
            # Rebuild candidates after adding
            if changed:
                candidates = set()
                for x in clique:
                    candidates |= adj[x]
                candidates -= clique
        
        # Convert to sorted list
        clique_list = sorted(clique)
        cliques.append(clique_list)
        
        # Remove all edges within clique from uncovered
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = clique_list[i], clique_list[j]
                e = (min(a,b), max(a,b))
                uncovered.discard(e)
    
    # Post-processing: merge cliques if union is a clique
    if time.time() < deadline:
        merged = True
        while merged and time.time() < deadline:
            merged = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    if time.time() > deadline:
                        break
                    A = set(cliques[i])
                    B = set(cliques[j])
                    # Check if every vertex in A is adjacent to every vertex in B
                    ok = True
                    for a in A:
                        for b in B:
                            if b not in adj[a]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # Merge
                        new_clique = sorted(A | B)
                        cliques[i] = new_clique
                        cliques.pop(j)
                        merged = True
                        break
                if merged:
                    break
    
    return cliques

def construct():
    start = time.time()
    deadline = start + 130  # Safety margin before 150s kill
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            results.append([])
            continue
        cliques = _solve_instance(n, p, seed, deadline)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
