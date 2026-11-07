"""Cochromatic partition via branch-and-bound set cover over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys

def _maximal_sets(adj, n):
    """Return list of maximal homogeneous sets (each a clique or independent set)."""
    sets = []
    # Bron-Kerbosch for maximal cliques
    def bronk(R, Pset, X):
        if not Pset and not X:
            sets.append(list(R))
        for v in list(Pset):
            bronk(R + [v], [u for u in Pset if adj[v][u]], [u for u in X if adj[v][u]])
            Pset.remove(v)
            X.append(v)
    bronk([], list(range(n)), [])
    # Maximal independent sets = maximal cliques of complement
    comp_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            comp_adj[i][j] = (not adj[i][j]) and (i != j)
    def bronk2(R, Pset, X):
        if not Pset and not X:
            sets.append(list(R))
        for v in list(Pset):
            bronk2(R + [v], [u for u in Pset if comp_adj[v][u]], [u for u in X if comp_adj[v][u]])
            Pset.remove(v)
            X.append(v)
    bronk2([], list(range(n)), [])
    return sets

def _max_clique_size(adj_mask, n, deadline):
    """Compute maximum clique size in graph defined by adj_mask bitmasks using branch and bound."""
    if not any(adj_mask):
        return 0
    best = 0
    # Simple Bron-Kerbosch with pivot for max clique
    def bk(P, X, R):
        nonlocal best
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best:
                best = len(R)
            return
        # Choose pivot u from P ∪ X with max degree in P
        candidates = P | X
        if not candidates:
            return
        u = (candidates & -candidates).bit_length() - 1
        # Neighbors of u in P
        nu = P & adj_mask[u]
        for v in list(nu):
            bk(P & adj_mask[v], X & adj_mask[v], R | (1 << v))
            P &= ~(1 << v)
            X |= (1 << v)
    bk((1 << n) - 1, 0, 0)
    return best

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 90s kill

    sys.setrecursionlimit(10000)

    # Build adjacency bitmasks
    adj = [0]*n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Find all maximal homogeneous sets
    adj_bool = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_bool[i][j] = bool(adj[i] >> j & 1)

    hom_sets = _maximal_sets(adj_bool, n)
    # Convert to bitmasks and deduplicate
    set_masks = []
    seen = set()
    for s in hom_sets:
        m = 0
        for v in s:
            m |= (1 << v)
        if m not in seen:
            seen.add(m)
            set_masks.append(m)

    full = (1 << n) - 1

    # Greedy initial upper bound: repeatedly pick a set covering the most uncovered vertices
    uncovered = full
    chosen = []
    while uncovered:
        best = None
        best_cnt = -1
        for m in set_masks:
            c = bin(m & uncovered).count('1')
            if c > best_cnt:
                best_cnt = c
                best = m
        chosen.append(best)
        uncovered &= ~best
    ub = len(chosen)

    # Also try greedy vertex assignment as fallback
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    if len(parts) < ub:
        ub = len(parts)
    best_parts = [list(p) for p in parts]

    # Sort sets by size descending for better pruning
    set_masks_sorted = sorted(set_masks, key=lambda m: bin(m).count('1'), reverse=True)

    # Precompute for each vertex which sets contain it
    vertex_sets = [[] for _ in range(n)]
    for idx, m in enumerate(set_masks_sorted):
        for v in range(n):
            if (m >> v) & 1:
                vertex_sets[v].append(idx)

    # Conflict graph: two vertices conflict if they cannot be in the same homogeneous set
    # i.e., they are adjacent in G and also adjacent in complement? No:
    # Two vertices can be in same clique iff they are adjacent in G
    # Two vertices can be in same independent set iff they are NOT adjacent in G
    # So two vertices can be in same homogeneous set iff (adj[i][j] XOR adj[j][i]) == False... 
    # Actually: they can be in same clique if adjacent, same indep set if not adjacent.
    # They CANNOT be in same homogeneous set only if... actually every pair can be in SOME homogeneous set.
    # The conflict is: if we put i and j in different parts, that's fine. The constraint is that
    # each part must be ALL clique or ALL independent set.
    # Lower bound via clique in "must-be-separate" graph: two vertices must be in different parts
    # if no homogeneous set contains both. But every pair is contained in some homogeneous set
    # (either as a clique of size 2 if adjacent, or indep set of size 2 if not adjacent).
    # So this lower bound is weak. Instead, use: max number of vertices that pairwise conflict
    # in the sense that no single homogeneous set can contain all of them.
    
    # Better lower bound: for the remaining vertices, compute the chromatic number of the graph
    # where edges connect vertices that CANNOT be in the same part. But since any pair CAN be in
    # the same part, this graph has no edges, giving lower bound 1. Not useful.
    
    # Alternative: use the fact that each part is either a clique or independent set.
    # Lower bound = max(omega(G[remaining]), alpha(G[remaining]))? No, that's not right either.
    # Actually, a valid lower bound: the size of the largest set of vertices such that no two
    # can be in the same part. Since any two can be in the same part, this is 1.
    
    # Let's use a simpler but effective bound: ceil(|remaining| / max_set_size)
    max_set_size = max(bin(m).count('1') for m in set_masks) if set_masks else 1

    def bnb(remaining, depth, chosen_masks):
        nonlocal ub, best_parts
        if time.time() > deadline:
            return
        if not remaining:
            if depth < ub:
                ub = depth
                assign = [-1]*n
                for idx, m in enumerate(chosen_masks):
                    for v in range(n):
                        if (m >> v) & 1 and assign[v] == -1:
                            assign[v] = idx
                parts_new = [[] for _ in range(len(chosen_masks))]
                for v in range(n):
                    if assign[v] >= 0:
                        parts_new[assign[v]].append(v)
                parts_new = [p for p in parts_new if p]
                if len(parts_new) <= depth:
                    best_parts = parts_new
            return
        # Lower bound: at least ceil(popcount(remaining)/max_set_size)
        rem_count = bin(remaining).count('1')
        lb = (rem_count + max_set_size - 1) // max_set_size
        if depth + lb >= ub:
            return
        # Pick a vertex in remaining to branch on (MRV: fewest candidates)
        best_v = -1
        best_cand_count = n + 1
        rbits = remaining
        while rbits:
            v = (rbits & -rbits).bit_length() - 1
            rbits &= rbits - 1
            # Count sets containing v and fully contained in remaining
            cnt = 0
            for idx in vertex_sets[v]:
                m = set_masks_sorted[idx]
                if (m & ~remaining) == 0:
                    cnt += 1
            if cnt < best_cand_count:
                best_cand_count = cnt
                best_v = v
                if cnt == 1:
                    break
        if best_v == -1:
            return
        if best_cand_count == 0:
            return
        # Get candidates for this vertex, sorted by size descending
        candidates = []
        for idx in vertex_sets[best_v]:
            m = set_masks_sorted[idx]
            if (m & ~remaining) == 0:
                candidates.append((bin(m).count('1'), m))
        candidates.sort(reverse=True)
        # Try candidates
        for _, m in candidates:
            if time.time() > deadline:
                return
            new_remaining = remaining & ~m
            chosen_masks.append(m)
            bnb(new_remaining, depth + 1, chosen_masks)
            chosen_masks.pop()
            if depth + 1 >= ub:
                return

    bnb(full, 0, [])

    # Ensure best_parts is a valid partition
    if not best_parts:
        return [list(p) for p in parts]
    # Check all vertices covered exactly once
    seen = set()
    for part in best_parts:
        for v in part:
            if v in seen:
                return [list(p) for p in parts]
            seen.add(v)
    if seen != set(range(n)):
        return [list(p) for p in parts]
    # Check each part homogeneous
    for part in best_parts:
        if len(part) == 1:
            continue
        hits = []
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                hits.append((part[i], part[j]) in es or (part[j], part[i]) in es)
        if not (all(hits) or not any(hits)):
            return [list(p) for p in parts]
    # Return as list of lists of ints
    return [[int(v) for v in part] for part in best_parts]
# EVOLVE-BLOCK-END
