"""Combined approach: NumPy bitmask local search with incremental score + targeted repair + polish."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_GUARD = 132.0  # safety margin before hard kill

def _compute_score(mask, n=16):
    """Compute max triangle degree-sum from a boolean adjacency matrix."""
    deg = mask.sum(axis=1)
    best = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if mask[i, j]:
                common = mask[i, j+1:] & mask[j, j+1:]
                if common.any():
                    ks = np.where(common)[0] + j + 1
                    sums = deg[i] + deg[j] + deg[ks]
                    m = sums.max()
                    if m > best:
                        best = m
    return best

def _compute_score_partial(mask, deg, candidate_tris, n=16):
    """Compute max degree-sum over a given set of candidate triangles.
    candidate_tris: list of (i, j, k) tuples.
    Returns the max sum, or 0 if no triangles."""
    if not candidate_tris:
        return 0
    best = 0
    for i, j, k in candidate_tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best

def _find_all_triangles(mask, n=16):
    """Return list of all triangles as (i,j,k) with i<j<k."""
    tris = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if mask[i, j]:
                common = mask[i, j+1:] & mask[j, j+1:]
                if common.any():
                    ks = np.where(common)[0] + j + 1
                    for k in ks:
                        tris.append((i, j, k))
    return tris

def _local_search_incremental(mask, n, edge_set, time_budget=120.0):
    """Local search with incremental score update.
    Only evaluates triangles that could change after a flip."""
    start_t = time.time()
    deg = mask.sum(axis=1)
    best_score = _compute_score(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    best_deg = deg.copy()
    all_tris = _find_all_triangles(mask, n)
    
    # Precompute: for each vertex, list of triangles it participates in
    vertex_tris = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(all_tris):
        vertex_tris[i].append(idx)
        vertex_tris[j].append(idx)
        vertex_tris[k].append(idx)
    
    # Build non-edge list
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not mask[i, j]]
    
    it = 0
    while time.time() - start_t < time_budget:
        it += 1
        if it % 2000 == 0 and time.time() > _DEADLINE_GUARD:
            break
        
        # Pick random existing edge to remove
        u, v = random.choice(list(edge_set))
        
        # Pick random non-edge to add
        a, b = random.choice(non_edges)
        if (a, b) == (u, v):
            continue
        
        # Apply flip
        mask[u, v] = False
        mask[v, u] = False
        edge_set.discard((u, v))
        deg[u] -= 1
        deg[v] -= 1
        
        mask[a, b] = True
        mask[b, a] = True
        edge_set.add((a, b))
        deg[a] += 1
        deg[b] += 1
        
        # Determine which triangles to check:
        # A triangle's degree-sum changes only if one of its vertices is u, v, a, or b.
        # Also, new triangles can form that include the new edge (a,b).
        # Old triangles broken: those that used edge (u,v).
        
        # Collect candidate triangle indices
        cand_set = set()
        for vrt in (u, v, a, b):
            cand_set.update(vertex_tris[vrt])
        
        # Also add any new triangles formed by edge (a,b)
        # A new triangle (a, b, x) exists if x is connected to both a and b
        common_ab = mask[a, :b] & mask[b, :b]  # x < b
        common_ab2 = mask[a, b+1:] & mask[b, b+1:]  # x > b
        # For x < b: need x < a or a < x < b, and also check ordering
        # Actually simpler: for each x, check if mask[a,x] and mask[b,x]
        # Let's just iterate over all x != a, b
        for x in range(n):
            if x == a or x == b:
                continue
            if mask[a, x] and mask[b, x]:
                # Triangle (min(a,b,x), mid, max)
                tris_x = sorted([a, b, x])
                # This is a new triangle; add it to candidate list
                cand_set.add(len(all_tris))  # sentinel: will handle separately
                # Actually let's just track new triangles separately
                break
        
        # More efficient: find new triangles involving edge (a,b)
        new_tris = []
        for x in range(n):
            if x == a or x == b:
                continue
            if mask[a, x] and mask[b, x]:
                new_tris.append((min(a,b,x), sorted([a,b,x])[1], max(a,b,x)))
        
        # Also, removing (u,v) may have broken triangles (u,v,x) — those are no longer triangles
        # We need to exclude them from consideration
        broken_tris = set()
        for x in range(n):
            if x == u or x == v:
                continue
            if mask[u, x] and mask[v, x]:
                # This triangle still exists (we removed u-v but u-x and v-x still there? No!)
                # Wait: we removed edge (u,v), so triangle (u,v,x) is broken
                # mask[u,v] is now False, so this won't be found
                pass
        # Actually since mask[u,v] = False now, triangles (u,v,x) won't be in all_tris anymore
        # But all_tris was computed before the flip. We need to handle this carefully.
        
        # Simpler approach: just recompute the score from scratch but only over
        # triangles that could have changed. The set of triangles that could have
        # changed is: all triangles involving u, v, a, or b (old ones), plus new
        # triangles involving the new edge (a,b).
        
        # Let me use a cleaner approach: collect all triangles that exist NOW
        # and involve at least one of {u, v, a, b}, plus verify no triangle
        # NOT involving any of them has a higher sum (which is impossible since
        # their degrees didn't change).
        
        # Actually the key insight: triangles NOT involving u, v, a, or b have
        # the same degree-sum as before (their degrees are unchanged). So the
        # new max is either:
        #   (1) the old max among triangles not involving {u,v,a,b}, or
        #   (2) the max among triangles involving at least one of {u,v,a,b}
        
        # For (1): we can precompute this. But it changes as we accept moves.
        # Let's just compute the max over ALL triangles that currently exist
        # and involve at least one of {u,v,a,b}, and compare with a cached
        # "background" max for triangles not involving any of them.
        
        # For simplicity and correctness, let's just compute the full score
        # but only over the relevant subset. Since n=16, even a full recompute
        # is fast (~1000 triangles max). The bottleneck was doing it in Python
        # loops. Let me use a vectorized approach.
        
        new_score = _compute_score(mask, n)
        
        if new_score < best_score:
            best_score = new_score
            best_mask = mask.copy()
            best_edge_set = set(edge_set)
            best_deg = deg.copy()
            all_tris = _find_all_triangles(mask, n)
            vertex_tris = [[] for _ in range(n)]
            for idx, (i, j, k) in enumerate(all_tris):
                vertex_tris[i].append(idx)
                vertex_tris[j].append(idx)
                vertex_tris[k].append(idx)
            # Update non_edges
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not mask[i, j]]
        else:
            # Revert
            mask[u, v] = True
            mask[v, u] = True
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
            
            mask[a, b] = False
            mask[b, a] = False
            edge_set.discard((a, b))
            deg[a] -= 1
            deg[b] -= 1
    
    return best_score, best_mask, best_edge_set

def _local_search_fast(mask, n, edge_set, time_budget=120.0):
    """Faster local search: use the fact that for n=16, full score recompute
    is cheap if vectorized well. Focus on maximizing iteration count."""
    start_t = time.time()
    best_score = _compute_score(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    # Precompute all non-edges
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not mask[i, j]]
    edge_list = list(edge_set)
    
    it = 0
    while time.time() - start_t < time_budget:
        it += 1
        if it % 5000 == 0 and time.time() > _DEADLINE_GUARD:
            break
        
        # Pick random flip
        u, v = random.choice(edge_list)
        a, b = random.choice(non_edges)
        if (a, b) == (u, v):
            continue
        
        # Apply
        mask[u, v] = False
        mask[v, u] = False
        mask[a, b] = True
        mask[b, a] = True
        
        new_score = _compute_score(mask, n)
        
        if new_score < best_score:
            best_score = new_score
            best_mask = mask.copy()
            best_edge_set = set(edge_set)
            # Update tracking structures
            edge_set.discard((u, v))
            edge_set.add((a, b))
            edge_list = list(edge_set)
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not mask[i, j]]
        else:
            # Revert
            mask[u, v] = True
            mask[v, u] = True
            mask[a, b] = False
            mask[b, a] = False
    
    return best_score, best_mask, best_edge_set

def _targeted_repair(mask, edge_set, n=16, time_budget=15.0):
    """Targeted repair: find worst triangle, try to break it or reduce member degrees."""
    start_t = time.time()
    best_score = _compute_score(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    while time.time() - start_t < time_budget:
        if time.time() > _DEADLINE_GUARD:
            break
        
        deg = mask.sum(axis=1)
        
        # Find worst triangle
        worst_score = 0
        worst_tri = None
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                if mask[i, j]:
                    common = mask[i, j+1:] & mask[j, j+1:]
                    if common.any():
                        ks = np.where(common)[0] + j + 1
                        sums = deg[i] + deg[j] + deg[ks]
                        idx = sums.argmax()
                        s = sums[idx]
                        if s > worst_score:
                            worst_score = s
                            worst_tri = (i, j, int(ks[idx]))
        
        if worst_tri is None:
            break
        
        i, j, k = worst_tri
        improved = False
        
        # Strategy 1: Remove one edge of the worst triangle, add a "safe" edge elsewhere
        for center in sorted([i, j, k], key=lambda x: -deg[x]):
            if time.time() > _DEADLINE_GUARD:
                break
            
            # Try removing edges from center
            neighbors = [x for x in range(n) if mask[center, x]]
            random.shuffle(neighbors)
            
            for nb in neighbors:
                # Find a non-edge to add (prefer low-degree vertices to avoid new high-sum triangles)
                non_edges_here = [(x, y) for x in range(n) for y in range(x+1, n)
                                  if not mask[x, y] and (x, y) != (min(center,nb), max(center,nb))]
                if not non_edges_here:
                    continue
                
                # Sort non-edges by sum of endpoint degrees (prefer low)
                non_edges_here.sort(key=lambda xy: deg[xy[0]] + deg[xy[1]])
                
                # Try a few candidates
                for a, b in non_edges_here[:20]:
                    # Apply
                    mask[center, nb] = False
                    mask[nb, center] = False
                    mask[a, b] = True
                    mask[b, a] = True
                    
                    new_score = _compute_score(mask, n)
                    
                    if new_score < worst_score:
                        worst_score = new_score
                        best_score = min(best_score, new_score)
                        best_mask = mask.copy()
                        edge_set.discard((min(center, nb), max(center, nb)))
                        edge_set.add((a, b))
                        best_edge_set = set(edge_set)
                        improved = True
                    else:
                        # Revert
                        mask[center, nb] = True
                        mask[nb, center] = True
                        mask[a, b] = False
                        mask[b, a] = False
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            # Try random perturbation to escape local optimum
            k_flip = random.randint(2, 5)
            for _ in range(k_flip):
                existing = list(edge_set)
                if not existing:
                    break
                u, v = random.choice(existing)
                mask[u, v] = False
                mask[v, u] = False
                edge_set.discard((u, v))
                
                non_edges_now = [(x, y) for x in range(n) for y in range(x+1, n) if not mask[x, y]]
                if non_edges_now:
                    a, b = random.choice(non_edges_now)
                    mask[a, b] = True
                    mask[b, a] = True
                    edge_set.add((a, b))
            
            # Do a quick local search from here
            score, m2, es2 = _local_search_fast(mask, n, edge_set, time_budget=2.0)
            if score < best_score:
                best_score = score
                best_mask = m2
                best_edge_set = es2
                mask = m2
                edge_set = es2
            else:
                # Revert to best
                mask = best_mask.copy()
                edge_set = set(best_edge_set)
            break  # Stop targeted repair after one perturbation attempt
    
    return best_score, best_mask, best_edge_set

def _polish(mask, edge_set, n=16, time_budget=10.0):
    """Final polish: exhaustive 2-flip neighborhood search with exact evaluation."""
    start_t = time.time()
    best_score = _compute_score(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    edges_list = list(edge_set)
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not mask[i, j]]
    
    # Single-flip neighborhood
    for u, v in edges_list:
        if time.time() > start_t + time_budget or time.time() > _DEADLINE_GUARD:
            break
        for a, b in non_edges:
            if (a, b) == (u, v):
                continue
            mask[u, v] = False
            mask[v, u] = False
            mask[a, b] = True
            mask[b, a] = True
            
            new_score = _compute_score(mask, n)
            
            if new_score < best_score:
                best_score = new_score
                best_mask = mask.copy()
                best_edge_set = set(edge_set)
                edge_set.discard((u, v))
                edge_set.add((a, b))
                edges_list = list(edge_set)
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not mask[i, j]]
            else:
                mask[u, v] = True
                mask[v, u] = True
                mask[a, b] = False
                mask[b, a] = False
    
    # Double-flip neighborhood (remove 2 edges, add 2 edges)
    if time.time() < start_t + time_budget and time.time() < _DEADLINE_GUARD:
        edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) not in edge_set]
        
        # Sample a subset of double flips to stay within budget
        random.shuffle(edges_list)
        random.shuffle(non_edges)
        
        for i in range(min(200, len(edges_list))):
            if time.time() > start_t + time_budget or time.time() > _DEADLINE_GUARD:
                break
            u1, v1 = edges_list[i]
            for j in range(i+1, min(i+50, len(edges_list))):
                if time.time() > start_t + time_budget or time.time() > _DEADLINE_GUARD:
                    break
                u2, v2 = edges_list[j]
                
                for a, b in non_edges[:30]:
                    if (a, b) in edge_set:
                        continue
                    if (a, b) == (u1, v1) or (a, b) == (u2, v2):
                        continue
                    # Check second add edge
                    found = False
                    for c, d in non_edges[:30]:
                        if (c, d) == (a, b) or (c, d) in edge_set:
                            continue
                        if (c, d) == (u1, v1) or (c, d) == (u2, v2):
                            continue
                        found = True
                        break
                    if not found:
                        continue
                    
                    # Apply double flip
                    mask[u1, v1] = False; mask[v1, u1] = False
                    mask[u2, v2] = False; mask[v2, u2] = False
                    mask[a, b] = True; mask[b, a] = True
                    mask[c, d] = True; mask[d, c] = True
                    
                    new_score = _compute_score(mask, n)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_mask = mask.copy()
                        edge_set.discard((u1, v1))
                        edge_set.discard((u2, v2))
                        edge_set.add((a, b))
                        edge_set.add((c, d))
                        best_edge_set = set(edge_set)
                    else:
                        mask[u1, v1] = True; mask[v1, u1] = True
                        mask[u2, v2] = True; mask[v2, u2] = True
                        mask[a, b] = False; mask[b, a] = False
                        mask[c, d] = False; mask[d, c] = False
    
    return best_score, best_mask, best_edge_set

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Start with K_{6,10} + 5 matching edges in B
    # Part A: 0-5 (size 6), Part B: 6-15 (size 10)
    edge_set = set()
    for u in range(6):
        for v in range(6, 16):
            edge_set.add((u, v))
    for a, b in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
        edge_set.add((a, b))
    
    # Build numpy adjacency matrix
    mask = np.zeros((n, n), dtype=bool)
    for u, v in edge_set:
        mask[u, v] = True
        mask[v, u] = True
    
    best_score = _compute_score(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    random.seed(42)
    
    # Phase 1: Multiple local search trials with perturbation
    time_per_trial = 10.0
    for trial in range(12):
        if time.time() > _DEADLINE_GUARD - 25:
            break
        
        if trial == 0:
            mask_cur = best_mask.copy()
            es_cur = set(best_edge_set)
        else:
            mask_cur = best_mask.copy()
            es_cur = set(best_edge_set)
            # Perturb: flip k random edges
            k = random.randint(3, 10)
            for _ in range(k):
                existing = list(es_cur)
                if not existing:
                    break
                e_rm = random.choice(existing)
                u, v = e_rm
                mask_cur[u, v] = False
                mask_cur[v, u] = False
                es_cur.discard(e_rm)
                
                # Find a non-edge to add
                for _try in range(100):
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    if (a, b) == (u, v) or mask_cur[a, b]:
                        continue
                    mask_cur[a, b] = True
                    mask_cur[b, a] = True
                    es_cur.add((a, b))
                    break
        
        time_left = _DEADLINE_GUARD - time.time() - 25
        budget = min(time_per_trial, max(1.0, time_left))
        if budget < 1.0:
            break
        
        score, mask_new, es_new = _local_search_fast(mask_cur, n, es_cur, time_budget=budget)
        
        if score < best_score:
            best_score = score
            best_mask = mask_new
            best_edge_set = es_new
        
        if best_score <= 23:
            break
    
    # Phase 2: Targeted repair on the best solution
    if time.time() < _DEADLINE_GUARD - 10:
        time_left = _DEADLINE_GUARD - time.time() - 10
        budget = min(12.0, time_left)
        if budget > 1.0:
            score2, mask2, es2 = _targeted_repair(best_mask, best_edge_set, n, time_budget=budget)
            if score2 < best_score:
                best_score = score2
                best_mask = mask2
                best_edge_set = es2
    
    # Phase 3: Final polish (exhaustive 1-flip and sampled 2-flip)
    if time.time() < _DEADLINE_GUARD - 2:
        time_left = _DEADLINE_GUARD - time.time() - 2
        budget = min(8.0, time_left)
        if budget > 0.5:
            score3, mask3, es3 = _polish(best_mask, best_edge_set, n, time_budget=budget)
            if score3 < best_score:
                best_score = score3
                best_mask = mask3
                best_edge_set = es3
    
    # Final verification
    final_score = _compute_score(best_mask, n)
    if final_score != best_score:
        best_score = final_score
    
    edges = [[u, v] for u, v in sorted(best_edge_set)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
