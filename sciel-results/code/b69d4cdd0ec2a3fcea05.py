"""Model-and-optimize approach for the 16-vertex, 65-edge triangle degree-sum problem."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_GUARD = 130.0  # safety margin

def _compute_score_from_mask(mask, n=16):
    """Compute max triangle degree-sum from a boolean adjacency matrix (numpy array)."""
    deg = mask.sum(axis=1)
    # Triangles: for each i < j < k, check if all three edges exist
    # Use vectorized approach: A[i,j] & A[i,k] & A[j,k]
    best = 0
    # Iterate over i < j, then find common neighbors k > j
    for i in range(n):
        for j in range(i+1, n):
            if mask[i, j]:
                # Common neighbors of i and j that are > j
                common = mask[i, j+1:] & mask[j, j+1:]
                ks = np.where(common)[0] + j + 1
                for k in ks:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best

def _compute_score_fast(mask, n=16):
    """Faster score computation using vectorized numpy operations."""
    deg = mask.sum(axis=1)
    best = 0
    # For each pair (i,j) with i<j and edge present, find all k>j with edges (i,k) and (j,k)
    for i in range(n-1):
        for j in range(i+1, n):
            if mask[i, j]:
                # Vertices k > j connected to both i and j
                common = mask[i, j+1:] & mask[j, j+1:]
                if common.any():
                    ks = np.where(common)[0] + j + 1
                    sums = deg[i] + deg[j] + deg[ks]
                    m = sums.max()
                    if m > best:
                        best = m
    return best

def _local_search_mask(mask, n, edge_set, time_budget=120.0):
    """Local search on numpy adjacency matrix with bitmask-based triangle detection."""
    start_t = time.time()
    best_score = _compute_score_fast(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    # Precompute all possible non-edges for quick sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_pairs = len(all_pairs)
    
    # Build a list of non-edges for random selection
    non_edges = [(i, j) for i, j in all_pairs if (i, j) not in edge_set]
    
    it = 0
    while time.time() - start_t < time_budget:
        it += 1
        if it % 1000 == 0 and time.time() > _DEADLINE_GUARD:
            break
        
        # Pick random existing edge to remove
        u, v = random.choice(list(edge_set))
        
        # Pick random non-edge to add
        # Rebuild non-edge list occasionally or use rejection
        a = random.randint(0, n-1)
        b = random.randint(0, n-1)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) == (u, v) or mask[a, b]:
            continue
        
        # Apply flip
        mask[u, v] = False
        mask[v, u] = False
        edge_set.discard((u, v))
        
        mask[a, b] = True
        mask[b, a] = True
        edge_set.add((a, b))
        
        # Compute new score
        new_score = _compute_score_fast(mask, n)
        
        if new_score < best_score:
            best_score = new_score
            best_mask = mask.copy()
            best_edge_set = set(edge_set)
        else:
            # Revert
            mask[u, v] = True
            mask[v, u] = True
            edge_set.add((u, v))
            
            mask[a, b] = False
            mask[b, a] = False
            edge_set.discard((a, b))
    
    return best_score, best_mask, best_edge_set

def _targeted_repair(mask, edge_set, n=16, time_budget=10.0):
    """Targeted repair: find the worst triangle and try to reduce its degree sum by flipping edges."""
    start_t = time.time()
    deg = mask.sum(axis=1)
    best_score = _compute_score_fast(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    while time.time() - start_t < time_budget:
        if time.time() > _DEADLINE_GUARD:
            break
        
        # Find the worst triangle
        deg = mask.sum(axis=1)
        worst_score = 0
        worst_tri = None
        for i in range(n-2):
            for j in range(i+1, n-1):
                if not mask[i, j]:
                    continue
                common = mask[i, j+1:] & mask[j, j+1:]
                ks = np.where(common)[0] + j + 1
                if len(ks) == 0:
                    continue
                sums = deg[i] + deg[j] + deg[ks]
                idx = sums.argmax()
                s = sums[idx]
                if s > worst_score:
                    worst_score = s
                    worst_tri = (i, j, int(ks[idx]))
        
        if worst_tri is None:
            break
        
        i, j, k = worst_tri
        # Try to reduce the degree sum by flipping edges incident to i, j, or k
        # Strategy: remove an edge from the highest-degree vertex in the triangle,
        # and add an edge elsewhere (preferably not creating new high-degree triangles)
        
        # Sort triangle vertices by degree descending
        verts = sorted([i, j, k], key=lambda x: -deg[x])
        
        improved = False
        for center in verts:
            if time.time() > _DEADLINE_GUARD:
                break
            
            # Try removing edges from center to other vertices
            neighbors = np.where(mask[center])[0]
            random.shuffle(neighbors.tolist())
            
            for nb in neighbors:
                # Don't remove edges that are part of the worst triangle (would just change which triangle is worst)
                # Actually, let's try removing any edge from center
                other = nb
                if other == center:
                    continue
                
                # Find a non-edge to add
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) in edge_set or (center, other) == (a, b):
                    continue
                
                # Apply
                mask[center, other] = False
                mask[other, center] = False
                edge_set.discard((min(center, other), max(center, other)))
                
                mask[a, b] = True
                mask[b, a] = True
                edge_set.add((a, b))
                
                new_score = _compute_score_fast(mask, n)
                
                if new_score < worst_score:
                    worst_score = new_score
                    best_mask = mask.copy()
                    best_edge_set = set(edge_set)
                    improved = True
                else:
                    # Revert
                    mask[center, other] = True
                    mask[other, center] = True
                    edge_set.add((min(center, other), max(center, other)))
                    
                    mask[a, b] = False
                    mask[b, a] = False
                    edge_set.discard((a, b))
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            # Try random flips as fallback
            break
    
    return worst_score, best_mask, best_edge_set

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
    
    best_score = _compute_score_fast(mask, n)
    best_mask = mask.copy()
    best_edge_set = set(edge_set)
    
    random.seed(42)
    
    # Phase 1: General local search
    time_per_trial = 8.0
    for trial in range(15):
        if time.time() > _DEADLINE_GUARD - 20:
            break
        
        # Create a perturbed starting point
        if trial == 0:
            mask_cur = best_mask.copy()
            es_cur = set(best_edge_set)
        else:
            mask_cur = best_mask.copy()
            es_cur = set(best_edge_set)
            # Perturb: flip k random edges
            k = random.randint(3, 8)
            for _ in range(k):
                existing = list(es_cur)
                if not existing:
                    break
                e_rm = random.choice(existing)
                u, v = e_rm
                mask_cur[u, v] = False
                mask_cur[v, u] = False
                es_cur.discard(e_rm)
                
                for _try in range(50):
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
        
        time_left = _DEADLINE_GUARD - time.time()
        budget = min(time_per_trial, time_left)
        if budget < 1.0:
            break
        
        score, mask_new, es_new = _local_search_mask(mask_cur, n, es_cur, time_budget=budget)
        
        if score < best_score:
            best_score = score
            best_mask = mask_new
            best_edge_set = es_new
        
        if best_score <= 23:
            break
    
    # Phase 2: Targeted repair on the best solution found
    if time.time() < _DEADLINE_GUARD - 5:
        time_left = _DEADLINE_GUARD - time.time() - 5
        budget = min(15.0, time_left)
        if budget > 1.0:
            score2, mask2, es2 = _targeted_repair(best_mask, best_edge_set, n, time_budget=budget)
            if score2 < best_score:
                best_score = score2
                best_mask = mask2
                best_edge_set = es2
    
    # Final verification
    final_score = _compute_score_fast(best_mask, n)
    if final_score != best_score:
        best_score = final_score
    
    edges = [[u, v] for u, v in sorted(best_edge_set)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
