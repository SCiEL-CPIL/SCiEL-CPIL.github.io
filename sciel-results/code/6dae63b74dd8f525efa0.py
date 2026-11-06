"""NumPy-bitmask local search for 16-vertex 65-edge triangle degree-sum minimization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_GUARD = 135.0  # safety margin before hard kill

N = 16
TARGET_EDGES = 65

def _edges_to_mask(edge_set):
    """edge_set: set of (u,v) with u<v. Return list of 16 ints (bitmasks)."""
    masks = [0] * N
    for u, v in edge_set:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _mask_to_edges(masks):
    edges = set()
    for i in range(N):
        m = masks[i]
        for j in range(i + 1, N):
            if m & (1 << j):
                edges.add((i, j))
    return edges

def _compute_score(masks):
    """Compute max triangle degree-sum from bitmask adjacency."""
    deg = [0] * N
    for i in range(N):
        deg[i] = bin(masks[i]).count('1')
    best = 0
    for i in range(N):
        mi = masks[i]
        for j in range(i + 1, N):
            if mi & (1 << j):
                common = mi & masks[j]
                # only k > j to avoid double count
                for k in range(j + 1, N):
                    if common & (1 << k):
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _compute_score_np(masks_np, deg_np):
    """masks_np: (N,) int64 array of bitmasks. deg_np: (N,) int array."""
    best = 0
    for i in range(N):
        mi = masks_np[i]
        for j in range(i + 1, N):
            if mi & (1 << j):
                common = mi & masks_np[j]
                for k in range(j + 1, N):
                    if common & (1 << k):
                        s = deg_np[i] + deg_np[j] + deg_np[k]
                        if s > best:
                            best = s
    return best

def _deg_from_masks(masks):
    return [bin(m).count('1') for m in masks]

def _find_worst_triangles(masks, deg, limit=50):
    """Return list of (i,j,k, score) for triangles, sorted by score desc."""
    tris = []
    for i in range(N):
        mi = masks[i]
        for j in range(i + 1, N):
            if mi & (1 << j):
                common = mi & masks[j]
                for k in range(j + 1, N):
                    if common & (1 << k):
                        s = deg[i] + deg[j] + deg[k]
                        tris.append((i, j, k, s))
    tris.sort(key=lambda x: -x[3])
    return tris[:limit]

def _local_search(masks, deg, edge_set, max_iters=2000000, time_limit=None):
    """Local search with bitmask representation. Returns (score, masks, edge_set)."""
    start = time.time()
    best_score = _compute_score(masks)
    best_masks = list(masks)
    best_es = set(edge_set)
    rng = random.Random(12345)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    # Precompute bit for each vertex
    bits = [1 << i for i in range(N)]
    
    it = 0
    no_improve = 0
    while it < max_iters:
        if time_limit is not None and time.time() > time_limit:
            break
        it += 1
        
        if it % 2000 == 0 and time.time() > _GUARD:
            break
        
        # Choose move: biased toward removing edges in high triangles
        # Pick a random existing edge to remove
        if not edge_set:
            break
        # Bias: with probability 0.5, pick from edges in worst triangles
        if rng.random() < 0.5 and best_score > 0:
            tris = _find_worst_triangles(masks, deg, limit=20)
            if tris:
                i, j, k, s = rng.choice(tris)
                # Pick one of the 3 edges of this triangle to remove
                e_rm = rng.choice([(i, j), (i, k), (j, k)])
            else:
                e_rm = rng.choice(list(edge_set))
        else:
            e_rm = rng.choice(list(edge_set))
        
        u, v = e_rm
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            continue
        
        # Pick a non-edge to add
        # Bias: prefer adding edges between low-degree vertices
        e_add = None
        for _try in range(30):
            a = rng.randint(0, N - 1)
            b = rng.randint(0, N - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) == (u, v):
                continue
            if (a, b) in edge_set:
                continue
            e_add = (a, b)
            break
        if e_add is None:
            continue
        
        a, b = e_add
        
        # Apply flip
        masks[u] &= ~bits[v]
        masks[v] &= ~bits[u]
        masks[a] |= bits[b]
        masks[b] |= bits[a]
        edge_set.discard((u, v))
        edge_set.add((a, b))
        
        # Update degrees
        deg[u] -= 1
        deg[v] -= 1
        deg[a] += 1
        deg[b] += 1
        
        # Recompute score
        new_score = _compute_score(masks)
        
        if new_score < best_score:
            best_score = new_score
            best_masks = list(masks)
            best_es = set(edge_set)
            no_improve = 0
        else:
            # Revert
            masks[u] |= bits[v]
            masks[v] |= bits[u]
            masks[a] &= ~bits[b]
            masks[b] &= ~bits[a]
            edge_set.discard((a, b))
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
            deg[a] -= 1
            deg[b] -= 1
            no_improve += 1
            
            if no_improve > 50000:
                # Perturb: do 5 random flips
                for _p in range(5):
                    if not edge_set:
                        break
                    e_rm2 = rng.choice(list(edge_set))
                    u2, v2 = e_rm2
                    if u2 > v2:
                        u2, v2 = v2, u2
                    e_add2 = None
                    for _try2 in range(50):
                        aa = rng.randint(0, N - 1)
                        bb = rng.randint(0, N - 1)
                        if aa == bb:
                            continue
                        if aa > bb:
                            aa, bb = bb, aa
                        if (aa, bb) == (u2, v2) or (aa, bb) in edge_set:
                            continue
                        e_add2 = (aa, bb)
                        break
                    if e_add2 is None:
                        continue
                    aa, bb = e_add2
                    masks[u2] &= ~bits[v2]
                    masks[v2] &= ~bits[u2]
                    masks[aa] |= bits[bb]
                    masks[bb] |= bits[aa]
                    edge_set.discard((u2, v2))
                    edge_set.add((aa, bb))
                    deg[u2] -= 1
                    deg[v2] -= 1
                    deg[aa] += 1
                    deg[bb] += 1
                no_improve = 0
    
    return best_score, best_masks, best_es

def _targeted_repair(masks, deg, edge_set, max_iters=100000):
    """Focused repair: only remove edges from worst triangles, add safe edges."""
    start = time.time()
    bits = [1 << i for i in range(N)]
    rng = random.Random(777)
    best_score = _compute_score(masks)
    best_masks = list(masks)
    best_es = set(edge_set)
    
    it = 0
    while it < max_iters:
        if time.time() > _GUARD:
            break
        it += 1
        
        tris = _find_worst_triangles(masks, deg, limit=10)
        if not tris:
            break
        
        i, j, k, s = tris[0]
        if s <= best_score - 1:
            break
        
        # Try removing each edge of this triangle
        candidates = [(i, j), (i, k), (j, k)]
        rng.shuffle(candidates)
        
        for e_rm in candidates:
            u, v = e_rm
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                continue
            
            # Find best edge to add: one that minimizes new triangle creation
            best_add = None
            best_add_score = best_score + 100
            
            # Sample candidate non-edges
            for _try in range(20):
                a = rng.randint(0, N - 1)
                b = rng.randint(0, N - 1)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) == (u, v) or (a, b) in edge_set:
                    continue
                
                # Estimate score change
                # Adding (a,b) creates triangles (a,b,x) for x in common neighbors
                # Removing (u,v) destroys triangles (u,v,x) for x in common neighbors
                new_deg = list(deg)
                new_deg[u] -= 1
                new_deg[v] -= 1
                new_deg[a] += 1
                new_deg[b] += 1
                
                # Quick estimate: just check triangles involving a or b
                est_score = 0
                # Triangles with new edge (a,b)
                common_ab = masks[a] & masks[b]
                for x in range(N):
                    if x != a and x != b and (common_ab & bits[x]):
                        est = new_deg[a] + new_deg[b] + new_deg[x]
                        if est > est_score:
                            est_score = est
                # Also check if removing (u,v) was the only thing keeping some triangle down
                # (approximate: just use est_score as lower bound)
                
                if est_score < best_add_score:
                    best_add_score = est_score
                    best_add = (a, b)
            
            if best_add is None:
                continue
            
            a, b = best_add
            
            # Apply
            masks[u] &= ~bits[v]
            masks[v] &= ~bits[u]
            masks[a] |= bits[b]
            masks[b] |= bits[a]
            edge_set.discard((u, v))
            edge_set.add((a, b))
            deg[u] -= 1
            deg[v] -= 1
            deg[a] += 1
            deg[b] += 1
            
            new_score = _compute_score(masks)
            
            if new_score < best_score:
                best_score = new_score
                best_masks = list(masks)
                best_es = set(edge_set)
                break  # restart triangle finding
            else:
                # Revert
                masks[u] |= bits[v]
                masks[v] |= bits[u]
                masks[a] &= ~bits[b]
                masks[b] &= ~bits[a]
                edge_set.discard((a, b))
                edge_set.add((u, v))
                deg[u] += 1
                deg[v] += 1
                deg[a] -= 1
                deg[b] -= 1
    
    return best_score, best_masks, best_es

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Initial construction: K_{6,10} + 5 matching edges in B
    edge_set = set()
    for u in range(6):
        for v in range(6, 16):
            edge_set.add((u, v))
    for a, b in [(6,7),(8,9),(10,11),(12,13),(14,15)]:
        edge_set.add((a, b))
    
    masks = _edges_to_mask(edge_set)
    deg = _deg_from_masks(masks)
    best_score = _compute_score(masks)
    best_masks = list(masks)
    best_es = set(edge_set)
    
    # Main local search
    score, masks_new, es_new = _local_search(masks, deg, edge_set, max_iters=3000000)
    if score < best_score:
        best_score = score
        best_masks = masks_new
        best_es = es_new
    
    # Targeted repair
    deg_r = _deg_from_masks(best_masks)
    score2, masks_r, es_r = _targeted_repair(best_masks, deg_r, best_es, max_iters=200000)
    if score2 < best_score:
        best_score = score2
        best_masks = masks_r
        best_es = es_r
    
    # Second round of local search from improved state
    if time.time() < _GUARD - 10:
        deg2 = _deg_from_masks(best_masks)
        score3, masks3, es3 = _local_search(best_masks, deg2, best_es, max_iters=2000000)
        if score3 < best_score:
            best_score = score3
            best_masks = masks3
            best_es = es3
    
    # Final targeted repair
    if time.time() < _GUARD - 5:
        deg3 = _deg_from_masks(best_masks)
        score4, masks4, es4 = _targeted_repair(best_masks, deg3, best_es, max_iters=100000)
        if score4 < best_score:
            best_score = score4
            best_masks = masks4
            best_es = es4
    
    # Final polish: try single-edge swaps exhaustively near the solution
    if time.time() < _GUARD - 2:
        deg_p = _deg_from_masks(best_masks)
        bits = [1 << i for i in range(N)]
        improved = True
        while improved and time.time() < _GUARD - 2:
            improved = False
            for e_rm in list(best_es):
                u, v = e_rm
                if u > v:
                    u, v = v, u
                for a in range(N):
                    for b in range(a + 1, N):
                        if (a, b) == (u, v) or (a, b) in best_es:
                            continue
                        # Try swap
                        best_masks[u] &= ~bits[v]
                        best_masks[v] &= ~bits[u]
                        best_masks[a] |= bits[b]
                        best_masks[b] |= bits[a]
                        best_es.discard((u, v))
                        best_es.add((a, b))
                        new_score = _compute_score(best_masks)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                        else:
                            # Revert
                            best_masks[u] |= bits[v]
                            best_masks[v] |= bits[u]
                            best_masks[a] &= ~bits[b]
                            best_masks[b] &= ~bits[a]
                            best_es.discard((a, b))
                            best_es.add((u, v))
                        if time.time() > _GUARD - 2:
                            break
                    if time.time() > _GUARD - 2:
                        break
                if time.time() > _GUARD - 2:
                    break
    
    # Verify
    final_edges = _mask_to_edges(best_masks)
    assert len(final_edges) == target_edges, f"Edge count {len(final_edges)} != {target_edges}"
    final_masks = _edges_to_mask(final_edges)
    final_score = _compute_score(final_masks)
    if final_score != best_score:
        best_score = final_score
        best_masks = final_masks
        best_es = final_edges
    
    edges = [[u, v] for u, v in sorted(best_es)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
