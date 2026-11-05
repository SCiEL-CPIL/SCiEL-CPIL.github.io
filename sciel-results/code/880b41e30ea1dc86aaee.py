import time
import random
from itertools import combinations

def _popcount(x):
    return x.bit_count()

def _is_regular_subset(adj_mask, subset_mask, n):
    """Check if the induced subgraph on subset_mask is regular."""
    k = _popcount(subset_mask)
    if k < 6:
        return True
    first_deg = -1
    for i in range(n):
        if subset_mask & (1 << i):
            deg = _popcount(adj_mask[i] & subset_mask)
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
    return True

def _find_regular_subset(adj_mask, n):
    """Find any subset of size >= 6 that induces a regular graph.
    Returns (mask, degree) or (0, -1) if none found."""
    # Check sizes from 6 up to n
    # Optimization: If a subset of size k is regular, it's bad.
    # We check all subsets. This is expensive, so we do it carefully.
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subset(adj_mask, mask, n):
                # Find the degree
                first_v = next(v for v in range(n) if mask & (1 << v))
                deg = _popcount(adj_mask[first_v] & mask)
                return mask, deg
    return 0, -1

def _construct_circulant(n, steps):
    """Construct a circulant graph with given steps."""
    edges = []
    adj_mask = [0] * n
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj_mask, edges

def _construct_random(n, seed):
    """Construct a random graph."""
    rng = random.Random(seed)
    edges = []
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append([i, j])
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj_mask, edges

def _local_search(adj_mask, n, edges, max_iters=1000, deadline=None):
    """Try to break all regular subsets by flipping edges."""
    # We need to maintain edge list consistency if we flip edges.
    # For speed, we work with adj_mask and reconstruct edges at the end if successful.
    # But we need to know which edges to flip.
    # Let's store edges as a set of frozensets for fast lookup/update? 
    # No, n<=22, so adj_mask is fast. We just need to track if we changed anything.
    # We will just modify adj_mask and return it. The caller will convert to edges.
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        mask, deg = _find_regular_subset(adj_mask, n)
        if mask == 0:
            return adj_mask, True
        
        # We found a regular subset 'mask' with degree 'deg'.
        # Try to break it by flipping an edge within the subset.
        # Get list of vertices in subset
        verts = [i for i in range(n) if mask & (1 << i)]
        
        # Try flipping a random edge within the subset
        found_flip = False
        for _ in range(10): # Try a few random flips
            if len(verts) < 2:
                break
            u, v = random.sample(verts, 2)
            # Check if edge (u,v) exists
            if adj_mask[u] & (1 << v):
                # Flip: remove edge
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
            else:
                # Flip: add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            
            # Check if this flip broke the regularity of 'mask'
            # Note: it might break it, or it might just change the degree.
            # If the subset is no longer regular, we are good for *this* subset.
            # But we need to re-check the whole graph.
            # To save time, just check if 'mask' is still regular.
            if not _is_regular_subset(adj_mask, mask, n):
                found_flip = True
                break
        
        if not found_flip:
            # If random flips didn't break it quickly, try a systematic one?
            # Or just break to avoid infinite loop on same bad subset
            # Actually, if we flip an edge, the subset degrees change.
            # If we flip (u,v), deg(u) and deg(v) change by +/- 1.
            # If all others were 'deg', now we have 'deg+1, deg-1, deg, deg...'
            # This is definitely not regular unless k=2 (but k>=6).
            # So any flip inside the subset MUST break regularity of that specific subset.
            # Why did found_flip stay False? 
            # Ah, `_is_regular_subset` returns True if k<6. But k>=6 here.
            # So if we flip an edge inside, it *should* break regularity.
            # Unless... the flip was not inside? No, we sampled from verts.
            # Let's force a flip if we found a subset.
            
            # Pick first two vertices
            u, v = verts[0], verts[1]
            if adj_mask[u] & (1 << v):
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
            else:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            # This flip guarantees 'mask' is no longer regular.
            # We proceed to next iteration to check for other subsets.

    # Final check
    mask, deg = _find_regular_subset(adj_mask, n)
    if mask == 0:
        return adj_mask, True
    return adj_mask, False

def _adj_to_edges(adj_mask, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_mask[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try n=21 first (to beat 20)
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        # Try several circulant structures
        # Common circulant graphs that might avoid regular subgraphs
        # We need to break symmetry.
        step_sets = [
            [1, 2, 3, 5],
            [1, 3, 4, 7],
            [1, 2, 5, 8],
            [1, 4, 6, 9],
            [2, 3, 5, 7],
            [1, 2, 4, 6, 9],
            [1, 3, 5, 7, 11]
        ]
        
        for steps in step_sets:
            if time.time() > deadline - 5:
                break
            # Filter steps to be < n/2 and unique
            valid_steps = [s for s in steps if 0 < s < n//2]
            if not valid_steps:
                continue
            
            adj_mask, edges = _construct_circulant(n, valid_steps)
            
            # Check if valid
            mask, deg = _find_regular_subset(adj_mask, n)
            if mask == 0:
                return {'n': n, 'edges': edges}
            
            # Try local search
            refined_adj, success = _local_search(adj_mask, n, edges, max_iters=50, deadline=deadline)
            if success:
                final_edges = _adj_to_edges(refined_adj, n)
                return {'n': n, 'edges': final_edges}
        
        # Try random graphs with local search
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            adj_mask, edges = _construct_random(n, seed + n)
            
            # Quick check
            mask, deg = _find_regular_subset(adj_mask, n)
            if mask == 0:
                return {'n': n, 'edges': edges}
            
            # Local search
            refined_adj, success = _local_search(adj_mask, n, edges, max_iters=100, deadline=deadline)
            if success:
                final_edges = _adj_to_edges(refined_adj, n)
                return {'n': n, 'edges': final_edges}
                
    # Fallback: Return a known valid small graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}
