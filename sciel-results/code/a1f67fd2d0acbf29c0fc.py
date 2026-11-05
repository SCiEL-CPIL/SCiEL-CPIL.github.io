"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_MASKS_CACHE = {}

def _get_masks(n, size):
    """Precompute masks for subsets of a specific size."""
    key = (n, size)
    if key not in _MASKS_CACHE:
        masks = []
        for combo in combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks.append(mask)
        _MASKS_CACHE[key] = masks
    return _MASKS_CACHE[key]

def _is_regular_induced_fast(adj, mask):
    """Check if induced subgraph on mask is regular."""
    count = mask.bit_count()
    if count < 6:
        return False
    m = mask
    first_deg = -1
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        deg = (adj[idx] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _find_one_regular_subgraph(adj, n, time_budget=0.5):
    """Find any regular induced subgraph of size >= 6."""
    start = time.time()
    # Check sizes in increasing order
    for size in range(6, n + 1):
        if time.time() - start > time_budget:
            break
        masks = _get_masks(n, size)
        for mask in masks:
            if time.time() - start > time_budget:
                break
            if _is_regular_induced_fast(adj, mask):
                return mask
    return None

def _extract_vertices(mask):
    """Extract list of vertex indices from mask."""
    verts = []
    m = mask
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        verts.append(idx)
        m ^= lb
    return verts

def _count_regular_subgraphs(adj, n, max_check=200):
    """Count how many regular subgraphs of size 6 exist (for fitness)."""
    count = 0
    masks = _get_masks(n, 6)
    checked = 0
    for mask in masks:
        if checked >= max_check:
            break
        if _is_regular_induced_fast(adj, mask):
            count += 1
        checked += 1
    return count

def _local_search(adj, n, deadline, rng):
    """Local search: flip edges to eliminate regular induced subgraphs."""
    adj = adj[:]  # copy
    max_steps = 300
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Find a regular subgraph
        mask = _find_one_regular_subgraph(adj, n, time_budget=0.2)
        
        if mask is None:
            # Verify it's truly valid (no regular subgraph of any size >= 6)
            if _find_one_regular_subgraph(adj, n, time_budget=0.8) is None:
                return adj, True
            # Otherwise continue searching
            continue
        
        # Found a regular subgraph. Flip an edge to break regularity.
        verts = _extract_vertices(mask)
        if len(verts) < 2:
            continue
        
        # Compute degrees within the induced subgraph for better edge selection
        degs = {}
        for v in verts:
            degs[v] = (adj[v] & mask).bit_count()
        
        # Try multiple random edge flips and pick the best one
        best_adj = None
        best_score = float('inf')
        
        # Prioritize edges that connect vertices with same degree (more likely to maintain regularity)
        same_deg_pairs = []
        diff_deg_pairs = []
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if degs[u] == degs[v]:
                    same_deg_pairs.append((u, v))
                else:
                    diff_deg_pairs.append((u, v))
        
        # Try same-degree pairs first (flipping these is more likely to break regularity)
        candidates = same_deg_pairs[:3] + diff_deg_pairs[:2]
        if len(candidates) < 5:
            all_pairs = same_deg_pairs + diff_deg_pairs
            while len(candidates) < 5 and all_pairs:
                candidates.append(all_pairs.pop(rng.randrange(len(all_pairs))))
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Measure fitness: count regular subgraphs of size 6
            score = _count_regular_subgraphs(adj, n, max_check=80)
            
            if score < best_score:
                best_score = score
                best_adj = adj[:]
            
            # Undo flip
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        
        # Apply the best flip
        if best_adj is not None:
            adj = best_adj
            
    # Final check
    if _find_one_regular_subgraph(adj, n, time_budget=0.8) is None:
        return adj, True
    return adj, False

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_n = 5
    best_edges = []
    rng = random.Random(42)
    
    # Try to find a valid graph for n=20 down to 15
    for target_n in range(20, 14, -1):
        if time.time() > deadline - 15:
            break
            
        n = target_n
        
        # Number of restarts depends on remaining time
        remaining = deadline - time.time()
        max_restarts = max(3, int(remaining / 12))
        
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
                
            # Use different initialization strategies based on restart number
            if r == 0:
                # Sparse random graph (lower edge probability)
                edge_prob = 0.3
            elif r == 1:
                # Medium density
                edge_prob = 0.4
            elif r == 2:
                # Use a constructive approach: start from a cycle and add random edges
                edge_prob = 0.35
            else:
                # Random density
                edge_prob = rng.uniform(0.25, 0.45)
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < edge_prob:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # For some restarts, add a structured component to break symmetry
            if r >= 2:
                # Add a random transitive permutation structure
                perm = list(range(n))
                rng.shuffle(perm)
                for k in range(n):
                    u = perm[k]
                    v = perm[(k + 1) % n]
                    # Add edge with some probability to create asymmetry
                    if rng.random() < 0.3:
                        if (adj[u] >> v) & 1:
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        
            # Local Search
            adj, found_valid = _local_search(adj, n, deadline - 15, rng)
            
            if found_valid:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break
        
        if best_n == target_n:
            break
    
    # Fallback if no graph > 5 found
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
