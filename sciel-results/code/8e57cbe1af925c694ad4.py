import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def elapsed():
        return time.time() - start_time

    def _precompute_masks(n, k):
        """Pre-compute all k-subsets as bitmasks and vertex lists."""
        masks = []
        vertex_lists = []
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks.append(mask)
            vertex_lists.append(comb)
        return masks, vertex_lists

    def _is_regular(adj, mask, vertices):
        """Check if the induced subgraph on 'vertices' is regular."""
        if not vertices:
            return False
        first_deg = (adj[vertices[0]] & mask).bit_count()
        for v in vertices[1:]:
            if (adj[v] & mask).bit_count() != first_deg:
                return False
        return True

    def _find_bad_subset_fast(adj, masks, verts):
        """Find any regular induced subgraph from the pre-computed lists."""
        for i, mask in enumerate(masks):
            if _is_regular(adj, mask, verts[i]):
                return verts[i]
        return None

    def _find_any_bad_subset_full(adj, n):
        """Check all sizes >= 6. Used for final verification."""
        for size in range(6, n + 1):
            masks, verts = _precompute_masks(n, size)
            bad = _find_bad_subset_fast(adj, masks, verts)
            if bad is not None:
                return bad
        return None

    def _flip_edge(adj, u, v):
        bit_v = 1 << v
        bit_u = 1 << u
        if adj[u] & bit_v:
            adj[u] &= ~bit_v
            adj[v] &= ~bit_u
        else:
            adj[u] |= bit_v
            adj[v] |= bit_u

    def _to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def _solve_for_n(n, seed):
        rng = random.Random(seed)
        adj = [0] * n
        
        # Initialize with random graph (p=0.5)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Pre-compute masks for size 6 and 7 for fast local search
        masks6, verts6 = _precompute_masks(n, 6)
        masks7, verts7 = _precompute_masks(n, 7)

        # Local Search
        max_iters = 2000
        for _ in range(max_iters):
            if elapsed() > deadline - 5:
                return None
            
            # Check size 6 first
            bad = _find_bad_subset_fast(adj, masks6, verts6)
            if bad is None:
                # Check size 7
                bad = _find_bad_subset_fast(adj, masks7, verts7)
            
            if bad is None:
                # No small violations found, assume valid for now (will verify later)
                return adj
            
            # Flip a random edge within the bad subset
            u, v = rng.sample(bad, 2)
            _flip_edge(adj, u, v)
            
        return None

    # Try n=20, then 19, then 18
    best_graph = None
    best_n = 0
    
    for n in [20, 19, 18]:
        if elapsed() > deadline - 15:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(20):
            if elapsed() > deadline - 5:
                break
            
            result = _solve_for_n(n, seed)
            if result is not None:
                # Final verification: check ALL sizes >= 6
                if _find_any_bad_subset_full(result, n) is None:
                    best_graph = result
                    best_n = n
                    found = True
                    break
        
        if found:
            break
    
    if best_graph is None:
        # Fallback: n=5 (trivially no regular induced subgraph of size 6)
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': _to_edges(best_graph, best_n)}
