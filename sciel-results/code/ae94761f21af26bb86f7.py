import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def elapsed():
        return time.time() - start_time

    def _is_regular(adj, mask, vertices):
        if not vertices:
            return False
        first_deg = (adj[vertices[0]] & mask).bit_count()
        for v in vertices[1:]:
            if (adj[v] & mask).bit_count() != first_deg:
                return False
        return True

    def _find_bad_subset_fast(adj, n):
        """Check only size-6 subsets first (most common obstruction)."""
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular(adj, mask, comb):
                return comb
        # Check size 7+
        for size in range(7, n + 1):
            for comb in combinations(range(n), size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular(adj, mask, comb):
                    return comb
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

    def _greedy_build(n, seed):
        """Build graph vertex-by-vertex, greedily adding edges while avoiding regular 6-subsets."""
        rng = random.Random(seed)
        adj = [0] * n
        
        # Process vertices in random order
        order = list(range(n))
        rng.shuffle(order)
        
        for idx, v in enumerate(order):
            # For each new vertex, decide edges to previous vertices
            for prev_idx in range(idx):
                u = order[prev_idx]
                # Randomly decide whether to add edge
                if rng.random() < 0.5:
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
                    # Check if this creates a regular 6-subset containing v
                    # Only need to check subsets of size 6 that include v and 5 from previous
                    # This is expensive, so only check every few steps
                    if (idx + 1) >= 6 and (idx + 1) % 3 == 0:
                        # Quick check: does any 6-subset containing v become regular?
                        prev_vertices = order[:idx]
                        # Sample a few random 5-subsets from previous vertices
                        for _ in range(5):
                            if len(prev_vertices) < 5:
                                break
                            sample5 = rng.sample(prev_vertices, 5)
                            comb = sample5 + [v]
                            mask = 0
                            for w in comb:
                                mask |= (1 << w)
                            if _is_regular(adj, mask, comb):
                                # Revert
                                adj[v] &= ~(1 << u)
                                adj[u] &= ~(1 << v)
                                break
        
        return adj

    def _local_search(adj, n, rng, max_iters=2000):
        """Local search with fast bad-subset detection."""
        for _ in range(max_iters):
            if elapsed() > deadline - 5:
                break
            bad = _find_bad_subset_fast(adj, n)
            if bad is None:
                return adj
            # Flip a random edge within the bad subset
            u, v = rng.sample(bad, 2)
            _flip_edge(adj, u, v)
        return None

    # Strategy: Try greedy build + local search for n=20, then 19, then 18
    best_graph = None
    best_n = 0
    
    for n in [20, 19, 18]:
        if elapsed() > deadline - 10:
            break
        
        for seed in range(50):
            if elapsed() > deadline - 5:
                break
            
            rng = random.Random(seed)
            adj = _greedy_build(n, seed)
            
            # Local search to fix any remaining bad subsets
            result = _local_search(adj, n, rng, max_iters=1000)
            if result is not None:
                # Final verification
                if _find_bad_subset_fast(result, n) is None:
                    best_graph = result
                    best_n = n
                    break
        
        if best_n > 0:
            break
    
    if best_graph is None:
        # Fallback: n=5
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': _to_edges(best_graph, best_n)}
