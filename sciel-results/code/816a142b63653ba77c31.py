import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    def _is_regular(adj, mask, vertices):
        """Check if the induced subgraph on 'vertices' (with bitmask 'mask') is regular."""
        if not vertices:
            return False
        first_deg = (adj[vertices[0]] & mask).bit_count()
        for v in vertices[1:]:
            if (adj[v] & mask).bit_count() != first_deg:
                return False
        return True

    def _find_bad_subset(adj, n):
        """Find any regular induced subgraph of size >= 6. Return (size, vertices) or None."""
        # Check size 6 first, as it's the most common obstruction
        # For efficiency, we can just iterate combinations of size 6.
        # If a larger subset is regular, it contains a 6-subset that might not be regular,
        # but checking all sizes is safer. However, checking size 6 is the bottleneck.
        # Let's check sizes 6 to n.
        for size in range(6, n + 1):
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

    def _solve_local_search(n, time_budget, seed):
        rng = random.Random(seed)
        target_time = time.time() + time_budget
        
        # Try a few different initial densities
        for p in [0.5, 0.4, 0.6, 0.3, 0.7]:
            if time.time() > target_time or time.time() > deadline - 2:
                break
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            max_iters = 2000
            for _ in range(max_iters):
                if time.time() > target_time or time.time() > deadline - 2:
                    break
                
                bad = _find_bad_subset(adj, n)
                if bad is None:
                    return adj
                
                # Try to fix the bad subset
                # Strategy: Flip a random edge within the bad subset
                # Or flip an edge connecting a vertex in the subset to one outside?
                # Flipping an edge inside is more likely to break regularity of that subset.
                u, v = rng.sample(bad, 2)
                _flip_edge(adj, u, v)
                
                # If we still have bad subsets, continue
                # We could also try to flip edges outside to prevent formation of new ones,
                # but fixing the current one is priority.
        
        return None

    def _solve_greedy(n, time_budget, seed):
        rng = random.Random(seed)
        target_time = time.time() + time_budget
        
        adj = [0] * n
        edges = []
        
        all_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                all_edges.append((u, v))
        
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > target_time or time.time() > deadline - 2:
                break
            
            # Tentatively add edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check validity
            # Optimization: Only check subsets containing u or v?
            # Any new regular subset must contain u or v (since only edges incident to u/v changed).
            # Actually, adding an edge (u,v) can only create a regular induced subgraph that contains u and v.
            # It cannot create one that doesn't contain u or v.
            # So we only need to check subsets S of size >= 6 such that u in S and v in S.
            
            valid = True
            # We need to check all subsets S of size k >= 6 that contain u and v.
            # This is still expensive but less than all subsets.
            # For n=20, subsets of size 6 containing u,v: C(18, 4) = 3060.
            # For size 7: C(18, 5) = 8568.
            # Sum is manageable.
            
            for size in range(6, n + 1):
                # Choose size-2 other vertices from the remaining n-2
                for others in combinations([x for x in range(n) if x != u and x != v], size - 2):
                    comb = tuple(sorted((u, v) + others))
                    mask = 0
                    for v_idx in comb:
                        mask |= (1 << v_idx)
                    if _is_regular(adj, mask, comb):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                edges.append([u, v])
            else:
                # Remove the edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return adj

    # Search Strategy
    # Try n=21 first, then 20, then 19
    # Use local search for n=21 (harder), greedy for n=20 (more likely to succeed)
    
    # 1. Try n=21 with local search
    if time.time() < deadline - 30:
        result = _solve_local_search(21, 25, seed=42)
        if result is not None:
            return {'n': 21, 'edges': _to_edges(result, 21)}
            
    # 2. Try n=20 with local search
    if time.time() < deadline - 30:
        result = _solve_local_search(20, 25, seed=123)
        if result is not None:
            return {'n': 20, 'edges': _to_edges(result, 20)}
            
    # 3. Try n=20 with greedy (multiple seeds)
    if time.time() < deadline - 10:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            result = _solve_greedy(20, 10, seed=seed)
            # Greedy might not be fully valid if it stopped early, so we must verify
            if result is not None:
                if _find_bad_subset(result, 20) is None:
                    return {'n': 20, 'edges': _to_edges(result, 20)}

    # 4. Fallback to n=19
    if time.time() < deadline - 5:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 2:
                break
            result = _solve_local_search(19, 2, seed=seed)
            if result is not None:
                return {'n': 19, 'edges': _to_edges(result, 19)}
            
    # 5. Fallback to n=18
    if time.time() < deadline - 2:
        result = _solve_local_search(18, 1, seed=99)
        if result is not None:
            return {'n': 18, 'edges': _to_edges(result, 18)}
            
    # Fallback: return a known valid small graph
    # n=5 is valid
    return {'n': 5, 'edges': []}
