import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s for cleanup/return

    def elapsed():
        return time.time() - start_time

    def _is_regular(adj, mask, vertices):
        """Check if the induced subgraph on 'vertices' (with bitmask 'mask') is regular."""
        if not vertices:
            return False
        first_deg = (adj[vertices[0]] & mask).bit_count()
        for v in vertices[1:]:
            if (adj[v] & mask).bit_count() != first_deg:
                return False
        return True

    def _find_bad_subsets_6_7(adj, n):
        """Find all regular induced subgraphs of size 6 or 7.
        Returns a list of tuples (size, vertices_tuple)."""
        bads = []
        # Check size 6
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular(adj, mask, comb):
                bads.append((6, comb))
        # Check size 7
        for comb in combinations(range(n), 7):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular(adj, mask, comb):
                bads.append((7, comb))
        return bads

    def _full_verify(adj, n):
        """Check if ANY subset of size >= 6 is regular. Returns True if valid."""
        # We already know 6 and 7 are clean from the search loop, but we must check larger ones.
        # However, for n <= 22, checking all sizes is expensive.
        # But if we found a graph where no 6 or 7 subset is regular, it is *very* likely valid.
        # We still need to check 8..n to be sure.
        for size in range(8, n + 1):
            for comb in combinations(range(n), size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular(adj, mask, comb):
                    return False
        return True

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

    def _count_bad_6_involving_edge(adj, n, u, v):
        """Estimate how many 6-subsets containing edge (u,v) or its flip would be regular.
        This is an approximation to guide the flip."""
        # This function is a heuristic. We just check if flipping (u,v) creates new 6-regular subsets.
        # Actually, it's cheaper to just try the flip, check bads, and revert if worse.
        # But checking all bads is expensive.
        # Let's stick to a simpler heuristic: Flip a random edge in the bad subset.
        # To improve: Try 3 random edges in the bad subset, pick the one that leaves the fewest bad 6-subsets.
        pass

    def _solve(n, time_budget, seed):
        rng = random.Random(seed)
        target_time = time.time() + time_budget
        
        best_adj = None
        best_bad_count = float('inf')
        
        # Try a few different initial densities
        for p in [0.4, 0.5, 0.6]:
            if time.time() > target_time or elapsed() > deadline - 2:
                break
                
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            max_iters = 10000
            stuck_counter = 0
            for _ in range(max_iters):
                if time.time() > target_time or elapsed() > deadline - 2:
                    break
                
                bads = _find_bad_subsets_6_7(adj, n)
                
                if not bads:
                    # No 6 or 7 bad subsets. Check full validity.
                    if _full_verify(adj, n):
                        return adj
                    else:
                        # There is a bad subset of size >= 8.
                        # We need to find it to fix it.
                        # This is rare if 6 and 7 are clean.
                        # For now, we treat it as a "stuck" state and try to perturb.
                        # We don't know which edge to flip, so we do a random flip.
                        u, v = rng.sample(range(n), 2)
                        _flip_edge(adj, u, v)
                        stuck_counter += 1
                        if stuck_counter > 100:
                            break # Restart this density
                        continue

                bad_count = len(bads)
                if bad_count < best_bad_count:
                    best_bad_count = bad_count
                    best_adj = adj[:] # Store copy
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    if stuck_counter > 500:
                        break # Restart this density

                # Pick a random bad subset to fix
                size, comb = rng.choice(bads)
                
                # Try to fix by flipping an edge within 'comb'
                # Heuristic: Try a few random edges in 'comb' and pick the one that results in the fewest new 6-bads
                best_flip = None
                min_bad_after = float('inf')
                
                # Sample up to 5 random pairs from the bad subset
                candidates = rng.sample(list(combinations(comb, 2)), min(5, len(list(combinations(comb, 2)))))
                
                for u, v in candidates:
                    # Tentative flip
                    _flip_edge(adj, u, v)
                    
                    # Quick check: count 6-bads
                    # We only need to check 6-subsets that are affected by the flip.
                    # But _find_bad_subsets_6_7 is O(n^6) or O(n^7). For n=20, this is ~ 38k + 77k = 115k checks.
                    # Each check is fast. 115k * 5 candidates = 500k checks per iteration.
                    # This might be too slow if we do it for 10k iterations.
                    # Optimization: Only check 6-bads.
                    bads_after = _find_bad_subsets_6_7(adj, n)
                    count_after = len(bads_after)
                    
                    # Revert
                    _flip_edge(adj, u, v)
                    
                    if count_after < min_bad_after:
                        min_bad_after = count_after
                        best_flip = (u, v)
                
                if best_flip:
                    _flip_edge(adj, best_flip[0], best_flip[1])
        
        # If we didn't find a perfect graph, return the best we found?
        # No, we must return a valid graph.
        # If best_adj has no 6/7 bads, it might still have 8+ bads.
        # We should verify best_adj.
        if best_adj and _full_verify(best_adj, n):
            return best_adj
            
        return None

    # Try n=21 first (if we succeed, score is huge)
    # Then n=20
    # Then n=19
    
    best_graph = None
    best_n = 0
    
    # Allocate time wisely.
    # n=21 is very hard. Give it 20s.
    # n=20 is known to exist. Give it 60s.
    # n=19 is easier. Give it 20s.
    # n=18 is easy. Give it 10s.
    
    time_alloc = {
        21: 20,
        20: 60,
        19: 20,
        18: 10,
        17: 10
    }
    
    for n in [21, 20, 19, 18, 17]:
        if elapsed() > deadline - 5:
            break
            
        budget = time_alloc.get(n, 5)
        # Adjust budget if previous steps took longer than expected
        if elapsed() + budget > deadline - 5:
            budget = max(0, deadline - 5 - elapsed())
            if budget < 2:
                continue
                
        # Try multiple seeds for this n
        seeds = range(100) if n >= 20 else range(50)
        
        for seed in seeds:
            if elapsed() > deadline - 5:
                break
            # Give each seed a small slice of the budget
            seed_budget = min(1.0, budget / 10) # ~1s per seed
            g = _solve(n, seed_budget, seed)
            if g is not None:
                # Final verification just in case
                if _full_verify(g, n):
                    best_graph = g
                    best_n = n
                    break # Found a valid graph for this n, move to next n (or return if max)
        
        if best_n == n:
            # If we found a graph for this n, we can stop if n==21 or 20?
            # The task says "Maximize n". So we should try higher n first.
            # Since we iterate 21 -> 20 -> ..., if we find 21, we return.
            # If we find 20, we return (since 21 failed).
            # Wait, the loop structure above tries 21, then 20.
            # If 21 fails, it tries 20. If 20 succeeds, it sets best_n=20.
            # Then the loop ends (or we break).
            # We should break if we found a valid graph for the current n, 
            # because we are iterating downwards.
            break

    if best_graph is None:
        # Fallback: n=5
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': _to_edges(best_graph, best_n)}
