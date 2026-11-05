import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    N = 256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets
    # adj[i] contains j (0-indexed) such that (i+1) + (j+1) is a perfect square
    adj = [set() for _ in range(N)]
    
    # Identify vertices with self-loops (2*v is a square)
    # These vertices cannot be in the solution set.
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i].add(j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Precompute degrees for greedy ordering
    degrees = [len(adj[i]) for i in range(N)]
    
    # Bitset representation for faster intersection checks
    # candidate_mask: bits set for all candidates
    # adj_mask[i]: bits set for neighbors of i
    candidate_mask = 0
    for i in candidates:
        candidate_mask |= (1 << i)
        
    adj_mask = [0] * N
    for i in range(N):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    best_size = 0
    best_set = frozenset()
    
    def greedy_solve(order=None):
        """
        Greedy construction. If order is None, use a randomized degree-based order.
        """
        if order is None:
            # Sort by degree with random tie-breaking
            # This tends to pick "less constrained" vertices first
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
        
        current_set = set()
        current_mask = 0
        
        for v in order:
            # Check if v conflicts with any u in current_set
            # v cannot conflict with itself (self-loops are removed)
            # We just need to ensure no neighbor is in current_set
            if not (adj_mask[v] & current_mask):
                current_set.add(v)
                current_mask |= (1 << v)
                
        return current_set, current_mask

    def local_improve(s, mask):
        """Try to add vertices that don't conflict."""
        s = set(s)
        improved = True
        # Iterate over candidates not in s
        # We can optimize by maintaining a list of available candidates, 
        # but N=256 is small enough that iterating all candidates is fine.
        while improved:
            improved = False
            for v in candidates:
                if v in s:
                    continue
                if not (adj_mask[v] & mask):
                    s.add(v)
                    mask |= (1 << v)
                    improved = True
        return s, mask

    def local_search_1swap(s, mask):
        """
        Try to remove one vertex and refill.
        """
        s = set(s)
        mask = mask
        improved = True
        iter_count = 0
        while improved and iter_count < 50 and (time.time() - start_time < DEADLINE - 1):
            improved = False
            iter_count += 1
            vertices_to_try = list(s)
            random.shuffle(vertices_to_try)
            for v in vertices_to_try:
                # Try removing v
                s.remove(v)
                mask ^= (1 << v)
                # Try to fill up again
                s_new, mask_new = local_improve(s, mask)
                if len(s_new) > len(s) + 1: # Must gain more than we lost
                    s = s_new
                    mask = mask_new
                    improved = True
                    break # Restart the loop with new s
                else:
                    # Restore v and try to fill (in case removal opened up space but not enough)
                    s.add(v)
                    mask |= (1 << v)
                    s, mask = local_improve(s, mask)
        
        return s, mask

    def local_search_2swap(s, mask):
        """
        Try to remove two vertices and refill.
        This can escape local minima where single swaps fail.
        """
        s = set(s)
        mask = mask
        improved = True
        iter_count = 0
        while improved and iter_count < 10 and (time.time() - start_time < DEADLINE - 1):
            improved = False
            iter_count += 1
            vertices_to_try = list(s)
            random.shuffle(vertices_to_try)
            
            # Try pairs
            # To limit complexity, we don't try ALL pairs if the set is large,
            # but N=256 means max size is ~100-150. Pairs are ~10k-20k checks.
            # Each check involves a local_improve which is fast.
            for i in range(len(vertices_to_try)):
                v1 = vertices_to_try[i]
                for j in range(i + 1, len(vertices_to_try)):
                    v2 = vertices_to_try[j]
                    
                    # Skip if they are adjacent? No, removing adjacent vertices is often useful.
                    
                    # Try removing v1 and v2
                    s.remove(v1)
                    s.remove(v2)
                    mask ^= (1 << v1)
                    mask ^= (1 << v2)
                    
                    # Try to fill up again
                    s_new, mask_new = local_improve(s, mask)
                    
                    # We lost 2, so we need to gain > 2, i.e., size > len(s) + 2
                    # Wait, len(s) is the size BEFORE removal.
                    # Current size in s is len(original) - 2.
                    # If len(s_new) > len(original) - 2 + 2 = len(original), we have an improvement.
                    # Actually, simpler: if len(s_new) > len(s) + 2, we gained more than we lost.
                    if len(s_new) > len(s) + 2:
                        s = s_new
                        mask = mask_new
                        improved = True
                        break
                    else:
                        # Restore
                        s.add(v1)
                        s.add(v2)
                        mask |= (1 << v1)
                        mask |= (1 << v2)
                        
                if improved:
                    break # Restart loop with new s
                    # Also, we should probably re-shuffle or just break out of the pair loop
                    # The outer while loop will handle re-shuffling if we continue
        
        return s, mask

    # Iterated Local Search / Random Restart Greedy
    while time.time() - start_time < DEADLINE:
        # Strategy 1: Random order
        if random.random() < 0.5:
            order = candidates[:]
            random.shuffle(order)
        else:
            # Strategy 2: Degree-based order with random tie-breaking
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
            
        s, mask = greedy_solve(order)
        s, mask = local_improve(s, mask)
        
        # Apply local searches
        if time.time() - start_time < DEADLINE - 2:
            s, mask = local_search_1swap(s, mask)
            
        if time.time() - start_time < DEADLINE - 1:
            s, mask = local_search_2swap(s, mask)
            
        if len(s) > best_size:
            best_size = len(s)
            best_set = frozenset(s)
            
    # Convert 0-based indices back to 1-based values
    return sorted([v + 1 for v in best_set])

# EVOLVE-BLOCK-END
