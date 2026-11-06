import time
import random
import numpy as np

N = 64
TIME_LIMIT = 290.0  # Safety margin

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT

    # Precompute coordinates
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    total = N * N
    
    # Precompute squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = dx * dx + dy * dy
    
    # Precompute ring masks: ring_mask[i, d] is a bitset of points at distance d from i
    # We need to map distance values to indices or just use a dict of bitsets per point?
    # Max distance is 7938. 
    # Instead of a huge 3D array, we can precompute for each point i, a list of (distance, bitset)
    # Or, since we only check specific distances, maybe we can just use the dist matrix directly?
    # The bitset trick is: S_bitset & ring_mask[Q, d] != 0.
    # If we don't precompute ring masks, we can check: 
    # "Is there R in S such that dist[Q, R] == d?"
    # This is a search in the row dist[Q, :] restricted to S.
    # With bitsets, we can precompute:
    # For each point i, and each possible distance d, a bitset of points at that distance.
    
    # Number of unique distances is small.
    # Let's create a dictionary: dist_to_bitset[i][d] = bitset
    # But creating 4096 dictionaries might be slow.
    
    # Alternative: Precompute a 2D array of bitsets? 
    # Max dist ~ 8000. 4096 * 8000 * 8 bytes = 262 MB. Too much? Maybe.
    # Let's try a different approach for the Apex Q check.
    # For a fixed P and S:
    # We need to check for each Q in S if dist(P, Q) is equal to dist(Q, R) for some R in S.
    # This is equivalent to: For each Q in S, let d = dist(P, Q). 
    # Check if any other point in S is at distance d from Q.
    
    # Let's stick to the bitset precomputation but sparse.
    # For each point i, collect all points j and their distance d.
    # Group by d.
    
    ring_masks = [None] * total
    for i in range(total):
        # Get all distances from i
        di = dist[i]
        # Unique distances
        unique_d = np.unique(di)
        masks = {}
        # This loop is O(N^2) per point, so O(N^4) total. 4096^2 = 16M. 
        # 16M ops in Python loop is too slow for precomputation.
        pass
        
    # Better precomputation:
    # dist is 4096x4096.
    # For each row i, we want to build a dict {d: bitset}.
    # We can vectorize this?
    # It might be faster to just use the dist matrix in the check, but optimized.
    
    # Let's try a hybrid: 
    # Precompute for each point i, a list of (distance, bitset) is hard to vectorize.
    # However, we can precompute `dist_sorted` and `indices_sorted`?
    
    # Let's go with the "Vectorized Check" but optimized for the specific logic.
    # Current code does:
    # 1. new_dists = dist[idx, current_indices]
    # 2. Check duplicates in new_dists (Apex P)
    # 3. For each pair in current, check base cases.
    
    # My optimized check:
    # 1. new_dists = dist[idx, current_indices]
    # 2. If len(np.unique(new_dists)) != len(new_dists): return False (Apex P)
    # 3. For Apex Q:
    #    We need to check if for any Q in current, dist(idx, Q) == dist(Q, R) for some R in current.
    #    This is: for any Q in current, is dist(idx, Q) present in the set of distances from Q to other points in current?
    
    # Let's precompute `dist_to_bitset` more efficiently.
    # We can use a dictionary of dictionaries.
    # Precomputation time:
    # For i in range(total):
    #   for j in range(total):
    #     d = dist[i, j]
    #     if d not in dict[i]: dict[i][d] = 0
    #     dict[i][d] |= (1 << j)
    # This is 16M iterations. In Python, this is ~1-2 seconds. Acceptable.
    
    dist_dict = [dict() for _ in range(total)]
    for i in range(total):
        row = dist[i]
        # To speed up, we can iterate over unique distances?
        # Or just loop j.
        for j in range(total):
            d = row[j]
            if d in dist_dict[i]:
                dist_dict[i][d] |= (1 << j)
            else:
                dist_dict[i][d] = (1 << j)
                
    # Optimization: Convert to list of (d, mask) for faster lookup? 
    # Or keep as dict. Dict lookup is O(1).
    
    def can_add(idx, current_indices, current_bitset):
        """
        Check if idx can be added to current_indices (represented by current_bitset).
        """
        K = len(current_indices)
        if K < 2:
            return True
            
        # 1. Apex at idx (P)
        new_dists = dist[idx, current_indices]
        # Check for duplicates
        if len(np.unique(new_dists)) != K:
            return False
            
        # 2. Apex at Q (in current)
        # For each Q in current, check if dist(idx, Q) is a distance from Q to any R in current.
        # We have current_bitset.
        # We have dist_dict[Q].
        # We need to check if (current_bitset & dist_dict[Q].get(dist(idx, Q), 0)) != 0.
        # Note: dist(idx, Q) is new_dists[i] where current_indices[i] == Q.
        
        for i, Q in enumerate(current_indices):
            d_pq = new_dists[i]
            # Get bitset of points at distance d_pq from Q
            mask_q = dist_dict[Q].get(d_pq, 0)
            # Check if any point in current is in this mask
            # Note: Q is not in mask_q because dist(Q,Q)=0 and d_pq > 0 (since idx != Q)
            # So if intersection is non-zero, there is an R in current such that dist(Q, R) == d_pq.
            # This forms isosceles triangle (idx, Q, R) with apex Q.
            if current_bitset & mask_q:
                return False
                
        return True

    def greedy_random():
        current_indices = []
        current_bitset = 0
        remaining = np.arange(total)
        random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline:
                break
            if can_add(idx, current_indices, current_bitset):
                current_indices.append(idx)
                current_bitset |= (1 << idx)
                
        return current_indices, current_bitset

    def local_search(initial_indices, initial_bitset, time_budget):
        current_indices = list(initial_indices)
        current_bitset = initial_bitset
        best_indices = list(current_indices)
        best_bitset = current_bitset
        best_size = len(current_indices)
        
        end_time = time.time() + time_budget
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            # Try random swaps
            for _ in range(1000): # Fixed number of swap attempts per round
                if time.time() > end_time:
                    break
                if not current_indices:
                    break
                
                # Pick a random point to remove
                i_remove = random.randrange(len(current_indices))
                idx_remove = current_indices[i_remove]
                
                # Pick a random candidate to add
                # To make it faster, pick from all points not in current
                # Generating a random point not in current:
                # We can pick a random index and check if it's in current_bitset
                # If it is, pick another.
                # Or, maintain a list of available? 
                # Simpler: pick random idx. If in bitset, skip.
                
                while True:
                    idx_add = random.randrange(total)
                    if not (current_bitset & (1 << idx_add)):
                        break
                        
                # Check if we can remove idx_remove and add idx_add
                # First, remove idx_remove temporarily
                temp_bitset = current_bitset & ~(1 << idx_remove)
                temp_indices = [x for x in current_indices if x != idx_remove]
                
                if can_add(idx_add, temp_indices, temp_bitset):
                    # Accept swap
                    current_bitset = temp_bitset | (1 << idx_add)
                    current_indices[i_remove] = idx_add
                    
                    # Try to add more points greedily
                    # This is a "kick"
                    # We can try to add a few more
                    for _ in range(10):
                         if time.time() > end_time: break
                         # Pick random candidate
                         while True:
                             idx_try = random.randrange(total)
                             if not (current_bitset & (1 << idx_try)):
                                 break
                         if can_add(idx_try, current_indices, current_bitset):
                             current_indices.append(idx_try)
                             current_bitset |= (1 << idx_try)
                             
                    if len(current_indices) > best_size:
                        best_size = len(current_indices)
                        best_indices = list(current_indices)
                        best_bitset = current_bitset
                    improved = True
                    
        return best_indices, best_bitset

    # Phase 1: Greedy restarts
    best_indices = []
    best_bitset = 0
    best_size = 0
    
    # Run greedy until time is low
    while time.time() < deadline - 30:
        res_idx, res_bit = greedy_random()
        if len(res_idx) > best_size:
            best_size = len(res_idx)
            best_indices = res_idx
            best_bitset = res_bit
            
    # Phase 2: Local search on best
    if best_size > 0 and time.time() < deadline - 10:
        remaining_time = max(5, deadline - 10 - time.time())
        final_idx, final_bit = local_search(best_indices, best_bitset, remaining_time)
        if len(final_idx) > best_size:
            best_size = len(final_idx)
            best_indices = final_idx
            best_bitset = final_bit

    # Convert to tuples
    result = [(xs[idx], ys[idx]) for idx in best_indices]
    return result

if __name__ == "__main__":
    # Test
    points = construct()
    print(f"Found {len(points)} points")
