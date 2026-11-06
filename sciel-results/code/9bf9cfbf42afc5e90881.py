import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s

    # Helper to generate masks for subsets of size k
    def generate_masks(n, k):
        masks = []
        # Use Gosper's hack
        mask = (1 << k) - 1
        limit = 1 << n
        while mask < limit:
            masks.append(mask)
            c = mask & -mask
            r = mask + c
            mask = (((r ^ mask) >> 2) // c) | r
        return masks

    # Helper to check if a specific mask is regular
    def is_mask_regular(adj, mask, n):
        # Get degrees of all vertices in mask
        first_v = -1
        first_deg = -1
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            d = (adj[v] & mask).bit_count()
            if first_v == -1:
                first_v = v
                first_deg = d
            else:
                if d != first_deg:
                    return False
            m ^= lsb
        return True

    # Helper to check full validity
    def is_valid_full(n, adj):
        # Check sizes 6 to n
        # To save time, check size 6 first (most likely to fail)
        for k in range(6, n + 1):
            # Generate masks for this k on the fly to save memory if n is large?
            # For n=21, k=6 is 54k masks. k=11 is 350k masks.
            # We'll generate on the fly for each k.
            mask = (1 << k) - 1
            limit = 1 << n
            while mask < limit:
                if is_mask_regular(adj, mask, n):
                    return False
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Search strategy
    # 1. Try n=21 with Simulated Annealing / Local Search
    # 2. Try n=20 with Random Search (should be fast)
    # 3. Fallback n=19
    
    # Precompute size 6 masks for n=21 and n=20 for fast checking
    # We need them for the cost function
    masks_6_21 = generate_masks(21, 6)
    masks_6_20 = generate_masks(20, 6)
    
    def count_bad_size_6(adj, n, all_masks_6, sample_size=None):
        """
        Count how many size-6 subsets are regular.
        If sample_size is provided, only check a random sample.
        """
        if sample_size and sample_size < len(all_masks_6):
            sample = random.sample(all_masks_6, sample_size)
        else:
            sample = all_masks_6
            
        count = 0
        for mask in sample:
            if is_mask_regular(adj, mask, n):
                count += 1
        return count

    def search_local(n, time_limit, masks_6):
        """
        Local search to find a graph with no regular size-6 subsets.
        If found, verify full validity.
        """
        best_adj = None
        best_cost = float('inf')
        
        # Initial random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        current_cost = count_bad_size_6(adj, n, masks_6, sample_size=500)
        
        if current_cost == 0:
            if is_valid_full(n, adj):
                return adj
        
        temperature = 1.0
        cooling_rate = 0.99999
        min_temp = 0.01
        
        while time.time() < time_limit and temperature > min_temp:
            # Propose move: flip a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge
            mask_uv = (1 << u) | (1 << v)
            # Check if edge exists
            has_edge = bool(adj[u] & (1 << v))
            
            if has_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_cost = count_bad_size_6(adj, n, masks_6, sample_size=500)
            delta = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_cost = new_cost
                if current_cost == 0:
                    # Check full size 6
                    full_cost = count_bad_size_6(adj, n, masks_6)
                    if full_cost == 0:
                        # Check full validity
                        if is_valid_full(n, adj):
                            return adj
            
            # Revert if not accepted
            if delta >= 0 and random.random() >= math.exp(-delta / temperature):
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            temperature *= cooling_rate
            
            # Periodic restart if stuck
            if current_cost > best_cost:
                if random.random() < 0.1: # 10% chance to restart if not improving
                    adj = [0] * n
                    for i in range(n):
                        for j in range(i + 1, n):
                            if random.random() < 0.5:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                    current_cost = count_bad_size_6(adj, n, masks_6, sample_size=500)
                    temperature = 1.0
            else:
                best_cost = current_cost
                best_adj = adj[:]

        return None

    # Try n=21
    if time.time() < deadline - 20.0:
        result = search_local(21, time.time() + (deadline - time.time() - 20.0), masks_6_21)
        if result:
            return {'n': 21, 'edges': get_edges(result, 21)}

    # Try n=20
    if time.time() < deadline - 5.0:
        # Random search is likely sufficient for n=20
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            adj = [0] * 20
            for u in range(20):
                for v in range(u + 1, 20):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            # Quick check: size 6
            cost = count_bad_size_6(adj, 20, masks_6_20)
            if cost == 0:
                if is_valid_full(20, adj):
                    return {'n': 20, 'edges': get_edges(adj, 20)}

    # Fallback n=19
    adj = [0] * 19
    for u in range(19):
        for v in range(u + 1, 19):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    if is_valid_full(19, adj):
        return {'n': 19, 'edges': get_edges(adj, 19)}
        
    # Final fallback
    return {'n': 5, 'edges': []}
