import random
import time

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer

    def is_valid_bitmask(adj, n):
        """
        adj: list of n ints, where adj[i] is bitmask of neighbors of i.
        Checks if any subset of size >= 6 induces a regular subgraph.
        """
        # Precompute popcounts for all masks up to 1<<n? 
        # No, just use .bit_count() which is fast in Python 3.8+
        
        # We need to check all subsets S of {0..n-1} with |S| >= 6
        # Iterate through all masks
        # Optimization: If a subset S is regular, return False.
        
        # To speed up, we can iterate by size or just all masks.
        # There are 2^n masks. For n=20, 1M masks.
        # For each mask, we check O(n) vertices.
        # 1M * 20 = 20M operations.
        # Each operation is: mask & adj[i], then bit_count().
        # This should be fast enough in Python if optimized.
        
        # We can prune: if a smaller subset is regular, it's invalid.
        # We don't need to check supersets if a subset is regular? 
        # No, the condition is "No set of 6 or more". So if ANY set is regular, invalid.
        
        # Iterate masks
        # We can skip masks with < 6 bits.
        
        # A faster way to iterate subsets of size >= 6 is hard in pure Python loops.
        # But 1M iterations is small.
        
        # Let's optimize the inner loop.
        for mask in range(1, 1 << n):
            if mask.bit_count() < 6:
                continue
            
            # Check if induced subgraph on 'mask' is regular
            # Get the first vertex in the mask to determine the target degree
            # Find first set bit
            # Or just iterate through all bits in mask
            
            # We need to check if deg(v) is constant for all v in mask.
            # deg(v) = (adj[v] & mask).bit_count()
            
            # Optimization: 
            # If the mask is a superset of a known regular set, we don't need to check?
            # No, regularity is not monotonic.
            
            # Let's just do the check.
            # To speed up, pick one vertex v0 in mask.
            # target_deg = (adj[v0] & mask).bit_count()
            # Check all other v in mask.
            
            # Find v0
            # Use lowbit trick?
            # v0 = (mask & -mask).bit_length() - 1
            
            lowbit = mask & -mask
            v0 = lowbit.bit_length() - 1
            target_deg = (adj[v0] & mask).bit_count()
            
            # Check other vertices
            m = mask & (mask - 1) # Remove v0
            while m:
                lowbit = m & -m
                v = lowbit.bit_length() - 1
                if (adj[v] & mask).bit_count() != target_deg:
                    break
                m = m & (m - 1)
            
            if m == 0:
                # All vertices had the same degree
                return False
        
        return True

    # Try to find a valid graph for n=20, 19, etc.
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() - start_time > time_limit:
            break
            
        found = False
        best_edges = None
        
        # Use multiple random seeds
        # To increase chances, we can try biased probabilities or just uniform
        for seed in range(50):
            if time.time() - start_time > time_limit:
                break
            
            random.seed(seed)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if is_valid_bitmask(adj, n):
                found = True
                edges = []
                for i in range(n):
                    # Extract neighbors
                    m = adj[i]
                    while m:
                        lowbit = m & -m
                        j = lowbit.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        m = m & (m - 1)
                best_edges = edges
                break
        
        if found:
            return {'n': n, 'edges': best_edges}
            
    # Fallback: n=6, K6 minus an edge
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
