"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Strategy:
    # 1. Try to find a valid graph on 21 vertices using local search with fast approximate validation.
    # 2. If successful, verify fully and return.
    # 3. If not, try 20 vertices.
    # 4. Fallback to a hardcoded valid 20-vertex graph if search fails.
    
    # Helper: Convert adjacency matrix to list of 64-bit integer masks
    def mat_to_masks(adj):
        n = adj.shape[0]
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            masks[i] = mask
        return masks

    # Helper: Check if a specific subset (given by mask) is regular
    # Returns True if regular, False otherwise
    def is_subset_regular(masks, subset_mask, n):
        # Get indices in subset
        indices = []
        temp = subset_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            indices.append(idx)
            temp ^= lsb
        
        if not indices:
            return False
            
        # Calculate degrees within subset
        degrees = []
        for idx in indices:
            # Count bits in masks[idx] that are also in subset_mask
            # Note: masks[idx] might have bits for vertices outside subset, so we AND
            inter = masks[idx] & subset_mask
            deg = bin(inter).count('1')
            degrees.append(deg)
        
        return len(set(degrees)) == 1

    # Helper: Fast validation - check only subsets of size k
    def has_regular_subset_size_k(masks, n, k):
        # Generate all masks of popcount k
        # For n=21, k=6, this is ~54k subsets.
        # We iterate through all 2^n masks? No, that's 2M.
        # We can iterate through combinations of indices.
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for idx in subset:
                mask |= (1 << idx)
            if is_subset_regular(masks, mask, n):
                return True, mask
        return False, 0

    # Helper: Full validation - check sizes 6 to n
    def is_valid_full(masks, n):
        for k in range(6, n + 1):
            found, _ = has_regular_subset_size_k(masks, n, k)
            if found:
                return False
        return True

    # Helper: Convert masks back to edges
    def masks_to_edges(masks, n):
        edges = []
        for i in range(n):
            mask = masks[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    # Local Search for n vertices
    def local_search(n, rng, time_limit):
        # Initialize random graph
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        masks = mat_to_masks(adj)
        
        # Initial check
        has_reg, _ = has_regular_subset_size_k(masks, n, 6)
        if not has_reg:
            # Verify full
            if is_valid_full(masks, n):
                return masks_to_edges(masks, n)
        
        # Local search: Flip edges to reduce number of regular size-6 subsets
        # We track the count of regular size-6 subsets
        # To make it fast, we just re-evaluate the count after each flip?
        # Counting all 54k subsets per flip is ~0.1s? Maybe too slow.
        # Let's try a simpler heuristic: Random restarts with fast check.
        
        attempts = 0
        while time.time() < time_limit and attempts < 50:
            attempts += 1
            # Random graph
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            masks = mat_to_masks(adj)
            
            # Fast check for size 6
            has_reg, _ = has_regular_subset_size_k(masks, n, 6)
            if not has_reg:
                # Full check
                if is_valid_full(masks, n):
                    return masks_to_edges(masks, n)
            
            # If fast check fails, try to fix by flipping edges
            # This is complex, so we just do more random restarts
            pass
            
        return None

    # Try 21
    rng = np.random.default_rng(42)
    edges_21 = local_search(21, rng, time.time() + 20)
    if edges_21 is not None:
        return {'n': 21, 'edges': edges_21}
        
    # Try 20
    edges_20 = local_search(20, rng, time.time() + 20)
    if edges_20 is not None:
        return {'n': 20, 'edges': edges_20}
        
    # Fallback: Hardcoded 20-vertex graph
    # We generate one with a fixed seed and assume it's valid (high probability)
    # To be safer, we can check it once here
    adj_fb = np.zeros((20, 20), dtype=np.int8)
    rng_fb = np.random.default_rng(12345)
    for i in range(20):
        for j in range(i + 1, 20):
            if rng_fb.random() < 0.5:
                adj_fb[i, j] = 1
                adj_fb[j, i] = 1
    
    masks_fb = mat_to_masks(adj_fb)
    # Verify fallback
    if is_valid_full(masks_fb, 20):
        return {'n': 20, 'edges': masks_to_edges(masks_fb, 20)}
    
    # If even fallback fails (very unlikely), try another seed
    for seed in [1, 2, 3]:
        adj_fb = np.zeros((20, 20), dtype=np.int8)
        rng_fb = np.random.default_rng(seed)
        for i in range(20):
            for j in range(i + 1, 20):
                if rng_fb.random() < 0.5:
                    adj_fb[i, j] = 1
                    adj_fb[j, i] = 1
        masks_fb = mat_to_masks(adj_fb)
        if is_valid_full(masks_fb, 20):
            return {'n': 20, 'edges': masks_to_edges(masks_fb, 20)}
            
    # Last resort: small valid graph
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
