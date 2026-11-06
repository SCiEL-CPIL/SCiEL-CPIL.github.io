import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Precompute masks by popcount for n up to 22
    # We'll build these on demand per n
    
    def build_masks_by_count(n):
        """Return list of lists: masks_by_count[k] = list of masks with popcount k"""
        masks_by_count = [[] for _ in range(n + 1)]
        for mask in range(1 << n):
            k = mask.bit_count()
            masks_by_count[k].append(mask)
        return masks_by_count
    
    def is_regular_mask(mask, adj):
        """Check if induced subgraph on vertices in mask is regular.
        Returns True if regular (all degrees equal), False otherwise."""
        k = mask.bit_count()
        if k < 6:
            return True  # not a violation
        
        # Get degrees of vertices in mask
        # For efficiency, check if all degrees are the same
        # We need all vertices in mask to have the same number of neighbors in mask
        
        m = mask
        first_deg = -1
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            deg = (adj[i] & mask).bit_count()
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False  # not regular
            m ^= lsb
        return True  # all degrees equal, so regular
    
    def check_graph_valid(n, adj, masks_by_count):
        """Check if graph is valid: no subset of size >= 6 induces a regular subgraph."""
        for k in range(6, n + 1):
            for mask in masks_by_count[k]:
                if is_regular_mask(mask, adj):
                    return False
        return True
    
    def try_find_graph(n, time_budget):
        """Try to find a valid graph on n vertices within time_budget seconds."""
        end_time = time.time() + time_budget
        
        # Precompute masks
        masks_by_count = build_masks_by_count(n)
        
        # Try random graphs with various edge probabilities
        # Start with p=0.5, then try other values
        p_values = [0.5, 0.4, 0.6, 0.3, 0.7, 0.45, 0.55, 0.35, 0.65]
        
        random.seed(42)  # deterministic for reproducibility
        
        attempts = 0
        while time.time() < end_time:
            p = p_values[attempts % len(p_values)]
            
            # Generate random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if check_graph_valid(n, adj, masks_by_count):
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            attempts += 1
        
        return None
    
    # Strategy: 
    # 1. Try n=21 with a moderate time budget (say 40s)
    # 2. If that fails, try n=20 with remaining time
    # 3. If n=20 fails (unlikely), fall back to smaller n
    
    # First, try n=21
    result = try_find_graph(21, 40.0)
    if result is not None:
        return result
    
    # Then try n=20
    remaining = deadline - time.time()
    if remaining > 5:
        result = try_find_graph(20, remaining - 5.0)
        if result is not None:
            return result
    
    # Fallback: try n=19, 18, etc.
    for n in range(19, 5, -1):
        remaining = deadline - time.time()
        if remaining < 3:
            break
        result = try_find_graph(n, remaining - 3.0)
        if result is not None:
            return result
    
    # Last resort: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
