import time
import sys

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Build adjacency list and bitsets
    # adj[i] is a bitset (int) of neighbors of i
    adj = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
    # Find isolated vertices
    isolated = []
    non_isolated = []
    for i in range(1, N + 1):
        if adj[i] == 0:
            isolated.append(i)
        else:
            non_isolated.append(i)
            
    # The solution will include all isolated vertices
    # We need to find MIS in the subgraph induced by non_isolated vertices
    
    # Map non_isolated to 0..M-1 for bitset operations
    M = len(non_isolated)
    # Create mapping from original index to new index
    idx_map = {v: i for i, v in enumerate(non_isolated)}
    
    # Build adjacency bitsets for the subgraph
    # adj_sub[i] is bitset of neighbors in subgraph
    adj_sub = [0] * M
    for i in range(M):
        v_i = non_isolated[i]
        # Get neighbors in original graph
        neighbors_orig = adj[v_i]
        # Filter to non_isolated
        # We can iterate over bits or just check each non_isolated vertex
        # M is small enough to just loop
        for j in range(M):
            v_j = non_isolated[j]
            if i != j and (neighbors_orig >> v_j) & 1:
                adj_sub[i] |= (1 << j)
                
    # Branch and Bound for MIS
    # We want to maximize the number of vertices selected.
    # State: (candidates, current_set, size)
    # candidates: bitset of vertices that can still be added
    # current_set: list of vertices in current set (or just size is enough for optimization, but we need to reconstruct)
    
    best_size = 0
    best_set = []
    
    # Initial candidates: all vertices in subgraph
    all_mask = (1 << M) - 1
    
    # Order vertices by degree descending for better pruning
    # We will use a recursive function
    # To speed up, we can use a greedy heuristic to find a good initial solution
    
    # Greedy heuristic
    def greedy_mis(adj_list, mask):
        selected = []
        remaining = mask
        while remaining:
            # Pick vertex with max degree in remaining? Or min degree?
            # For MIS, picking a vertex with high degree removes many neighbors, which is good for pruning but might reduce size.
            # Standard greedy for MIS: pick vertex with min degree? No, that's for coloring.
            # For MIS, a common heuristic is to pick a vertex, add it, remove it and its neighbors.
            # Let's try picking vertex with max degree in the remaining graph to prune fast?
            # Actually, for maximum independent set, we want to keep as many as possible.
            # A good heuristic is to sort vertices by degree and pick the one with min degree?
            # Let's just pick the first available vertex for simplicity in the heuristic, or max degree.
            
            # Let's use a simple heuristic: pick vertex with max degree in the induced subgraph defined by 'remaining'
            # This might not be optimal for size, but good for pruning.
            # Actually, for MIS, a vertex with LOW degree is "safer" to pick because it conflicts with fewer others.
            # So we should pick the vertex with MIN degree in the remaining graph.
            
            min_deg = M
            best_v = -1
            temp_mask = remaining
            while temp_mask:
                # Extract lowest set bit
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                # Calculate degree of v in 'remaining'
                deg = (adj_sub[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                break
                
            selected.append(best_v)
            # Remove best_v and its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~adj_sub[best_v]
            
        return selected

    init_set = greedy_mis(adj_sub, all_mask)
    best_size = len(init_set)
    best_set = init_set[:]
    
    # Refine best_set to original indices
    def to_original(indices):
        return [non_isolated[i] for i in indices]
    
    current_best_original = to_original(best_set)
    
    # Branch and Bound
    # We use a stack or recursion. Recursion depth M (<=256) is fine.
    # But we need to manage time.
    
    def solve(candidates, current_indices, current_size):
        nonlocal best_size, best_set, current_best_original
        
        if time.time() - start_time > limit:
            return False # Time out
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_indices[:]
                current_best_original = to_original(best_set)
            return True
            
        # Upper bound: current_size + popcount(candidates)
        # If current_size + popcount(candidates) <= best_size, prune
        if current_size + candidates.bit_count() <= best_size:
            return True
            
        # Pick a vertex to branch on
        # Heuristic: pick vertex with max degree in candidates?
        # Or min degree?
        # Let's pick vertex v in candidates.
        # Branch 1: Include v. New candidates = candidates & ~adj_sub[v] & ~(1<<v).
        # Branch 2: Exclude v. New candidates = candidates & ~(1<<v).
        
        # To improve pruning, pick v with high degree in candidates?
        # If v has high degree, including it removes many candidates.
        # Excluding it leaves many.
        # Usually, branching on high degree vertices is better for MIS.
        
        temp_mask = candidates
        lsb = temp_mask & -temp_mask
        v = lsb.bit_length() - 1
        
        # Option 1: Include v
        new_candidates_1 = candidates & ~(1 << v) & ~adj_sub[v]
        new_indices_1 = current_indices + [v]
        
        # Option 2: Exclude v
        new_candidates_2 = candidates & ~(1 << v)
        new_indices_2 = current_indices
        
        # Order branches: Try the one with larger potential first?
        # Include v usually reduces candidates more.
        # Let's try Include first.
        
        if not solve(new_candidates_1, new_indices_1, current_size + 1):
            return False
        
        if not solve(new_candidates_2, new_indices_2, current_size):
            return False
            
        return True

    solve(all_mask, [], 0)
    
    # Combine with isolated
    final_set = current_best_original + isolated
    return final_set
