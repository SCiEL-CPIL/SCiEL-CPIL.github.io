import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask for edges
    edge_mask = 0
    for u, v in edges:
        edge_mask |= (1 << (u * n + v))
        edge_mask |= (1 << (v * n + u))
        
    def has_edge(u, v):
        return (edge_mask >> (u * n + v)) & 1
        
    # Precompute is_clique and is_independent for all subsets
    # is_clique[mask] = 1 if mask is a clique, 0 otherwise
    # is_indep[mask] = 1 if mask is an independent set, 0 otherwise
    size = 1 << n
    is_clique = [0] * size
    is_indep = [0] * size
    
    is_clique[0] = 1
    is_indep[0] = 1
    
    # Precompute for single vertices
    for v in range(n):
        m = 1 << v
        is_clique[m] = 1
        is_indep[m] = 1
        
    # DP for subsets
    # A subset S is a clique if S\{v} is a clique and v is connected to all in S\{v}
    # A subset S is an independent set if S\{v} is an independent set and v is NOT connected to any in S\{v}
    
    # We can compute this by iterating masks and picking the lowest set bit
    # Let v be the lowest set bit in mask. Let rest = mask ^ (1<<v).
    # is_clique[mask] = is_clique[rest] AND (all u in rest are connected to v)
    # is_indep[mask] = is_indep[rest] AND (no u in rest is connected to v)
    
    # Precompute neighbor masks for each vertex
    neigh_mask = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            if has_edge(u, v):
                neigh_mask[u] |= (1 << v)
                neigh_mask[v] |= (1 << u)
                
    for mask in range(1, size):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest]:
            # Check if v is connected to all in rest
            if (neigh_mask[v] & rest) == rest:
                is_clique[mask] = 1
                
        if is_indep[rest]:
            # Check if v is NOT connected to any in rest
            if (neigh_mask[v] & rest) == 0:
                is_indep[mask] = 1
                
    # Function to check if a mask is a valid part (clique or independent)
    def is_valid_part(mask):
        return is_clique[mask] or is_indep[mask]
        
    # Calculate lower bound: max(chi(G), chi(complement(G)))
    # chi(G) can be found by DP:
    # chi[mask] = min number of colors for mask
    # chi[mask] = 1 + min_{S subset mask, S is independent} chi[mask \ S]
    # This is O(3^n) which is too slow for n=14? 3^14 = 4.78M. This is actually fine in Python if optimized, or we can use the simpler bound.
    # Actually, a simpler lower bound is just checking if k=3 is possible, etc.
    # Let's just run the backtracking search starting from k=3, 4, 5...
    # The search itself will fail quickly if k is too small.
    
    # Backtracking search for k parts
    # We want to partition vertices 0..n-1 into k parts.
    # parts: list of k masks, initially all 0.
    # We assign vertices one by one.
    
    # To speed up, we can fix the order of vertices.
    # Vertices with high degree are more constrained.
    # Let's order vertices by degree descending.
    degrees = [bin(neigh_mask[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map vertex to its position in order? No, just iterate through order.
    # But we need to track which vertices are assigned.
    
    # Better approach:
    # State: current index in 'order', list of current part masks.
    # Pruning:
    # 1. If any part mask becomes invalid (not clique and not indep), prune.
    # 2. Symmetry breaking: If we have empty parts, only assign to the first empty part.
    
    # Let's implement a simple recursive search with memoization or just backtracking.
    # Given n=14, and k likely small, backtracking should be fast.
    
    def solve_k(k):
        # parts is a list of k integers (masks)
        parts = [0] * k
        # We assign vertices in 'order'
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            v_bit = 1 << v
            
            # Try to add v to existing parts
            for i in range(k):
                new_mask = parts[i] | v_bit
                if is_valid_part(new_mask):
                    parts[i] = new_mask
                    if backtrack(idx + 1):
                        return True
                    parts[i] = parts[i] ^ v_bit # restore
            
            # Try to start a new part?
            # We only have k parts total.
            # Symmetry breaking: If we haven't used all k parts yet, we can start a new one.
            # But we must ensure we don't permute empty parts.
            # Standard trick: If part i is 0, and we put v in it, don't try part j>i if part j is also 0.
            # Actually, since we iterate i from 0 to k-1, and we only reach "start new" if no existing part works,
            # we should only start a new part in the first available empty slot.
            
            # Find first empty slot
            first_empty = -1
            for i in range(k):
                if parts[i] == 0:
                    first_empty = i
                    break
                    
            if first_empty != -1:
                parts[first_empty] = v_bit
                if backtrack(idx + 1):
                    return True
                parts[first_empty] = 0
                
            return False
            
        return backtrack(0)
        
    # Lower bound: 
    # If graph has a clique of size w, chi(G) >= w.
    # If graph has an independent set of size alpha, chi(complement) >= alpha.
    # Let's just start from k=3. Z(G) >= 3 for most non-trivial graphs.
    # Actually, Z(G) >= max(omega(G), alpha(G)).
    
    # Calculate omega (max clique size) and alpha (max indep set size)
    omega = 0
    alpha = 0
    for mask in range(1, size):
        if is_clique[mask]:
            omega = max(omega, bin(mask).count('1'))
        if is_indep[mask]:
            alpha = max(alpha, bin(mask).count('1'))
            
    min_k = max(omega, alpha)
    
    # Try k from min_k upwards
    for k in range(min_k, n + 1):
        if solve_k(k):
            # Reconstruct partition
            result = []
            for mask in parts:
                if mask != 0:
                    vertices = []
                    for v in range(n):
                        if (mask >> v) & 1:
                            vertices.append(v)
                    result.append(vertices)
            # Ensure types are Python int and list
            # vertices are already ints
            # result is list of lists
            return result
            
    # Fallback: should not happen
    return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
