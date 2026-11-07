"""Cochromatic partition solver for pinned random graph on 14 vertices."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for u, v in P['edges']:
        edges.add((min(u,v), max(u,v)))
    
    # Adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    def is_edge(u, v):
        if u == v: return False
        return adj[u][v]
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if not is_edge(vertices[i], vertices[j]):
                    return False
        return True
    
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if is_edge(vertices[i], vertices[j]):
                    return False
        return True
    
    def is_valid_part(vertices):
        return is_clique(vertices) or is_independent(vertices)
    
    deadline = time.time() + 55  # safety margin for 60s limit
    
    # Precompute all valid subsets (cliques or independent sets)
    # For n=14, there are 2^14 = 16384 subsets, manageable
    all_valid = {}  # bitmask -> list of vertices
    for mask in range(1, 1 << n):
        verts = [i for i in range(n) if (mask >> i) & 1]
        if is_valid_part(verts):
            all_valid[mask] = verts
    
    # For each mask, precompute the list of vertices
    mask_to_verts = {mask: [i for i in range(n) if (mask >> i) & 1] for mask in range(1, 1 << n)}
    
    # Try k from 1 upward
    # For a given k, we need to partition [0..n-1] into k parts, each a valid part
    # Use backtracking with bitmask
    
    def solve_k(k):
        """Find a partition into exactly k valid parts."""
        full_mask = (1 << n) - 1
        
        # Precompute valid subsets that are subsets of a given mask
        # We'll use memoization or just iterate
        
        # Backtracking: assign vertices one by one to k slots
        # But better: pick a valid subset for part 1, then recurse on remaining with k-1 parts
        
        # To make this efficient, we use the fact that we can order the parts
        # and break symmetry by requiring part i to contain the smallest unassigned vertex
        
        def backtrack(remaining_mask, parts_left, current_parts):
            if time.time() > deadline:
                return None
            
            if parts_left == 0:
                if remaining_mask == 0:
                    return current_parts
                return None
            
            if remaining_mask == 0:
                return None  # Can't have empty parts
            
            # Find the smallest vertex in remaining_mask
            # This vertex must go in the first part we form
            # This breaks symmetry
            lowest_bit = 1
            while (remaining_mask & lowest_bit) == 0:
                lowest_bit <<= 1
            
            # We need to find a valid subset S of remaining_mask that contains lowest_bit
            # such that the remaining can be partitioned into parts_left - 1 valid parts
            
            # Iterate over all valid subsets that are subsets of remaining_mask and contain lowest_bit
            # To be efficient, we can iterate over all valid masks and check
            # But there are too many. Instead, let's enumerate subsets of remaining_mask that contain lowest_bit
            
            # Better approach: enumerate all valid masks that are subsets of remaining_mask
            # and contain lowest_bit, then recurse
            
            # For efficiency, precompute for each mask the list of valid submasks containing a specific bit
            # This is expensive. Instead, let's just iterate over all valid masks.
            
            # Alternative: use the structure. Since n=14 is small, we can try all valid subsets
            # of remaining_mask that contain lowest_bit.
            
            # Let's iterate over all valid masks
            for vmask in all_valid:
                # Check if vmask is a subset of remaining_mask
                if (vmask & remaining_mask) != vmask:
                    continue
                # Check if vmask contains lowest_bit
                if (vmask & lowest_bit) == 0:
                    continue
                # Don't allow vmask to be the entire remaining_mask if parts_left > 1
                # (to avoid empty parts later, though backtrack will handle it)
                
                new_remaining = remaining_mask ^ vmask
                result = backtrack(new_remaining, parts_left - 1, current_parts + [all_valid[vmask]])
                if result is not None:
                    return result
            
            return None
        
        return backtrack(full_mask, k, [])
    
    # Try k from 1 upward
    for k in range(1, n + 1):
        if time.time() > deadline - 5:
            break
        result = solve_k(k)
        if result is not None:
            # Convert to list of lists of ints
            return [list(part) for part in result]
    
    # Fallback: should not reach here for k=n (each vertex alone is valid)
    # But just in case, return singleton partition
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
