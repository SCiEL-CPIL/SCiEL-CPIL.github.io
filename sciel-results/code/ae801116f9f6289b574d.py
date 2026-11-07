"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute which subsets are cliques or independent sets using bitmask
    # Represent each subset as a bitmask (n bits)
    # For n=14, there are 2^14 = 16384 possible subsets - manageable
    
    # Precompute clique and independent set status for all subsets
    # is_clique_mask[mask] = True if the subset represented by mask is a clique
    # is_indep_mask[mask] = True if the subset represented by mask is an independent set
    
    is_clique_mask = [False] * (1 << n)
    is_indep_mask = [False] * (1 << n)
    
    # Empty set is both a clique and independent set
    is_clique_mask[0] = True
    is_indep_mask[0] = True
    
    for mask in range(1, 1 << n):
        # Get the vertices in this subset
        verts = [i for i in range(n) if mask & (1 << i)]
        
        # Check if it's a clique
        clique = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] not in adj[verts[i]]:
                    clique = False
                    break
            if not clique:
                break
        is_clique_mask[mask] = clique
        
        # Check if it's an independent set
        indep = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] in adj[verts[i]]:
                    indep = False
                    break
            if not indep:
                break
        is_indep_mask[mask] = indep
    
    # A subset is "valid" if it's either a clique or an independent set
    is_valid_mask = [is_clique_mask[m] or is_indep_mask[m] for m in range(1 << n)]
    
    # Full mask of all vertices
    full_mask = (1 << n) - 1
    
    # Use iterative deepening to find minimum number of parts
    # Start with lower bound: at least 1 part, and at most n parts
    # We can get a better lower bound: chromatic number of the graph (for cliques)
    # or independence number related bounds
    
    # Get a greedy upper bound first
    def greedy_partition():
        """Greedy construction to get an upper bound."""
        remaining = full_mask
        parts = []
        while remaining:
            # Find the largest valid subset of remaining
            best_sub = 0
            best_size = 0
            # Try all subsets of remaining
            sub = remaining
            while sub:
                if is_valid_mask[sub]:
                    size = bin(sub).count('1')
                    if size > best_size:
                        best_size = size
                        best_sub = sub
                sub = (sub - 1) & remaining
            if best_sub == 0:
                # Single vertex
                lsb = remaining & (-remaining)
                best_sub = lsb
            parts.append(best_sub)
            remaining ^= best_sub
        return parts
    
    greedy_result = greedy_partition()
    best_count = len(greedy_result)
    best_partition = greedy_result
    
    # Iterative deepening: try to find a partition with k parts
    # k ranges from 1 to best_count - 1
    
    def can_partition_with_k_parts(k):
        """Check if we can partition all vertices into k valid parts."""
        if time.time() > deadline:
            return None
        
        # Use backtracking with bitmask
        # parts_masks: list of k masks, initially all 0
        # We assign vertices to parts
        
        # Order vertices by degree for better pruning
        vertices = list(range(n))
        vertices.sort(key=lambda v: len(adj[v]), reverse=True)
        
        # State: tuple of k masks representing current parts
        # We assign one vertex at a time
        
        def backtrack(idx, part_masks):
            if time.time() > deadline:
                return None
            
            if idx == len(vertices):
                # Check all parts are non-empty and valid
                for pm in part_masks:
                    if pm == 0 or not is_valid_mask[pm]:
                        return None
                return list(part_masks)
            
            v = vertices[idx]
            v_bit = 1 << v
            
            # Try adding v to each existing part
            for i in range(k):
                new_mask = part_masks[i] | v_bit
                if is_valid_mask[new_mask]:
                    part_masks[i] = new_mask
                    result = backtrack(idx + 1, part_masks)
                    if result is not None:
                        return result
                    part_masks[i] = new_mask ^ v_bit
            
            # Try putting v in a new part (if we haven't used all k parts yet)
            for i in range(k):
                if part_masks[i] == 0:
                    part_masks[i] = v_bit
                    result = backtrack(idx + 1, part_masks)
                    if result is not None:
                        return result
                    part_masks[i] = 0
            
            return None
        
        result = backtrack(0, [0] * k)
        return result
    
    # Try smaller k values
    for k in range(1, best_count):
        if time.time() > deadline:
            break
        result = can_partition_with_k_parts(k)
        if result is not None:
            best_count = k
            best_partition = result
            break  # Found minimum, no need to continue
    
    # Convert bitmask partition to list of vertex lists
    result_parts = []
    for mask in best_partition:
        if mask:
            verts = [i for i in range(n) if mask & (1 << i)]
            if verts:
                result_parts.append([int(v) for v in verts])
    
    # Ensure we have a valid result
    if not result_parts:
        # Fallback: each vertex as its own part
        result_parts = [[i] for i in range(n)]
    
    return result_parts
# EVOLVE-BLOCK-END
