"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_score = -1
    best_subset = list(range(min_vertices)) # Fallback
    
    # Iterate over all possible subsets
    # We can iterate from 0 to 2^n - 1
    # To optimize, we can skip subsets with popcount < min_vertices
    # But 2^17 is small enough to just check all.
    
    for mask in range(1 << n):
        # Count bits in mask
        # Python 3.10+ has int.bit_count()
        # For compatibility, we can use bin(mask).count('1') or a lookup table, but bit_count is standard now.
        # Let's assume Python 3.8+ which doesn't have bit_count, so use bin().count('1') or a helper.
        # Actually, 131k iterations is tiny. bin().count('1') is fine.
        pc = bin(mask).count('1')
        if pc < min_vertices:
            continue
            
        # Calculate degrees for vertices in the subset
        degrees = set()
        for i in range(n):
            if mask & (1 << i):
                # Degree of i in induced subgraph is popcount(adj[i] & mask)
                deg = bin(adj[i] & mask).count('1')
                degrees.add(deg)
                
        score = len(degrees)
        if score > best_score:
            best_score = score
            # Extract vertices
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(i)
            best_subset = subset
            
    return best_subset

# EVOLVE-BLOCK-END
