"""Branch-and-bound for minimum cochromatic partition using maximal homogeneous sets."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    deadline = time.time() + 55.0
    
    # Precompute valid_part for all subsets
    valid_part = [False] * (1 << n)
    valid_part[0] = True
    for mask in range(1, 1 << n):
        is_c = True
        m = mask
        while m and is_c:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if (adj[v] & m) != m:
                is_c = False
        if is_c:
            valid_part[mask] = True
        else:
            is_i = True
            m = mask
            while m and is_i:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & m) != 0:
                    is_i = False
            if is_i:
                valid_part[mask] = True
    
    # Find all maximal valid parts (cliques or independent sets)
    # A maximal valid part is a valid part that cannot be extended by adding any other vertex
    # while remaining valid
    maximal_parts = []
    for mask in range(1, 1 << n):
        if not valid_part[mask]:
            continue
        # Check if this is maximal: no superset is valid
        # Try adding each vertex not in mask
        is_maximal = True
        for v in range(n):
            if (mask >> v) & 1:
                continue
            new_mask = mask | (1 << v)
            if valid_part[new_mask]:
                is_maximal = False
                break
        if is_maximal:
            maximal_parts.append(mask)
    
    # Sort maximal parts by size descending for better pruning
    maximal_parts.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Branch and bound: find minimum number of maximal parts that cover all vertices
    # with no overlap (partition)
    full_mask = (1 << n) - 1
    best = [n + 1]
    best_partition = [None]
    
    def bnb(remaining, parts_used, current_parts):
        if time.time() > deadline:
            return
        if remaining == 0:
            if parts_used < best[0]:
                best[0] = parts_used
                best_partition[0] = list(current_parts)
            return
        if parts_used >= best[0]:
            return
        
        # Lower bound: minimum number of parts needed
        # Each part can cover at most max_part_size vertices
        # But we need a tighter bound
        # Simple bound: ceil(popcount(remaining) / max_possible_part_size)
        # For now, use 1 as minimum additional
        
        # Try each maximal part that is a subset of remaining
        for part in maximal_parts:
            if part & remaining != part:
                continue
            # This part can be used
            new_remaining = remaining & ~part
            current_parts.append(part)
            bnb(new_remaining, parts_used + 1, current_parts)
            current_parts.pop()
    
    bnb(full_mask, 0, [])
    
    # If we found a solution, convert to vertex lists
    if best_partition[0] is not None:
        result_parts = []
        for part_mask in best_partition[0]:
            vertices = [v for v in range(n) if (part_mask >> v) & 1]
            result_parts.append(vertices)
        return result_parts
    
    # Fallback: return individual vertices as parts
    return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
