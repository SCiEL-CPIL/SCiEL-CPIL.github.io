import time
from itertools import product

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Non-adjacency mask (for independent sets, we need v to be non-adjacent to all in part)
    # non_adj[v] is bitmask of vertices NOT adjacent to v (excluding v itself)
    non_adj = [0] * n
    for v in range(n):
        # All vertices except v
        all_mask = (1 << n) - 1
        # Remove v and its neighbors
        non_adj[v] = all_mask & ~(adj[v] | (1 << v))

    deadline = time.time() + 55
    
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
    
    def solve_k(k, types):
        """
        Try to partition vertices into k parts with given types.
        types[i] is 0 if part i must be a clique, 1 if independent set.
        Returns list of parts (lists of vertices) or None.
        """
        part_masks = [0] * k
        part_vertices = [[] for _ in range(k)]
        
        def can_add(part_idx, v):
            if types[part_idx] == 0: # Clique
                # All vertices in part must be adjacent to v
                # part_masks[part_idx] & ~adj[v] must be 0
                if part_masks[part_idx] & ~adj[v]:
                    return False
                return True
            else: # Independent Set
                # No vertex in part can be adjacent to v
                # part_masks[part_idx] & adj[v] must be 0
                if part_masks[part_idx] & adj[v]:
                    return False
                return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            
            # Symmetry breaking: if part i is empty, and we've already tried an empty part of the same type, skip
            seen_empty_types = set()
            
            for i in range(k):
                if not part_vertices[i]:
                    if types[i] in seen_empty_types:
                        continue
                    seen_empty_types.add(types[i])
                
                if can_add(i, v):
                    part_masks[i] |= (1 << v)
                    part_vertices[i].append(v)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    part_masks[i] &= ~(1 << v)
                    part_vertices[i].pop()
                    
            return False
        
        if backtrack(0):
            return [list(p) for p in part_vertices if p]
        return None

    # Try k from 1 to n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Generate all type combinations
        # Optimization: we can skip type combinations that are permutations of each other?
        # No, the parts are labeled in the backtracking, but symmetry breaking handles the empty parts.
        # However, if two parts have the same type, they are symmetric.
        # The number of unique type assignments is the number of ways to choose which parts are cliques.
        # This is sum_{j=0}^{k} C(k, j) = 2^k.
        # For k=14, 2^14 = 16384. This is small.
        
        for types in product([0, 1], repeat=k):
            if time.time() > deadline:
                break
            
            res = solve_k(k, types)
            if res:
                # Ensure return type is list of lists of ints
                return res
        
    # Fallback
    return [[v] for v in range(n)]
