"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for u, v in P['edges']:
        edges.add((u, v))
        edges.add((v, u))
    
    # Precompute adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute which subsets are cliques or independent sets
    # Use bitmask representation: subset S is a clique if for all u in S, S \ {u} subset of adj[u]
    # S is independent if for all u in S, S \ {u} disjoint from adj[u]
    
    all_valid = [False] * (1 << n)
    all_valid[0] = True  # empty set is both clique and independent
    
    for mask in range(1, 1 << n):
        # Check if mask is a clique
        is_clique = True
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            rest = mask ^ lb
            if rest & ~adj[u]:
                is_clique = False
                break
            m ^= lb
        if is_clique:
            all_valid[mask] = True
            continue
        
        # Check if mask is an independent set
        is_independent = True
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            rest = mask ^ lb
            if rest & adj[u]:
                is_independent = False
                break
            m ^= lb
        if is_independent:
            all_valid[mask] = True
    
    full_mask = (1 << n) - 1
    
    # Precompute list of valid masks for quick lookup
    valid_masks = [mask for mask in range(1, 1 << n) if all_valid[mask]]
    
    # BFS from 0 to full_mask, tracking parent pointers for reconstruction
    # dist[mask] = minimum number of parts to cover 'mask'
    dist = [-1] * (1 << n)
    parent = [None] * (1 << n)  # parent[mask] = (prev_mask, added_mask)
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        state = queue.popleft()
        if state == full_mask:
            break
        # Try adding each valid mask that is disjoint from state
        for vm in valid_masks:
            if vm & state:
                continue
            new_state = state | vm
            if dist[new_state] == -1:
                dist[new_state] = dist[state] + 1
                parent[new_state] = (state, vm)
                queue.append(new_state)
    
    # Reconstruct the partition
    parts = []
    state = full_mask
    while state != 0:
        prev_state, added = parent[state]
        vertices = []
        m = added
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            vertices.append(v)
            m ^= lb
        parts.append(vertices)
        state = prev_state
    
    # Ensure types are Python int
    result = []
    for part in parts:
        result.append([int(v) for v in part])
    
    return result
# EVOLVE-BLOCK-END
