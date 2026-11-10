"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict, deque
import math
import sys

# Increase recursion limit for DFS in Hopcroft-Karp
sys.setrecursionlimit(10000)

def construct():
    p = 101
    # Precompute modular inverses for denominators 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Generate all non-empty subsets of {1..10}
    # Store: (mask, size, residue, terms)
    all_subsets = []
    
    for mask in range(1, 1 << 10):
        terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
        residue = sum(inv[x] for x in terms) % p
        size = len(terms)
        all_subsets.append((mask, size, residue, terms))
        
    # Sort subsets by size to easily filter by k
    # We need to find min k such that perfect matching exists using subsets with size <= k
    # Sort indices by size
    subset_indices = list(range(len(all_subsets)))
    subset_indices.sort(key=lambda i: all_subsets[i][1])
    
    # Hopcroft-Karp Algorithm
    def hopcroft_karp(n_left, n_right, adj):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u

    # Try k from 1 to 10
    best_k = 10
    best_pair_u = None
    
    # We can binary search or linear search. Linear is fine for k=1..10.
    # But we can optimize by checking if enough subsets exist.
    
    # Precompute counts of subsets per size
    count_by_size = [0] * 11
    for i in range(len(all_subsets)):
        size = all_subsets[i][1]
        count_by_size[size] += 1
        
    cumulative_count = 0
    for k in range(1, 11):
        cumulative_count += count_by_size[k]
        if cumulative_count < 101:
            continue
            
        # Build adjacency list for this k
        # Right side nodes are the subsets with size <= k
        # We need to map global subset index to local right node index
        # And build adj for each residue (0..100)
        
        # To be efficient, we can just iterate through all_subsets and pick those with size <= k
        # But we sorted subset_indices by size.
        # Let's find the cutoff in subset_indices
        # Actually, just filtering all_subsets is O(1023) which is tiny.
        
        right_subsets = [] # list of (local_idx, global_idx)
        local_to_global = {}
        adj = [[] for _ in range(101)]
        
        local_idx = 0
        for g_idx in subset_indices:
            mask, size, residue, terms = all_subsets[g_idx]
            if size <= k:
                local_to_global[local_idx] = g_idx
                adj[residue].append(local_idx)
                local_idx += 1
            else:
                # Since sorted by size, we can break if size > k
                # But subset_indices is sorted by size, so once size > k, all subsequent are > k
                break
        
        n_right = local_idx
        
        # Check if every residue has at least one neighbor
        valid = True
        for r in range(101):
            if not adj[r]:
                valid = False
                break
        if not valid:
            continue
            
        # Run Hopcroft-Karp
        matching_size, pair_u = hopcroft_karp(101, n_right, adj)
        
        if matching_size == 101:
            best_k = k
            best_pair_u = pair_u
            # We need to keep the local_to_global mapping for this k
            # But we will re-run or store the result directly
            # To avoid storing large objects, let's extract the result here
            result = []
            for r in range(101):
                local_idx_r = pair_u[r]
                if local_idx_r == -1:
                    # Should not happen
                    result.append([1]) # Fallback
                else:
                    g_idx = local_to_global[local_idx_r]
                    terms = all_subsets[g_idx][3]
                    result.append(terms)
            return result
            
    # Fallback if no perfect matching found (should not happen for k=10)
    # Use greedy
    representations = {}
    for g_idx in range(len(all_subsets)):
        mask, size, residue, terms = all_subsets[g_idx]
        if residue not in representations:
            representations[residue] = terms
            
    # Ensure all residues are present
    for r in range(101):
        if r not in representations:
             # Find any subset with this residue
             for g_idx in range(len(all_subsets)):
                 if all_subsets[g_idx][2] == r:
                     representations[r] = all_subsets[g_idx][3]
                     break
             if r not in representations:
                 representations[r] = [1] # Absolute fallback
            
    return [representations[r] for r in range(101)]
# EVOLVE-BLOCK-END
