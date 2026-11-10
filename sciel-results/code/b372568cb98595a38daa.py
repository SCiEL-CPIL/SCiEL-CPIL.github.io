"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    p = 101
    # Precompute modular inverses for denominators 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Generate all non-empty subsets of {1..10} and their inverse sums
    # subset represented as bitmask, size = popcount
    all_subsets = []  # (mask, size, residue)
    subsets_by_size = [[] for _ in range(11)]
    
    for mask in range(1, 1 << 10):
        terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
        residue = sum(inv[x] for x in terms) % p
        size = len(terms)
        subsets_by_size[size].append((mask, residue))
        all_subsets.append((mask, size, residue))
    
    # Precompute indices by size
    subset_indices_by_size = [[] for _ in range(11)]
    for i, (mask, size, residue) in enumerate(all_subsets):
        subset_indices_by_size[size].append(i)

    def kuhn_matching(n_left, n_right, adj):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if pair_v[v] == -1 or dfs(pair_v[v], visited):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            return False
            
        matching = 0
        for u in range(n_left):
            visited = [False] * n_right
            if dfs(u, visited):
                matching += 1
        return matching, pair_u

    # Try k from 1 to 10
    for k in range(1, 11):
        right_indices = []
        for s in range(1, k + 1):
            right_indices.extend(subset_indices_by_size[s])
        
        n_right = len(right_indices)
        if n_right < 101:
            continue
        
        # Build adjacency: for each residue r (0..100), list of local right indices
        adj = [[] for _ in range(101)]
        for local_idx, global_idx in enumerate(right_indices):
            mask, size, residue = all_subsets[global_idx]
            adj[residue].append(local_idx)
        
        # Check if every residue has at least one subset
        valid = True
        for r in range(101):
            if not adj[r]:
                valid = False
                break
        if not valid:
            continue
        
        # Run Kuhn's algorithm
        matching_size, pair_u = kuhn_matching(101, n_right, adj)
        
        if matching_size == 101:
            # Extract result
            result = []
            for r in range(101):
                local_idx = pair_u[r]
                if local_idx == -1:
                    # Fallback (should not happen)
                    terms = [1]
                    result.append(terms)
                else:
                    global_idx = right_indices[local_idx]
                    mask = all_subsets[global_idx][0]
                    terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
                    result.append(terms)
            return result
    
    # Fallback if no matching found (should not happen)
    representations = {}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
        residue = sum(inv[x] for x in terms) % p
        if residue not in representations:
            representations[residue] = terms
    return [representations[r] for r in range(101)]
# EVOLVE-BLOCK-END
