import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    P = 27720  # LCM(1, 2, ..., 12)
    N = 12     # number of divisors to match
    MAX_L = 144  # maximum L we need to check
    
    # For each residue r in 0..P-1, and for each position j in 0..MAX_L-1,
    # compute which i (1..12) divide (r+1+j).
    # Then f(r) = smallest L such that there's a perfect matching.
    
    # Precompute: for each r, for each i, the bitmask of j in 0..MAX_L-1 where (r+1+j) % i == 0
    # We'll use numpy for speed
    
    # Create array: vals[r, j] = r + 1 + j for r in 0..P-1, j in 0..MAX_L-1
    # Shape: (P, MAX_L)
    r_arr = np.arange(P, dtype=np.int32)
    j_arr = np.arange(MAX_L, dtype=np.int32)
    vals = r_arr[:, None] + 1 + j_arr[None, :]  # (P, MAX_L)
    
    # For each i in 1..12, compute mask: (P, MAX_L) boolean array where vals % i == 0
    # mask[i-1][r, j] = True iff (r+1+j) % i == 0
    
    # Precompute all masks
    masks = []
    for i in range(1, N + 1):
        mask = (vals % i == 0).astype(np.uint8)  # (P, MAX_L)
        masks.append(mask)
    
    # Now for each r, we need to find the smallest L such that there's a perfect matching
    # where left node i can match to right node j if masks[i-1][r, j] == 1
    
    # For matching with 12 left nodes and up to 144 right nodes, we can use:
    # - Try L from 1 to MAX_L
    # - For each L, check if perfect matching exists in the subgraph with j < L
    
    # Speedup: use numpy to extract the relevant submasks and use a fast matching check
    
    # For 12 nodes, we can use a simple DFS-based matching that's very fast
    # Or we can use the fact that for small graphs, we can precompute
    
    # Let's use a different approach: for each r, precompute the full adjacency
    # and then incrementally check L
    
    # Actually, let's vectorize the matching check using numpy where possible
    
    # For each r, extract the 12xL binary matrix and check matching
    # We'll do this in a loop over r, but use numpy for the mask extraction
    
    best_L = 0
    best_r = 0
    
    # Process residues in batches for efficiency
    # For each r, we need to find min L with perfect matching
    
    # Helper: given a 12xL boolean matrix, check if perfect matching exists
    # Use Kuhn's algorithm with numpy-friendly operations
    
    def has_perfect_matching(adj_matrix):
        """adj_matrix: shape (N, L) boolean. Returns True if perfect matching exists."""
        n, L = adj_matrix.shape
        if L < n:
            return False
        
        # Convert to list of sets for fast iteration
        adj = [set(np.where(adj_matrix[i])[0]) for i in range(n)]
        
        # Check if any node has no edges
        for i in range(n):
            if len(adj[i]) == 0:
                return False
        
        # Kuhn's algorithm
        match_j_to_i = [-1] * L
        
        def try_match(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_j_to_i[j] < 0 or try_match(match_j_to_i[j], visited):
                    match_j_to_i[j] = i
                    return True
            return False
        
        count = 0
        for i in range(n):
            visited = [False] * L
            if try_match(i, visited):
                count += 1
            else:
                return False
        return count == n
    
    # Process each residue
    for r in range(P):
        if time.time() > deadline:
            break
        
        # Get the 12xMAX_L matrix for this residue
        adj_full = np.column_stack([masks[i][r] for i in range(N)])  # (12, MAX_L)
        
        # Find smallest L with perfect matching
        found = False
        for L in range(1, MAX_L + 1):
            if time.time() > deadline:
                break
            if L < N:
                continue  # Can't have perfect matching with fewer right nodes
            adj_sub = adj_full[:, :L]
            if has_perfect_matching(adj_sub):
                if L > best_L:
                    best_L = L
                    best_r = r
                found = True
                break
        
        if found and best_L == MAX_L:
            # Can't do better than MAX_L
            break
    
    # Find m <= 1000000 with m ≡ best_r (mod P)
    m = best_r
    if m > 1000000:
        # Find smallest m in [0, 1000000] with m ≡ best_r (mod P)
        # m = best_r + k*P, need m <= 1000000
        # Since best_r < P, m = best_r works if best_r <= 1000000
        # Otherwise no valid m exists, but P = 27720 < 1000000 so best_r < 1000000
        m = best_r % P
    
    return int(m)

# EVOLVE-BLOCK-END
