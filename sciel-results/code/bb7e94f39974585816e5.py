import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    # Helper to check if graph is (5,5)-free
    # adjacency: numpy array of shape (n, n) with 0/1
    # We need to check for K5 and independent set of size 5.
    # Independent set of size 5 in G is K5 in complement.
    
    def is_valid(adj):
        n = adj.shape[0]
        # Check K5
        # For each edge (i, j), check if there is a K3 in common neighbors
        # Or just iterate 5-cliques? 42 choose 5 is 850,668.
        # With bitsets, we can do:
        # For each i, j > i, k > j, l > k, check if m > l connects to all.
        # This is still slow in Python loops.
        
        # Better: Use bitsets for neighborhoods.
        # N[i] is a bitset of neighbors of i.
        # K5 exists if there are i<j<k<l<m such that m in N[i] & N[j] & N[k] & N[l].
        
        # Let's use a faster heuristic check for small n.
        # For n=42, 850k checks is okay if each check is fast.
        # But we do this in a loop for local search. We need it very fast.
        
        # Optimization:
        # Precompute common neighbors for all pairs? 42*41/2 = 861 pairs.
        # common[i][j] = N[i] & N[j]
        # Then for each triple (i,j,k), common[i][j] & common[i][k] & common[j][k]
        # If non-empty, there is a K4. Then check if any vertex in that set connects to all 4?
        # Actually, if we have a K4 (i,j,k,l), we just need to check if there is an m connected to all 4.
        # Number of K4s can be large.
        
        # Alternative:
        # Count K5s directly using bit operations.
        # For each i < j < k:
        #   candidates = N[i] & N[j] & N[k]
        #   candidates &= (1 << n) - 1
        #   candidates &= ~((1 << (k+1)) - 1) # only vertices > k
        #   # Now check if any m in candidates is connected to l? No, we need 5 vertices.
        #   # This finds K3s extended.
        
        # Let's try a simpler approach for the validity check in the search:
        # We only need to know if count == 0.
        
        # We will implement a fast counter for K5 and Ind5.
        
        n = adj.shape[0]
        N = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            N[i] = mask
            
        # Count K5
        k5_count = 0
        # Iterate i < j < k
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                # Common neighbors of i and j, greater than j
                common_ij = N[i] & N[j] & ~((1 << (j+1)) - 1)
                # Iterate k in common_ij
                k_mask = common_ij
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    k_mask ^= lsb
                    # Common neighbors of i, j, k, greater than k
                    common_ijk = common_ij & N[k] & ~((1 << (k+1)) - 1)
                    # Iterate l in common_ijk
                    l_mask = common_ijk
                    while l_mask:
                        lsb_l = l_mask & -l_mask
                        l = lsb_l.bit_length() - 1
                        l_mask ^= lsb_l
                        # Common neighbors of i, j, k, l, greater than l
                        common_ijkl = common_ijk & N[l] & ~((1 << (l+1)) - 1)
                        if common_ijkl:
                            # There is at least one m
                            k5_count += 1
                            # We can break early if we just want validity?
                            # But for local search, we want the count to guide us.
                            # However, breaking early makes it non-smooth.
                            # For now, let's just count.
                            
        # Count Ind5 (K5 in complement)
        ind5_count = 0
        # Complement neighborhoods
        # Nc[i] = ~N[i] & ((1<<n)-1) & ~(1<<i)
        # We can just run the same logic on ~adj
        
        # This might be too slow for a local search loop.
        # Let's try to find a static graph first.
        
        return k5_count == 0 and ind5_count == 0

    # 1. Check Paley graphs
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        adj = np.zeros((q, q), dtype=np.uint8)
        for i in range(q):
            for j in range(i+1, q):
                if (j - i) % q in residues:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj

    # Check q=41
    if time.time() < deadline:
        adj41 = get_paley(41)
        if is_valid(adj41):
            edges = [(i, j) for i in range(41) for j in range(i+1, 41) if adj41[i][j]]
            return 41, edges

    # Check q=47 (just in case, though 47 > 42, but if it's valid, score 47? No, R(5,5)<=46)
    # So max possible is 42.
    
    # 2. Local Search for n=42
    # We will try to find a valid graph on 42 vertices.
    # Objective: Minimize K5_count + Ind5_count
    
    n_target = 42
    best_score = 37
    best_edges = [(i, j) for i in range(37) for j in range(i+1, 37) if ((j-i)%37 in {(x*x)%37 for x in range(1,37)})]
    
    # We need a fast evaluation function for the search.
    # The is_valid function above is O(n^3) or worse due to nested while loops.
    # For n=42, it might be acceptable if we don't call it too often.
    # But local search needs thousands of evaluations.
    
    # Let's optimize the counting.
    # We can use NumPy matrix multiplication to count triangles/cliques?
    # Number of K5s is not directly A^5.
    
    # Alternative: Use the fact that we only need to find ONE valid graph.
    # We can use a randomized local search.
    # Start with random graph on 42 vertices.
    # Flip edges that reduce the number of K5s and Ind5s.
    
    # To speed up evaluation, we can precompute neighbor bitsets and use them.
    
    def count_k5_ind5(adj):
        n = adj.shape[0]
        N = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            N[i] = mask
            
        k5 = 0
        ind5 = 0
        
        # Count K5
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                common_ij = N[i] & N[j] & ~((1 << (j+1)) - 1)
                k_mask = common_ij
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    k_mask ^= lsb
                    common_ijk = common_ij & N[k] & ~((1 << (k+1)) - 1)
                    l_mask = common_ijk
                    while l_mask:
                        lsb_l = l_mask & -l_mask
                        l = lsb_l.bit_length() - 1
                        l_mask ^= lsb_l
                        common_ijkl = common_ijk & N[l] & ~((1 << (l+1)) - 1)
                        if common_ijkl:
                            k5 += 1
                            
        # Count Ind5 (K5 in complement)
        # Complement of adj
        # We can just invert the logic or build Nc
        Nc = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            Nc[i] = full_mask & ~N[i] & ~(1 << i)
            
        for i in range(n):
            for j in range(i+1, n):
                if not Nc[i] & (1 << j): continue # i and j not connected in complement (i.e. connected in G)
                common_ij_c = Nc[i] & Nc[j] & ~((1 << (j+1)) - 1)
                k_mask = common_ij_c
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    k_mask ^= lsb
                    common_ijk_c = common_ij_c & Nc[k] & ~((1 << (k+1)) - 1)
                    l_mask = common_ijk_c
                    while l_mask:
                        lsb_l = l_mask & -l_mask
                        l = lsb_l.bit_length() - 1
                        l_mask ^= lsb_l
                        common_ijkl_c = common_ijk_c & Nc[l] & ~((1 << (l+1)) - 1)
                        if common_ijkl_c:
                            ind5 += 1
                            
        return k5 + ind5

    # Local Search
    # We will try a few restarts.
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize
        # Start with a random graph, or maybe a biased one?
        # Random is fine.
        adj = np.zeros((n_target, n_target), dtype=np.uint8)
        for i in range(n_target):
            for j in range(i+1, n_target):
                if np.random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        current_cost = count_k5_ind5(adj)
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each edge
            # To speed up, we can randomize the order or pick a random subset
            # For n=42, there are 861 edges.
            # 861 evaluations per step.
            # If each eval takes 1ms, that's ~1s per step.
            # We might get 100 steps.
            
            # Optimization: Only check edges that might improve?
            # Or just random sampling of edges.
            
            for _ in range(100): # Sample 100 random edges
                if time.time() > deadline:
                    break
                i = np.random.randint(n_target)
                j = np.random.randint(n_target)
                if i == j: continue
                if i > j: i, j = j, i
                
                # Flip edge (i, j)
                old_val = adj[i][j]
                adj[i][j] = 1 - old_val
                adj[j][i] = adj[i][j]
                
                new_cost = count_k5_ind5(adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost == 0:
                        # Found valid graph!
                        edges = [(a, b) for a in range(n_target) for b in range(a+1, n_target) if adj[a][b]]
                        return n_target, edges
                else:
                    # Revert
                    adj[i][j] = old_val
                    adj[j][i] = old_val
                    
    # If no 42-vertex graph found, return the 37-vertex Paley graph
    adj37 = get_paley(37)
    edges37 = [(i, j) for i in range(37) for j in range(i+1, 37) if adj37[i][j]]
    return 37, edges37
