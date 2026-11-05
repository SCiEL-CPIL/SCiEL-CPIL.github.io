import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def is_valid_circulant(n, S):
        """
        Checks if the circulant graph on n vertices with connection set S 
        has no regular induced subgraph of size >= 6.
        S is a set of integers in {1, ..., n//2}.
        """
        # Build adjacency matrix for fast checking
        # adj[i][j] = 1 if (j-i) % n in S or (i-j) % n in S
        # Actually, for circulant, adj[i][j] depends only on (j-i) % n.
        
        # Precompute the "distance mask" for each vertex pair? 
        # No, just use the property: 
        # deg(v, subset) = sum_{u in subset, u != v} [ (u-v)%n in S or (v-u)%n in S ]
        
        # To be fast, we can precompute a boolean matrix M where M[i][j] = 1 if (j-i)%n in S.
        # Then deg(v, subset) = sum(M[subset, v]) excluding v itself.
        
        # Let's construct the full adjacency matrix for this specific S.
        # n is small (<=21), so n x n matrix is tiny.
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for d in S:
                j1 = (i + d) % n
                j2 = (i - d) % n
                if j1 != i:
                    adj[i, j1] = 1
                if j2 != i:
                    adj[i, j2] = 1
                # Note: if n is even and d = n/2, j1 == j2, but adj is symmetric so it's fine.
                # We just need to ensure we don't double count in logic, but for matrix entry it's idempotent.
        
        # Check all subsets of size 6
        # C(20, 6) = 38760, C(21, 6) = 54264.
        # For each subset, compute degrees of its vertices within the subset.
        # If all degrees are equal, it's a violation.
        
        # Optimization: We can vectorize the degree calculation for all subsets of size k.
        # But iterating 54k subsets in Python is slow if we do it naively.
        # Let's use a slightly smarter loop or NumPy tricks.
        
        # For n=20, 38760 subsets.
        # For each subset (tuple of 6 indices), we extract the 6x6 submatrix of adj.
        # Then sum rows.
        
        # Let's try a direct loop first, it might be fast enough for 1024 graphs.
        # 1024 graphs * 38760 subsets * (6x6 extraction + sum) 
        # 1024 * 38760 = 39M operations. In Python, this is too slow (~minutes).
        
        # We need a faster verification.
        
        # Alternative: Precompute the "regularity" of subsets for the *complement*? No.
        
        # Let's use the fact that we are checking *many* S.
        # Can we precompute something?
        # The structure of the violation depends on S.
        
        # Let's optimize the inner loop using NumPy.
        # Generate all combinations of size 6 as a numpy array of shape (N_subsets, 6).
        # Then for each subset, we need to check if the degrees are constant.
        
        # Actually, let's just use a highly optimized Python loop with bitmasks if possible,
        # or accept that we might not check all S if it's too slow.
        
        # Better idea: 
        # We can check size 6 first. If it fails, stop.
        # Size 6 is the most common violation.
        
        # Let's try to vectorize the degree calculation for a single S.
        # adj is n x n.
        # For a subset S_idx (list of 6 vertices), the degree of vertex v in S_idx is:
        # sum(adj[v, u] for u in S_idx if u != v)
        
        # We can precompute all subsets of size 6 as a numpy array `subs` of shape (M, 6).
        # Then, for a given adj, we can compute the degree matrix D of shape (M, 6) where:
        # D[i, j] = sum(adj[subs[i, j], subs[i, :]] ) - 1 (to exclude self-loop if any, but simple graph has no self-loops)
        # Wait, adj[v, v] is 0. So D[i, j] = sum(adj[subs[i, j], subs[i, :]])
        
        # This can be done with matrix multiplication?
        # Let A be the adj matrix.
        # Let M be the indicator matrix of subsets? Too big.
        
        # Let's stick to a fast Python loop but with minimal overhead.
        # Precompute the list of subsets once.
        
        return True # Placeholder, we will implement the check inside the loop below

    # Precompute all subsets of size 6 for n=20 and n=21
    # We will test n=21 first, then n=20.
    
    def check_circulant(n, S):
        # S is a set of distances
        # Build adj
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for d in S:
                j1 = (i + d) % n
                j2 = (i - d) % n
                if j1 != i:
                    adj[i][j1] = 1
                    adj[j1][i] = 1
                if j2 != i:
                    adj[i][j2] = 1
                    adj[j2][i] = 1
        
        # Check size 6
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                deg = 0
                for u in subset:
                    if u != v and adj[v][u]:
                        deg += 1
                degs.append(deg)
            
            if len(set(degs)) == 1:
                return False
        
        # If size 6 is clean, check larger sizes (7 to n)
        # This is rare, but we must do it.
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                degs = []
                for v in subset:
                    deg = 0
                    for u in subset:
                        if u != v and adj[v][u]:
                            deg += 1
                    degs.append(deg)
                
                if len(set(degs)) == 1:
                    return False
        return True

    # We will iterate over all possible connection sets S for n=21, then n=20.
    # For n, the possible distances are 1 to n//2.
    # Number of S is 2^(n//2).
    # For n=21, n//2 = 10, so 1024 sets.
    # For n=20, n//2 = 10, so 1024 sets.
    
    # To speed up the inner check, we can prune:
    # If a subset of size 6 is regular, we return False immediately.
    
    # Let's try n=21 first.
    for n in [21, 20]:
        if time.time() > deadline - 5.0:
            break
            
        max_d = n // 2
        # Generate all subsets of {1, ..., max_d}
        # We can iterate from 0 to 2^max_d - 1 and interpret bits as S.
        
        # Optimization: Randomly shuffle the order of S to find a valid one faster?
        # Or systematic? Systematic is safer to ensure we don't miss one.
        # But 1024 is small.
        
        # We can also bias towards "balanced" S? 
        # Regular graphs often have issues. Non-regular induced subgraphs are more likely if S is "irregular".
        
        for mask in range(1 << max_d):
            if time.time() > deadline - 5.0:
                break
                
            S = set()
            for d in range(1, max_d + 1):
                if mask & (1 << (d - 1)):
                    S.add(d)
            
            # Optional: Skip if S is empty or full? 
            # Empty S: no edges. Induced subgraphs are empty (regular with deg 0). 
            # Wait, empty graph on 6 vertices is regular (deg 0). So S cannot be empty.
            # Full S: complete graph. Induced subgraphs are complete (regular). So S cannot be full.
            if len(S) == 0 or len(S) == max_d:
                continue
                
            if check_circulant(n, S):
                # Found a valid graph!
                # Construct edges
                edges = []
                for i in range(n):
                    for d in S:
                        j1 = (i + d) % n
                        if i < j1:
                            edges.append([i, j1])
                        j2 = (i - d) % n
                        if i < j2:
                            edges.append([i, j2])
                
                # Deduplicate edges just in case (though the logic above should handle it)
                # Actually, if d1 != d2, edges are distinct.
                # If d = n/2 (only for even n), j1 == j2. 
                # For n=20, d=10. j1 = (i+10)%20, j2 = (i-10)%20 = (i+10)%20.
                # So we might add the same edge twice if we are not careful.
                # My loop: 
                # for d in S:
                #   j1 = ...
                #   if i < j1: add
                #   j2 = ...
                #   if i < j2: add
                # If j1 == j2, we add it twice? No, because i < j1 is same condition.
                # We should use a set of edges or a boolean matrix to avoid duplicates.
                
                edge_set = set()
                for i in range(n):
                    for d in S:
                        j1 = (i + d) % n
                        if i < j1:
                            edge_set.add((i, j1))
                        j2 = (i - d) % n
                        if i < j2:
                            edge_set.add((i, j2))
                
                edges = [[u, v] for u, v in sorted(edge_set)]
                
                return {'n': n, 'edges': edges}
                
    # Fallback: If no circulant graph works, try a random graph for n=20?
    # Or return n=19 Paley?
    
    # Let's add a quick check for Paley(19) as a fallback.
    def is_quadratic_residue(a, p):
        if a == 0: return False
        return pow(a, (p-1)//2, p) == 1

    p = 19
    adj = [0]*p
    for i in range(p):
        for j in range(i+1, p):
            if is_quadratic_residue((i-j) % p, p) or is_quadratic_residue((j-i) % p, p):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Verify Paley(19)
    valid = True
    for subset in itertools.combinations(range(p), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        degs = []
        for v in subset:
            degs.append((adj[v] & mask).bit_count())
        if len(set(degs)) == 1:
            valid = False
            break
            
    if valid:
        for k in range(7, p + 1):
            for subset in itertools.combinations(range(p), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        edges = []
        for i in range(p):
            m = adj[i]
            for j in range(i+1, p):
                if m & (1 << j):
                    edges.append([i, j])
        return {'n': 19, 'edges': edges}
        
    # If all else fails, return n=5
    return {'n': 5, 'edges': []}
