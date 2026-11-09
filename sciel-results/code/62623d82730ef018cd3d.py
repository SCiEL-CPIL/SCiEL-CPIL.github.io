import itertools

def construct():
    n = 18
    # Define 3 parts
    A = list(range(0, 6))
    B = list(range(6, 12))
    C = list(range(12, 18))
    
    edges = []
    
    # Cyclic shifts for cross edges
    # S = {1, 2, 4} mod 6
    S = [1, 2, 4]
    
    # A to B: a in A, b in B. b-6 - a in S
    for a in A:
        for s in S:
            b_idx = (a + s) % 6
            b = 6 + b_idx
            edges.append([a, b])
            
    # B to C: b in B, c in C. c-12 - (b-6) in S
    for b in B:
        b_idx = b - 6
        for s in S:
            c_idx = (b_idx + s) % 6
            c = 12 + c_idx
            edges.append([b, c])
            
    # C to A: c in C, a in A. a - (c-12) in S
    for c in C:
        c_idx = c - 12
        for s in S:
            a_idx = (c_idx + s) % 6
            a = a_idx
            edges.append([c, a])
            
    # Verify K4-free (should be true for tripartite)
    # Build adjacency bitmasks for fast triangle check
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def is_triangle_free(mask):
        # Check if the subgraph induced by 'mask' is triangle-free
        # Iterate over all pairs in mask, check if they have a common neighbor in mask
        # Or iterate over all triples? 
        # Efficient way: For each edge (u,v) in mask, check if N(u) & N(v) & mask is empty.
        # If we find any such edge with common neighbor, it's not triangle-free.
        # But we need to check if the induced subgraph HAS a triangle.
        # If it HAS a triangle, it is NOT triangle-free.
        # So we return False if a triangle is found.
        
        # Get list of vertices in mask
        verts = []
        for i in range(n):
            if mask & (1 << i):
                verts.append(i)
                
        # Check all triples
        # This is O(k^3). For k=12, 12^3 = 1728. For 48620 subsets, 80M ops. Might be slow in Python.
        # Better: Iterate edges in mask.
        for i in range(len(verts)):
            u = verts[i]
            for j in range(i+1, len(verts)):
                v = verts[j]
                if adj[u] & (1 << v):
                    # Edge exists
                    common = adj[u] & adj[v] & mask
                    if common:
                        return False
        return True

    # Find max triangle-free induced subgraph size
    # Start from high k and go down
    # We can prune: if we find one of size k, return k.
    # To speed up, we can randomize or use specific ordering.
    # For n=18, checking k=12 down to 8 is feasible.
    
    # Optimization: 
    # If k is large, it's likely to contain a triangle.
    # We want the MAX size of a triangle-free set.
    # So we check k=18, 17, ..., 1.
    # The first k for which ANY subset is triangle-free is the answer.
    # Wait, if k=18 is triangle-free, answer is 18.
    # If k=18 is not, check k=17.
    # We stop at the first k where we find a triangle-free subset.
    
    # To make it faster, we can check k from 12 down to 1, assuming the answer is <= 12.
    # If we don't find any for 12..1, then it's < 1? No, empty set is triangle-free.
    # Single vertex is triangle-free. So answer is at least 1.
    
    # Let's check from 12 down to 1.
    # If we find one at 12, return 12.
    # If we find one at 11, return 11.
    # ...
    
    # However, checking ALL subsets of size 12 is 48620.
    # Checking ALL subsets of size 11 is 31824.
    # Total ~80k checks. Each check is O(k^2) or O(k^3).
    # With bitmasks, O(k^2) edge checks is fast.
    
    for k in range(12, 0, -1):
        found = False
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if is_triangle_free(mask):
                return edges # The score will be k. We don't return score, we return edges.
                # The evaluator computes the score.
                # So we just need to return the edges.
                # But we need to ensure we return a graph that has a SMALL score.
                # If the graph has a triangle-free set of size 12, the score is 12.
                # We want to MINIMIZE the score.
                # So we should return a graph where the max triangle-free set is as small as possible.
                # This construction is fixed. We just return it.
                # We don't need to compute the score inside construct() if we trust the construction.
                # But the current code computes it. I will remove the computation to save time, 
                # or keep it for verification? 
                # The evaluator computes the score. 
                # So I just need to return the edges.
                # I will remove the expensive scoring from construct() to ensure it finishes fast.
                pass
        # If no triangle-free set of size k found, continue to k-1
        
    return edges
