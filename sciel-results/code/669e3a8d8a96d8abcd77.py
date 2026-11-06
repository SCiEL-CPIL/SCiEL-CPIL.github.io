import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Helper to check if a graph is (5,5)-free
    # Graph represented as list of sets or bitmasks
    # n up to 42, so 64-bit integers work for adjacency masks
    
    def count_k5_and_indep5(adj_masks, n):
        """
        adj_masks: list of integers, adj_masks[i] has bit j set if edge (i,j) exists.
        Returns (num_k5, num_indep5)
        """
        num_k5 = 0
        num_indep5 = 0
        
        # Iterate over all combinations of 5 vertices
        # For n=41, C(41,5) is ~ 749,398. This is very fast.
        # For n=42, C(42,5) is ~ 850,668.
        
        for i in range(n-4):
            mask_i = adj_masks[i]
            for j in range(i+1, n-3):
                # Check if i-j edge exists
                if (mask_i >> j) & 1:
                    # i-j is edge
                    # We need 3 more vertices k, l, m such that:
                    # k, l, m form a triangle in the induced subgraph on N(i) \cap N(j)
                    # AND they are connected to i and j
                    # Actually, simpler: just check the 5-set directly?
                    # Checking 5-set directly is O(1) per set if we precompute?
                    # No, checking a specific 5-set {i,j,k,l,m} for K5:
                    # Need edges (i,j), (i,k), (i,l), (i,m), (j,k), (j,l), (j,m), (k,l), (k,m), (l,m)
                    
                    # Let's just iterate all 5-sets for simplicity first, n is small.
                    pass
        
        # More efficient: iterate 5-sets
        # C(41, 5) ~ 750k. 750k * 10 edge checks is ~ 7.5M ops. Very fast in Python?
        # 7.5M ops in pure Python might take ~0.5s - 1s.
        # We can do this repeatedly in a search loop.
        
        # Let's implement the 5-set check directly.
        
        # Optimization: Precompute nothing, just check.
        
        for i in range(n-4):
            for j in range(i+1, n-3):
                for k in range(j+1, n-2):
                    for l in range(k+1, n-1):
                        for m in range(l+1, n):
                            # Check K5
                            # Edges:
                            # i-j, i-k, i-l, i-m
                            # j-k, j-l, j-m
                            # k-l, k-m
                            # l-m
                            
                            # Using bitmasks
                            # Edge(i, x) is (adj_masks[i] >> x) & 1
                            
                            # To speed up, we can prune early.
                            # But for n=41, full enumeration is fine.
                            
                            # Let's write a helper for edge check
                            # e(a, b) = (adj_masks[a] >> b) & 1 if a < b else (adj_masks[b] >> a) & 1
                            # Since i<j<k<l<m, we always have a < b in pairs.
                            
                            # K5 check
                            if ((adj_masks[i] >> j) & 1) and \
                               ((adj_masks[i] >> k) & 1) and \
                               ((adj_masks[i] >> l) & 1) and \
                               ((adj_masks[i] >> m) & 1) and \
                               ((adj_masks[j] >> k) & 1) and \
                               ((adj_masks[j] >> l) & 1) and \
                               ((adj_masks[j] >> m) & 1) and \
                               ((adj_masks[k] >> l) & 1) and \
                               ((adj_masks[k] >> m) & 1) and \
                               ((adj_masks[l] >> m) & 1):
                                num_k5 += 1
                                
                            # Indep5 check
                            # No edges among i,j,k,l,m
                            if not ((adj_masks[i] >> j) & 1) and \
                               not ((adj_masks[i] >> k) & 1) and \
                               not ((adj_masks[i] >> l) & 1) and \
                               not ((adj_masks[i] >> m) & 1) and \
                               not ((adj_masks[j] >> k) & 1) and \
                               not ((adj_masks[j] >> l) & 1) and \
                               not ((adj_masks[j] >> m) & 1) and \
                               not ((adj_masks[k] >> l) & 1) and \
                               not ((adj_masks[k] >> m) & 1) and \
                               not ((adj_masks[l] >> m) & 1):
                                num_indep5 += 1
                                
        return num_k5, num_indep5

    def is_valid(adj_masks, n):
        k5, i5 = count_k5_and_indep5(adj_masks, n)
        return k5 == 0 and i5 == 0

    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    # 1. Try Paley Graph on 41 vertices
    n = 41
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                
    if is_valid(adj_masks, n):
        return n, get_edges(adj_masks, n)
        
    # 2. Local Search to fix or find better graph
    # Start from the invalid Paley 41 graph, or random
    # We will use a greedy local search / simulated annealing
    
    # To make it faster, we can use a heuristic that only checks affected 5-sets when an edge flips.
    # But for simplicity and given the short time limit, let's try a simple random restart local search.
    
    best_n = 37
    best_edges = None
    
    # Try to find a valid graph on n=41
    # If we fail, try n=40? No, 41 > 40.
    
    # Let's implement a faster check for a single move.
    # When we flip edge (u, v), only 5-sets containing both u and v are affected.
    # Number of such 5-sets is C(n-2, 3).
    # For n=41, C(39, 3) = 9139.
    # This is much faster than checking all C(41, 5) ~ 750k.
    
    def count_k5_and_indep5_affected(adj_masks, n, u, v):
        """
        Counts K5 and Indep5 that include the pair (u, v).
        Note: This does NOT count all K5/Indep5 in the graph, only those involving edge (u,v) 
        or non-edge (u,v). 
        To maintain a global count, we would need to update it incrementally.
        """
        # We need to know if (u,v) is an edge or not to determine what we are breaking/creating.
        # But this function just counts the current state of 5-sets containing u and v.
        
        num_k5 = 0
        num_indep5 = 0
        
        # Iterate over all triples (a, b, c) from the remaining n-2 vertices
        # such that {u, v, a, b, c} is a 5-set.
        # To avoid double counting or missing, we just pick 3 distinct vertices from V \ {u, v}.
        
        remaining = [x for x in range(n) if x != u and x != v]
        
        # C(39, 3) iterations
        for i in range(len(remaining)-2):
            a = remaining[i]
            for j in range(i+1, len(remaining)-1):
                b = remaining[j]
                for k in range(j+1, len(remaining)):
                    c = remaining[k]
                    
                    # Vertices are u, v, a, b, c
                    # We need to check if this 5-set is a K5 or Indep5.
                    
                    # Helper to check edge between two nodes
                    # e(x, y)
                    
                    # Edges involving u:
                    # e(u,v), e(u,a), e(u,b), e(u,c)
                    # Edges involving v (not u):
                    # e(v,a), e(v,b), e(v,c)
                    # Edges among a,b,c:
                    # e(a,b), e(a,c), e(b,c)
                    
                    # Get bits
                    # Assume u < v for bitmask access? No, adj_masks is symmetric.
                    # e(x, y) = (adj_masks[x] >> y) & 1 if x < y else (adj_masks[y] >> x) & 1
                    # But we can just use a helper function.
                    
                    def e(x, y):
                        if x < y:
                            return (adj_masks[x] >> y) & 1
                        else:
                            return (adj_masks[y] >> x) & 1

                    # Check K5
                    if e(u,v) and e(u,a) and e(u,b) and e(u,c) and \
                       e(v,a) and e(v,b) and e(v,c) and \
                       e(a,b) and e(a,c) and e(b,c):
                        num_k5 += 1
                        
                    # Check Indep5
                    if not e(u,v) and not e(u,a) and not e(u,b) and not e(u,c) and \
                       not e(v,a) and not e(v,b) and not e(v,c) and \
                       not e(a,b) and not e(a,c) and not e(b,c):
                        num_indep5 += 1
                        
        return num_k5, num_indep5

    # Global count maintenance
    # Let's initialize global counts for the current graph
    def init_counts(adj_masks, n):
        k5, i5 = count_k5_and_indep5(adj_masks, n)
        return k5, i5

    current_k5, current_i5 = init_counts(adj_masks, n)
    
    # Simulated Annealing / Greedy
    # We want to reduce current_k5 + current_i5 to 0.
    
    # To speed up, we will try random moves and accept if they reduce the cost.
    # If stuck, we can allow small increases (annealing).
    
    # However, calculating the *change* in count is tricky because a 5-set can be affected by multiple edge flips.
    # But we can just re-calculate the affected 5-sets for the new state and subtract the old affected 5-sets?
    # No, because the "affected" sets are defined by the pair (u,v).
    # If we flip (u,v), the status of all 5-sets containing u and v changes.
    # The status of 5-sets NOT containing u and v remains the same.
    # So:
    # New_K5 = Old_K5 - (K5s containing u,v in old state) + (K5s containing u,v in new state)
    # This works!
    
    # Let's implement the search
    
    # We'll try to fix n=41 first.
    # If we can't, we might try n=40 (which is likely valid if 41 is close) or just return 37.
    # But 41 is the target.
    
    # To be safe, let's also try n=40 with random init if 41 fails quickly.
    # But 41 Paley is a very good start.
    
    # Let's run a local search on n=41.
    
    # Random seed for reproducibility? No, random is fine.
    
    best_cost = current_k5 + current_i5
    best_adj = adj_masks[:]
    
    # If it's already valid, we returned above. So best_cost > 0.
    
    # Search loop
    # Time limit check
    while time.time() < deadline:
        # Pick a random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Calculate old counts for affected 5-sets
        old_k5_aff, old_i5_aff = count_k5_and_indep5_affected(adj_masks, n, u, v)
        
        # Flip edge
        # If u < v, bit v in mask u
        mask_u = adj_masks[u]
        mask_v = adj_masks[v]
        
        # Flip
        adj_masks[u] = mask_u ^ (1 << v)
        adj_masks[v] = mask_v ^ (1 << u)
        
        # Calculate new counts for affected 5-sets
        new_k5_aff, new_i5_aff = count_k5_and_indep5_affected(adj_masks, n, u, v)
        
        # Update global counts
        current_k5 = current_k5 - old_k5_aff + new_k5_aff
        current_i5 = current_i5 - old_i5_aff + new_i5_aff
        
        new_cost = current_k5 + current_i5
        
        # Acceptance criterion
        # Greedy: accept if new_cost < best_cost
        # Annealing: accept if new_cost < best_cost OR (random() < exp(-delta/T))
        
        # For simplicity and speed, let's use a greedy approach with random restarts if stuck.
        # Or simple simulated annealing.
        
        # Let's try greedy first. If we improve, update best.
        if new_cost < best_cost:
            best_cost = new_cost
            best_adj = adj_masks[:]
            
            if best_cost == 0:
                # Found valid graph!
                return n, get_edges(adj_masks, n)
                
        else:
            # Revert move if not accepted?
            # In greedy, we revert. In SA, we might keep.
            # Let's revert to keep the search stable.
            adj_masks[u] = mask_u
            adj_masks[v] = mask_v
            current_k5 = current_k5 - new_k5_aff + old_k5_aff
            current_i5 = current_i5 - new_i5_aff + old_i5_aff
            
    # If we didn't find a valid graph on 41, try 40?
    # Or just return the best found so far (which might be invalid, so we must return a valid one).
    # The task requires a VALID graph.
    # If we don't find one, we should fall back to the Paley 37.
    
    # Check if best_adj is valid
    if is_valid(best_adj, n):
        return n, get_edges(best_adj, n)
        
    # Fallback: Try n=40 with random init + local search?
    # Or just return Paley 37.
    # Given the time, let's try a quick search on n=40 if we have time left.
    
    if time.time() < deadline - 50:
         n2 = 40
         # Random graph on 40 vertices
         adj2 = [0] * n2
         for i in range(n2):
             for j in range(i+1, n2):
                 if random.random() < 0.5:
                     adj2[i] |= (1 << j)
                     adj2[j] |= (1 << i)
                     
         # Local search on n2
         k5_2, i5_2 = init_counts(adj2, n2)
         best_cost_2 = k5_2 + i5_2
         best_adj_2 = adj2[:]
         
         while time.time() < deadline - 10:
             u = random.randint(0, n2-1)
             v = random.randint(0, n2-1)
             if u == v: continue
             if u > v: u, v = v, u
             
             old_k5_aff, old_i5_aff = count_k5_and_indep5_affected(adj2, n2, u, v)
             
             mask_u = adj2[u]
             mask_v = adj2[v]
             adj2[u] = mask_u ^ (1 << v)
             adj2[v] = mask_v ^ (1 << u)
             
             new_k5_aff, new_i5_aff = count_k5_and_indep5_affected(adj2, n2, u, v)
             
             k5_2 = k5_2 - old_k5_aff + new_k5_aff
             i5_2 = i5_2 - old_i5_aff + new_i5_aff
             
             new_cost_2 = k5_2 + i5_2
             
             if new_cost_2 < best_cost_2:
                 best_cost_2 = new_cost_2
                 best_adj_2 = adj2[:]
                 if best_cost_2 == 0:
                     return n2, get_edges(adj2, n2)
             else:
                 adj2[u] = mask_u
                 adj2[v] = mask_v
                 k5_2 = k5_2 - new_k5_aff + old_k5_aff
                 i5_2 = i5_2 - new_i5_aff + old_i5_aff
                 
         if is_valid(best_adj_2, n2):
             return n2, get_edges(best_adj_2, n2)
             
    # Final fallback: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj37 = [0] * 37
    for i in range(37):
        for j in range(i+1, 37):
            if (j - i) % q in residues:
                adj37[i] |= (1 << j)
                adj37[j] |= (1 << i)
                
    return 37, get_edges(adj37, 37)
