import math

def construct():
    # Generate candidates
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency matrix
    # adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
                
    # Initial best clique
    # We know S_3 is a clique. Let's find its size and indices.
    # S_3 = {v in candidates | v % 3 == 0}
    s3_indices = [i for i, v in enumerate(candidates) if v % 3 == 0]
    best_clique = s3_indices[:]
    best_size = len(best_clique)
    
    # Order vertices by degree descending
    degrees = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    # Map new index to old index
    # We will reorder the graph for the search
    # Let's just use the order in the recursive call
    
    def expand(clique, candidates_set):
        nonlocal best_clique, best_size
        
        if not candidates_set:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Prune
        if len(clique) + len(candidates_set) <= best_size:
            return
            
        # Heuristic: pick a vertex from candidates_set
        # To improve pruning, we can use a simple coloring or just pick the one with highest degree in candidates_set
        # For now, let's iterate through candidates in order of degree within candidates_set
        
        # Calculate degrees within candidates_set
        # This is O(N^2) per node, which might be slow.
        # Better: precomputed adjacency lists.
        
        # Let's try a simple recursive branch
        # We iterate through candidates, adding one, and removing incompatible ones.
        
        # To ensure we find the max, we must explore all branches.
        # Optimization: If we pick a vertex v, the next candidates must be neighbors of v.
        
        # Let's use the standard "color sort" heuristic if possible, 
        # but for simplicity and robustness within 60s for N=127, 
        # a well-ordered branch and bound should suffice.
        
        # Re-order candidates_set based on degree in the induced subgraph
        # This is expensive to compute repeatedly.
        # Alternative: Fixed ordering by global degree.
        
        # Let's try the fixed ordering first.
        # candidates_set is a set of indices.
        # We iterate through them in the order defined by 'order'.
        
        # Create a list of candidates in the preferred order
        # This is still O(N) per step.
        
        # A faster approach:
        # Maintain a list of available candidates.
        
        pass # Placeholder

    # Given the complexity, I will implement a specific optimized B&B.
    
    # Actually, let's look at the structure again.
    # Max clique is likely 85.
    # If the B&B is too slow, we can fall back to the greedy result or the known S_3.
    # But the task is to maximize.
    
    # Let's implement a simple B&B with a timeout safeguard if needed, 
    # but 127 nodes is small.
    
    # I will implement the Tomita algorithm's core idea:
    # 1. Order vertices.
    # 2. Recursive expand.
    # 3. Use a greedy coloring to bound the clique size.
    
    # Since implementing full coloring is verbose, I'll use a simpler bound:
    # The number of remaining candidates is an upper bound.
    # This is weak.
    
    # Let's try a different heuristic:
    # Since S_3 is a clique of 85, we can try to extend it.
    # But we proved we can't extend it easily.
    
    # What if the max clique is NOT S_3?
    # Maybe a mix of S_5 and S_17?
    # S_5 size 51. S_17 size 15.
    # Intersection S_5, S_17 is size 3.
    # Union is 51+15-3 = 63. Less than 85.
    
    # So 85 is a strong candidate.
    
    # I will run a B&B to be sure.
    
    # To keep it fast, I'll use the adjacency list and a stack-based search.
    
    # Let's refine the expand function.
    
    def max_clique_bnb():
        # State: clique (list), candidates (set of indices)
        # We use the global 'order' to iterate candidates.
        
        # To make it efficient, we represent candidates as a bitmask? 
        # 127 bits fits in a Python int.
        # This allows fast intersection.
        
        # Let's use bitmasks.
        # adj_mask[i] is a bitmask of neighbors of i.
        
        adj_mask = [0]*n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            adj_mask[i] = mask
            
        # Full mask
        full_mask = (1 << n) - 1
        
        best_clique = []
        best_size = 0
        
        # Order vertices by degree descending
        # We will process vertices in this order.
        # In the B&B, we pick a vertex from candidates.
        
        # Standard algorithm:
        def expand(clique, candidates_mask):
            nonlocal best_clique, best_size
            
            if candidates_mask == 0:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return
                
            # Upper bound: len(clique) + popcount(candidates_mask)
            if len(clique) + candidates_mask.bit_count() <= best_size:
                return
                
            # Iterate through candidates
            # We should iterate in a specific order.
            # Let's iterate through bits in candidates_mask.
            
            # To improve pruning, we can estimate the max clique in candidates_mask.
            # A simple bound is the size of the largest independent set? No, clique.
            
            # Let's just iterate.
            # We pick a vertex v from candidates_mask.
            # We add v to clique.
            # New candidates = candidates_mask & adj_mask[v]
            
            # To ensure we explore all, we must try all v.
            # But we can prune branches.
            
            # Optimization: If we pick v, and the new candidates_mask is small, 
            # we can compute the exact max clique in that subgraph recursively.
            
            # Let's iterate through the bits of candidates_mask.
            # We want to try promising vertices first.
            # Let's get the list of vertices in candidates_mask sorted by degree.
            
            verts = [i for i in range(n) if (candidates_mask >> i) & 1]
            # Sort by degree in the induced subgraph? 
            # Approximation: global degree.
            verts.sort(key=lambda i: degrees[i], reverse=True)
            
            for v in verts:
                # Add v
                new_clique = clique + [v]
                new_candidates = candidates_mask & adj_mask[v]
                
                # Prune: if we can't beat best_size
                # We need a better bound than bit_count.
                # But let's try this first.
                
                expand(new_clique, new_candidates)
                
                # Remove v from candidates_mask for the next iteration
                # This is the standard "branch and bound" structure
                # where we fix the order of vertices.
                # Actually, the standard recursive structure is:
                # for v in candidates:
                #    expand(clique + [v], candidates & N(v))
                #    candidates = candidates - {v}
                
                # My loop above doesn't do the "candidates = candidates - {v}" correctly 
                # because I'm iterating over a static list 'verts'.
                # I need to update the mask as I go.
                
            # The logic above is flawed for standard B&B.
            # Correct standard B&B:
            
            def expand2(clique, candidates_mask):
                nonlocal best_clique, best_size
                
                if candidates_mask == 0:
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique[:]
                    return
                    
                if len(clique) + candidates_mask.bit_count() <= best_size:
                    return
                    
                # Get vertices in candidates_mask
                # We process them one by one.
                # We must remove the current vertex from the mask for subsequent iterations.
                
                # To avoid copying the mask in every recursive call unnecessarily, 
                # we pass the mask by value (Python ints are immutable).
                
                # We iterate through the vertices.
                # We need to iterate through the bits of candidates_mask.
                # Let's extract the lowest set bit? Or highest?
                # Order matters for performance.
                
                # Let's try iterating through the global order.
                # This is O(N) per node.
                
                # A faster way:
                # Pick a vertex v.
                # Branch 1: Include v. Expand with N(v) & candidates.
                # Branch 2: Exclude v. Expand with candidates \ {v}.
                
                # This is a binary tree of depth N. Too big.
                # The standard algorithm uses the property that if we include v, 
                # we only consider neighbors.
                
                # Let's stick to the "for v in candidates" loop.
                
                # We need to iterate through the set bits.
                # We can use a while loop on the mask.
                
                # However, we want to visit high-degree vertices first.
                # Let's create a list of vertices in candidates_mask sorted by degree.
                # This is O(K log K) where K is size of candidates.
                
                verts = []
                m = candidates_mask
                while m:
                    # Get lowest set bit
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    verts.append(i)
                    m ^= lsb
                    
                # Sort by degree descending
                verts.sort(key=lambda i: degrees[i], reverse=True)
                
                # We must be careful: if we skip a vertex, we must remove it from the mask 
                # for the subsequent recursive calls in the loop.
                
                # Standard pattern:
                # for v in verts:
                #    expand(clique + [v], candidates_mask & adj_mask[v])
                #    candidates_mask &= ~(1 << v)  # Remove v for next iterations
                
                # But `verts` is a static list. 
                # If we remove v from candidates_mask, it doesn't affect `verts`.
                # But the recursive call `expand` will receive the updated mask?
                # No, the recursive call receives `candidates_mask & adj_mask[v]`.
                # The local `candidates_mask` is updated for the NEXT iteration of the loop.
                
                # This works!
                
                for v in verts:
                    # Check if v is still in candidates_mask?
                    # Since we remove them one by one, and verts is sorted, 
                    # if we remove a higher index, it doesn't affect lower indices in the list?
                    # No, the list is just an order.
                    # We must ensure we only process v if it's still in the mask.
                    if not (candidates_mask & (1 << v)):
                        continue
                        
                    # Include v
                    new_clique = clique + [v]
                    new_candidates = candidates_mask & adj_mask[v]
                    
                    expand2(new_clique, new_candidates)
                    
                    # Exclude v
                    candidates_mask &= ~(1 << v)
                    
                    # Prune check at end of loop?
                    # The check at the beginning of expand2 handles it.
                    
            expand2([], full_mask)
            return best_clique

    # Run the search
    # This might be slow if the bound is weak.
    # Let's add a simple greedy coloring bound if time permits?
    # No, let's try this first.
    
    # To prevent timeout, we can add a node limit.
    # But 127 nodes is small.
    
    # One issue: `expand2` creates `verts` list every time.
    # This is O(N) per node.
    # Total nodes could be large.
    
    # Let's try to optimize the bound.
    # Instead of `bit_count`, we can use a greedy coloring of the subgraph.
    # This is complex to implement correctly in a few lines.
    
    # Alternative: Since we know S_3 is a clique of 85, 
    # we can initialize best_size = 85.
    # The prune `len(clique) + bit_count <= 85` will cut many branches.
    
    # Let's initialize best_clique to S_3 indices.
    
    s3_indices = [i for i, v in enumerate(candidates) if v % 3 == 0]
    best_clique = s3_indices[:]
    best_size = len(best_clique)
    
    # Run B&B
    # If it finds something bigger, great.
    # If it times out, we return the best found so far.
    
    # To handle timeout, we can't easily interrupt a recursive function in Python without signals.
    # But 60s is generous for N=127 with good pruning.
    
    # Let's refine the expand2 to use the global best_size.
    
    def expand_final(clique, candidates_mask):
        nonlocal best_clique, best_size
        
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        if len(clique) + candidates_mask.bit_count() <= best_size:
            return
            
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            verts.append(i)
            m ^= lsb
            
        verts.sort(key=lambda i: degrees[i], reverse=True)
        
        for v in verts:
            if not (candidates_mask & (1 << v)):
                continue
                
            new_clique = clique + [v]
            new_candidates = candidates_mask & adj_mask[v]
            
            expand_final(new_clique, new_candidates)
            
            candidates_mask &= ~(1 << v)
            
    expand_final([], full_mask)
    
    # Convert best_clique indices back to values
    result = [candidates[i] for i in best_clique]
    return result

