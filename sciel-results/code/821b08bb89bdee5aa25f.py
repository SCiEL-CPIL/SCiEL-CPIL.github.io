import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i to avoid duplicates
            mask &= ~((1 << (i + 1)) - 1)
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def check_k4(adj, n):
        """Returns a tuple (a, b, c, d) if a K4 is found, else None."""
        for a in range(n):
            mask_a = adj[a]
            # Iterate neighbors b > a
            mask_b = mask_a & ~((1 << (a + 1)) - 1)
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                # Common neighbors of a and b, with index > b
                common = mask_a & adj[b]
                common &= ~((1 << (b + 1)) - 1) # c > b
                common &= ~(1 << a)
                common &= ~(1 << b)
                
                # Now find an edge (c, d) in common
                # We need c < d.
                # Iterate c in common
                mask_c = common
                while mask_c:
                    lsb_c = mask_c & -mask_c
                    c = lsb_c.bit_length() - 1
                    mask_c ^= lsb_c
                    
                    # Check if any neighbor of c in common is > c
                    # Neighbors of c in common: adj[c] & common
                    # We need d > c.
                    candidates = adj[c] & common
                    candidates &= ~((1 << (c + 1)) - 1)
                    if candidates:
                        # Found K4: a, b, c, d
                        # d is the lowest set bit in candidates
                        lsb_d = candidates & -candidates
                        d = lsb_d.bit_length() - 1
                        return (a, b, c, d)
        return None

    def has_ind6_exact(adj, n):
        """
        Returns True if an independent set of size 6 exists in G.
        This is equivalent to finding a clique of size 6 in the complement of G.
        Uses backtracking with bitmasks.
        """
        comp_adj = [0] * n
        for i in range(n):
            full_mask = (1 << n) - 1
            full_mask &= ~(1 << i)
            comp_adj[i] = full_mask ^ adj[i]

        # Order vertices by degree in complement (descending) to help pruning
        # But for bitmask, we can just use a heuristic ordering or standard order.
        # Standard order is fine for n=36 if we prune well.
        
        found = [False]
        
        def backtrack(cand_mask, current_size):
            if found[0]:
                return
            if current_size == 6:
                found[0] = True
                return
            if not cand_mask:
                return
            if bin(cand_mask).count('1') + current_size < 6:
                return
            
            # Pick a vertex
            # Heuristic: Pick vertex with highest degree in cand_mask
            # For simplicity and speed, pick the first set bit
            # But better heuristic: pick one with max neighbors in cand_mask
            # Let's try picking the first one for now, it's fast.
            lsb = cand_mask & -cand_mask
            v = lsb.bit_length() - 1
            cand_mask ^= lsb
            
            # Include v
            # Neighbors of v in cand_mask
            neighbors = comp_adj[v] & cand_mask
            
            # If we include v, we can only pick from neighbors
            if bin(neighbors).count('1') + current_size + 1 >= 6:
                backtrack(neighbors, current_size + 1)
            
            if found[0]:
                return
                
            # Exclude v
            # We can pick from the rest of cand_mask
            # Note: cand_mask already has v removed
            if bin(cand_mask).count('1') + current_size >= 6:
                backtrack(cand_mask, current_size)

        backtrack((1 << n) - 1, 0)
        return found[0]

    def local_search(n, adj, max_steps=2000):
        """
        Performs local search on adj (list of ints) for n vertices.
        Returns (success, adj).
        """
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Check K4
            k4 = check_k4(adj, n)
            if k4:
                a, b, c, d = k4
                # Flip a random edge in this K4
                verts = [a, b, c, d]
                random.shuffle(verts)
                u, v = verts[0], verts[1]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            # Check Ind6 (Heuristic: Random Sampling)
            ind6_found = False
            # Try a few random samples
            for _ in range(30):
                # Pick 6 random vertices
                sample = random.sample(range(n), 6)
                # Check if they form an independent set
                is_ind = True
                for i in range(6):
                    for j in range(i + 1, 6):
                        if (adj[sample[i]] >> sample[j]) & 1:
                            is_ind = False
                            break
                    if not is_ind:
                        break
                if is_ind:
                    ind6_found = True
                    # Flip a random edge in this Ind6
                    verts = sample
                    random.shuffle(verts)
                    u, v = verts[0], verts[1]
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    break
            
            if ind6_found:
                continue
            
            # If no violations found (heuristically), we consider it valid for now.
            # But we need a rigorous check before accepting.
            # Let's do a rigorous check here.
            # Rigorous K4 check is fast.
            # Rigorous Ind6 check is the bottleneck, but we do it only when heuristic fails.
            
            if not has_ind6_exact(adj, n):
                return True, adj
                
        return False, adj

    # Strategy:
    # 1. Find a valid graph for n=35 using local search.
    # 2. Try to extend to n=36.
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid graph for n=35
    target_n = 35
    found = False
    for attempt in range(100):
        if time.time() > deadline - 10.0:
            break
            
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        success, adj = local_search(target_n, adj, max_steps=1000)
        
        if success:
            # Rigorous verification
            # K4 check
            if check_k4(adj, target_n) is None:
                # Ind6 check (Rigorous)
                if not has_ind6_exact(adj, target_n):
                    best_n = target_n
                    best_edges = edges_from_adj(adj, target_n)
                    found = True
                    break

    if not found:
        # Fallback: Try to find a smaller graph, e.g., n=30
        target_n = 30
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            success, adj = local_search(target_n, adj, max_steps=500)
            if success:
                if check_k4(adj, target_n) is None:
                    if not has_ind6_exact(adj, target_n):
                        best_n = target_n
                        best_edges = edges_from_adj(adj, target_n)
                        break

    # Step 2: Try to extend to 36
    if best_n == 35 and time.time() < deadline - 20.0:
        # Take the 35-vertex adj and add vertex 35
        adj35 = build_adj(35, best_edges)
        adj36 = [0] * 36
        for i in range(35):
            adj36[i] = adj35[i]
            
        # Connect vertex 35 to random subset
        for i in range(35):
            if random.random() < 0.5:
                adj36[i] |= (1 << 35)
                adj36[35] |= (1 << i)
                
        # Run local search on 36
        success, adj36 = local_search(36, adj36, max_steps=2000)
        
        if success:
            if check_k4(adj36, 36) is None:
                if not has_ind6_exact(adj36, 36):
                    best_n = 36
                    best_edges = edges_from_adj(adj36, 36)

    if best_n == 0:
        # Last resort: return a small valid graph to avoid error
        # e.g. n=1, no edges
        best_n = 1
        best_edges = []

    return {'n': best_n, 'edges': best_edges}
