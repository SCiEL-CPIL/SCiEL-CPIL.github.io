import random
import time

def construct():
        start_time = time.time()
        timeout = 110  # Safety margin for 120s limit
        
        n = 20
        adj = [0] * n
        
        # Precompute triples
        triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
        
        def is_valid(adj):
            for i, j, k in triples:
                common = adj[i] & adj[j] & adj[k]
                # Exclude i, j, k? 
                # If i is a neighbor of i? No, simple graph, no self loops.
                # adj[i] does not have bit i set.
                # So common neighbors cannot be i, j, k themselves?
                # Wait. If i is connected to j, and j is connected to i...
                # common = N(i) & N(j) & N(k).
                # Can i be in N(j) & N(k)? Yes, if i-j and i-k edges exist.
                # But i is not in N(i). So i is not in common.
                # Same for j and k.
                # So we don't need to mask out i,j,k.
                if common.bit_count() >= 3:
                    return False
            return True

        def find_violation(adj):
            for i, j, k in triples:
                common = adj[i] & adj[j] & adj[k]
                if common.bit_count() >= 3:
                    return (i, j, k), common
            return None, None

        def count_edges(adj):
            return sum(a.bit_count() for a in adj) // 2

        best_edges = 0
        best_adj = [0] * n
        
        # Multiple restarts
        # Each restart: Initialize, Cleanse, Improve
        # Budget per restart?
        # Let's say we run until timeout.
        
        while time.time() - start_time < timeout:
            # 1. Initialize
            # Try different densities
            p = random.uniform(0.5, 0.8)
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # 2. Cleanse
            # Remove edges until valid
            # Use a heuristic: remove edge from the triple with most common neighbors
            # To avoid infinite loop or too slow, limit iterations?
            # With n=20, it should be fast.
            max_clean_steps = 500
            steps = 0
            while steps < max_clean_steps:
                viol_triple, viol_common = find_violation(adj)
                if viol_triple is None:
                    break
                
                u, v, w = viol_triple
                # Pick a vertex from common neighbors
                # And a vertex from the triple
                # To maximize removal impact, maybe pick the most connected?
                # For now, random is okay, but let's pick the one with highest degree in the triple?
                # Actually, just pick random to diversify.
                
                # Get list of common neighbors
                cn_list = [x for x in range(n) if (viol_common >> x) & 1]
                if not cn_list:
                    # Should not happen if bit_count >= 3
                    break
                    
                # Pick random common neighbor
                cn = random.choice(cn_list)
                # Pick random vertex from triple
                tv = random.choice([u, v, w])
                
                # Remove edge (tv, cn)
                adj[tv] &= ~(1 << cn)
                adj[cn] &= ~(1 << tv)
                steps += 1
            
            # If still not valid after max steps, skip this restart or force clean
            if find_violation(adj)[0] is not None:
                # Force clean with more steps or skip
                # Let's try a few more steps
                for _ in range(100):
                    viol_triple, viol_common = find_violation(adj)
                    if viol_triple is None:
                        break
                    u, v, w = viol_triple
                    cn_list = [x for x in range(n) if (viol_common >> x) & 1]
                    if not cn_list: break
                    cn = random.choice(cn_list)
                    tv = random.choice([u, v, w])
                    adj[tv] &= ~(1 << cn)
                    adj[cn] &= ~(1 << tv)
                
                if find_violation(adj)[0] is not None:
                    continue # Skip this bad start

            current_edges = count_edges(adj)
            
            # 3. Improve
            # Greedy Add
            # Try to add every missing edge
            improved = True
            while improved and time.time() - start_time < timeout:
                improved = False
                # Shuffle missing edges to avoid bias
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
                random.shuffle(missing)
                
                for u, v in missing:
                    if time.time() - start_time > timeout:
                        break
                    
                    # Tentatively add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if is_valid(adj):
                        current_edges += 1
                        improved = True
                        # Don't break, keep trying to add more
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        
                # If no improvements in a full pass, try Swap (2-opt)
                if not improved:
                    # Try to remove one edge and add two?
                    # Or just remove one and add one (no gain)? No.
                    # Let's try: Remove random edge, then try to add 2 random missing edges.
                    # If valid, net +1 edge.
                    
                    existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                    if not existing:
                        break
                        
                    # Try a few swaps
                    swap_improved = False
                    for _ in range(100): # Limit swap attempts per pass
                        if time.time() - start_time > timeout:
                            break
                            
                        # Pick edge to remove
                        e_rem = random.choice(existing)
                        u_rem, v_rem = e_rem
                        
                        # Pick 2 edges to add
                        missing_now = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
                        if len(missing_now) < 2:
                            break
                            
                        e_add1, e_add2 = random.sample(missing_now, 2)
                        
                        # Apply
                        adj[u_rem] &= ~(1 << v_rem)
                        adj[v_rem] &= ~(1 << u_rem)
                        
                        u1, v1 = e_add1
                        u2, v2 = e_add2
                        
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        
                        if is_valid(adj):
                            current_edges += 1
                            swap_improved = True
                            improved = True
                            # Update existing list? It's expensive.
                            # Just break and restart the "Greedy Add" phase which rebuilds missing list.
                            break
                        else:
                            # Revert
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            adj[u2] &= ~(1 << v2)
                            adj[v2] &= ~(1 << u2)
                            
                            adj[u_rem] |= (1 << v_rem)
                            adj[v_rem] |= (1 << u_rem)
                            
                    if not swap_improved:
                        break # Stuck

            # Update best
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [a for a in adj]
                
            # Check time
            if time.time() - start_time > timeout - 10:
                break

        # Convert best_adj to list of edges
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
                    
        return edges
