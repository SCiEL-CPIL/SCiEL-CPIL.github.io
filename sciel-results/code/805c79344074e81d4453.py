import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin for safety

    def _find_k5(adj_masks, n):
        """Find a clique of size 5. Returns list of 5 vertices or None."""
        # Try all pairs (v, u) and look for a triangle in their common neighborhood
        for v in range(n):
            mask_v = adj_masks[v]
            # Only consider neighbors u > v to reduce redundancy? 
            # Actually, bitmask handles any order, but let's iterate through bits of mask_v
            mv = mask_v
            while mv:
                lsb = mv & -mv
                u = lsb.bit_length() - 1
                mv ^= lsb
                
                # Common neighbors of v and u
                common = mask_v & adj_masks[u]
                
                # We need a triangle in 'common'
                # Iterate through pairs (w, x) in common
                mc = common
                while mc:
                    lsb_w = mc & -mc
                    w = lsb_w.bit_length() - 1
                    mc ^= lsb_w
                    
                    # Look for x in common that is adjacent to w
                    # x must be in common and in adj_masks[w]
                    candidates_x = common & adj_masks[w]
                    if candidates_x:
                        lsb_x = candidates_x & -candidates_x
                        x = lsb_x.bit_length() - 1
                        return [v, u, w, x, v] # Wait, this returns 4 distincts + v? No.
                        # The clique is {v, u, w, x, y}. 
                        # Wait, my logic above:
                        # v, u are connected.
                        # common = N(v) ∩ N(u).
                        # w in common => w adj v, w adj u.
                        # x in common => x adj v, x adj u.
                        # If w adj x, then {v, u, w, x} is a K4.
                        # I need a K5.
                        # So I need {v, u, w, x, y} to be a K5.
                        # This means v, u, w, x, y are all mutually adjacent.
                        # Let's restart the logic for K5.
                        
                        # To find K5:
                        # Pick v.
                        # Pick u in N(v).
                        # Let C = N(v) ∩ N(u).
                        # We need a K3 in C.
                        # If we find w, x, y in C forming a triangle, then {v, u, w, x, y} is a K5.
                        
                        # So, inside the loop for w in C:
                        #   Look for an edge (x, y) in C such that x, y are adjacent.
                        #   Actually, just find a triangle in C.
                        
                        # Let's rewrite the inner part.
                        pass
        return None

    def _find_k5_correct(adj_masks, n):
        for v in range(n):
            mask_v = adj_masks[v]
            mv = mask_v
            while mv:
                lsb_u = mv & -mv
                u = lsb_u.bit_length() - 1
                mv ^= lsb_u
                
                # C = N(v) ∩ N(u)
                # Note: v is not in N(v) and u is not in N(u) usually, but bitmasks are set by edge presence.
                # adj_masks[v] has bit u set if edge (v,u) exists.
                # adj_masks[u] has bit v set.
                # Common neighbors are bits set in both.
                # Does 'common' include v or u?
                # adj_masks[v] does NOT have bit v set (no self loop).
                # adj_masks[u] has bit v set.
                # So v is in adj_masks[u] but not in adj_masks[v]. Thus v is NOT in common.
                # Similarly u is NOT in common.
                # So common contains only vertices w such that w!=v, w!=u, w adj v, w adj u.
                common = mask_v & adj_masks[u]
                
                # Find a triangle in 'common'
                # Iterate w in common
                mc = common
                while mc:
                    lsb_w = mc & -mc
                    w = lsb_w.bit_length() - 1
                    mc ^= lsb_w
                    
                    # Find edge (x, y) in common such that x, y are both neighbors of w?
                    # No, we just need x, y in common, x adj y, and x adj w, y adj w.
                    # Since x, y in common, they are already adj to v and u.
                    # If x, y, w form a triangle in common, then {v, u, w, x, y} is a K5.
                    
                    # Candidates for x: neighbors of w that are in common
                    # But x must be distinct from w.
                    # adj_masks[w] & common gives neighbors of w in common.
                    # Let's pick one, say x.
                    # Then we need y in common, y adj x, y adj w.
                    # y must be in (adj_masks[w] & common) AND (adj_masks[x] & common).
                    
                    candidates_x = adj_masks[w] & common
                    # Remove w from candidates_x? adj_masks[w] doesn't have w set. So it's fine.
                    # But we need distinct x, y.
                    
                    cx = candidates_x
                    while cx:
                        lsb_x = cx & -cx
                        x = lsb_x.bit_length() - 1
                        cx ^= lsb_x
                        
                        # Find y in common, adj to w, adj to x
                        # y must be in (adj_masks[w] & common) & adj_masks[x]
                        # And y != w, y != x.
                        # adj_masks[w] & common contains x?
                        # x is in common. x is adj to w (since x in adj_masks[w]).
                        # So x is in (adj_masks[w] & common).
                        # We need y != x.
                        
                        candidates_y = (adj_masks[w] & common) & adj_masks[x]
                        # Remove x from candidates_y
                        # x is in candidates_y?
                        # x is in adj_masks[w] (yes), in common (yes), in adj_masks[x]? No (no self loop).
                        # So x is NOT in candidates_y.
                        # So we just need any bit in candidates_y.
                        
                        if candidates_y:
                            # Found y
                            # lsb_y = candidates_y & -candidates_y
                            # y = lsb_y.bit_length() - 1
                            return [v, u, w, x] # Wait, I need 5 vertices.
                            # The vertices are v, u, w, x, y.
                            # I have v, u, w, x. I need y.
                            # candidates_y is non-empty, so pick one.
                            lsb_y = candidates_y & -candidates_y
                            y = lsb_y.bit_length() - 1
                            return [v, u, w, x, y]
        return None

    def _find_ind5(adj_masks, n):
        # Find independent set of size 5.
        # This is a K5 in the complement graph.
        # Build complement masks? Or just use ~adj_masks?
        # comp_masks[i] = full_mask ^ adj_masks[i] ^ (1<<i)
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for i in range(n):
            comp_masks[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        
        return _find_k5_correct(comp_masks, n)

    def _build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def _paley_graph(q):
        # q must be a prime p ≡ 1 (mod 4)
        # Quadratic residues
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff in residues:
                    edges.append((i, j))
        return edges

    best_n = 37
    best_edges = _paley_graph(37)

    # Try to repair Paley graphs for larger primes
    # Primes p ≡ 1 (mod 4)
    # 41, 53, 61, 73, 89, 97
    # Note: R(5,5) <= 46. So we can only aim for n <= 46.
    # Paley(41) has 41 vertices.
    # Paley(53) has 53 vertices. If we can repair it to be (5,5)-free, that's great, but n=53 > 46 is impossible.
    # Wait, the task says "n must be at most MAX_VERTICES". 
    # The task description says "Angeltveit and McKay proved R(5,5) <= 46".
    # This implies no such graph exists for n=47 or more.
    # So we are looking for n in [38, 46].
    # Paley(41) is a candidate.
    # We can also try to add vertices to Paley(37) or repair Paley(41).
    
    # Strategy 1: Repair Paley(41)
    # Strategy 2: Add vertices to Paley(37) or repaired Paley(41)
    # Strategy 3: Random search with repair (original)
    
    # Let's focus on Paley(41) repair first.
    q = 41
    seed_edges = _paley_graph(q)
    adj_masks = _build_adj_masks(q, seed_edges)
    edges_set = set(seed_edges)
    
    # Repair loop for Paley(41)
    # If we find a valid graph, we record it.
    # Then we try to add vertices to it.
    
    found_41 = False
    valid_edges_41 = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Use a copy of the edges
        current_edges = list(edges_set)
        current_adj = _build_adj_masks(q, current_edges)
        
        # Repair loop
        success = False
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Check for K5
            k5 = _find_k5_correct(current_adj, q)
            if k5:
                # Remove a random edge from the clique
                # The clique is k5[0], k5[1], k5[2], k5[3], k5[4]
                # Pick a random pair
                i, j = random.sample(k5, 2)
                if i > j:
                    i, j = j, i
                if (i, j) in current_edges:
                    current_edges.remove((i, j))
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
                continue
            
            # Check for Independent Set 5
            ind5 = _find_ind5(current_adj, q)
            if ind5:
                # Add a random edge in the independent set
                i, j = random.sample(ind5, 2)
                if i > j:
                    i, j = j, i
                if (i, j) not in current_edges:
                    current_edges.append((i, j))
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                continue
            
            # Valid!
            if q > best_n:
                best_n = q
                best_edges = current_edges
                valid_edges_41 = current_edges
                found_41 = True
            success = True
            break
        
        if found_41 and best_n >= q:
            # We found a valid 41-graph.
            # Can we add more vertices?
            # Let's try to extend it to 42, 43, ..., 46.
            break
    
    # If we found a valid 41-graph, try to extend it
    if found_41 and valid_edges_41 is not None:
        base_n = 41
        base_edges = list(valid_edges_41)
        base_adj = _build_adj_masks(base_n, base_edges)
        
        # Try to add vertices one by one
        current_n = base_n
        current_edges = base_edges
        current_adj = base_adj
        
        # We'll try to add vertices up to 46
        for target_n in range(base_n + 1, 47):
            if time.time() > deadline:
                break
            
            # Try to add vertex 'current_n'
            # Connect it to existing vertices to avoid independent sets and cliques.
            # Random restarts for adding a single vertex
            added = False
            for restart in range(50):
                if time.time() > deadline:
                    break
                
                # Initialize new vertex adjacencies
                # Try a random subset of existing vertices?
                # Or start with all edges and remove?
                # Or start with no edges and add?
                
                # Let's start with a random graph for the new vertex connections
                new_adj = 0
                # Probability of connection?
                # To avoid independent sets, it needs to be connected to many.
                # To avoid cliques, it shouldn't be connected to too many in a clique.
                # Let's try p=0.5 for connections.
                for i in range(current_n):
                    if random.random() < 0.5:
                        new_adj |= (1 << i)
                
                # Build new adj masks
                new_adj_list = list(current_adj)
                new_adj_list.append(new_adj)
                
                # Update adj masks for existing vertices
                for i in range(current_n):
                    if (new_adj >> i) & 1:
                        new_adj_list[i] |= (1 << current_n)
                
                # Check validity
                # This is expensive if we do full search every time.
                # But n is small (42-46).
                
                k5 = _find_k5_correct(new_adj_list, current_n + 1)
                if k5:
                    continue
                ind5 = _find_ind5(new_adj_list, current_n + 1)
                if ind5:
                    continue
                
                # Valid!
                # Update state
                # Add edges to list
                for i in range(current_n):
                    if (new_adj >> i) & 1:
                        current_edges.append((i, current_n))
                
                current_n += 1
                current_adj = new_adj_list
                added = True
                
                if current_n > best_n:
                    best_n = current_n
                    best_edges = list(current_edges)
                
                break # Try to add next vertex
            
            if not added:
                # Failed to add this vertex, maybe try a different base or give up?
                # We could try to repair the graph with current_n+1 vertices.
                # But for now, let's stop extending if we can't add the next one.
                # Or we could try multiple restarts for the whole extension process.
                pass
    
    # If we didn't find 41, or failed to extend, we might have a lower score.
    # Let's also run the original random repair for n=46 down to 38 if we have time?
    # The Paley strategy is likely faster.
    
    # If best_n is still 37, maybe we should try harder?
    # But the deadline is strict.
    
    return best_n, best_edges
