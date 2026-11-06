"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    n = 40
    
    def creates_c6(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6 using bit operations."""
        nu_mask = adj_bits[u] & ~(1 << v)
        nv_mask = adj_bits[v] & ~(1 << u)
        
        temp_u = nu_mask
        while temp_u:
            lsb_u = temp_u & -temp_u
            x1 = lsb_u.bit_length() - 1
            temp_u ^= lsb_u
            
            # Precompute mask for x1's useful neighbors
            mask_x1 = adj_bits[x1] & ~(1 << u) & ~(1 << v)
            
            temp_v = nv_mask
            while temp_v:
                lsb_v = temp_v & -temp_v
                x4 = lsb_v.bit_length() - 1
                temp_v ^= lsb_v
                
                if x1 == x4:
                    continue
                
                mask_x4 = adj_bits[x4] & ~(1 << u) & ~(1 << v)
                
                # Remove x1 from mask_x4 and x4 from mask_x1
                mask_x1_eff = mask_x1 & ~(1 << x4)
                mask_x4_eff = mask_x4 & ~(1 << x1)
                
                # Check if any vertex in mask_x1_eff is adjacent to any vertex in mask_x4_eff
                # Iterate over the smaller set
                if mask_x1_eff.bit_count() < mask_x4_eff.bit_count():
                    temp_a = mask_x1_eff
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x2 = lsb_a.bit_length() - 1
                        if (adj_bits[x2] & mask_x4_eff) != 0:
                            return True
                        temp_a ^= lsb_a
                else:
                    temp_a = mask_x4_eff
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x3 = lsb_a.bit_length() - 1
                        if (adj_bits[x3] & mask_x1_eff) != 0:
                            return True
                        temp_a ^= lsb_a
        return False

    def greedy_add(adj_bits, current_edges, current_count):
        """Try to add all possible non-C6-creating edges in random order."""
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    candidates.append((u, v))
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 5.0:
                break
            if not creates_c6(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges.append([u, v])
                current_count += 1
        return current_edges, current_count

    def local_search(adj_bits, current_edges, current_count):
        """Local search: remove an edge, try to add multiple new edges."""
        if not current_edges:
            return current_edges, current_count
            
        improved = True
        max_rounds = 50
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Try removing each edge and see if we can add more
            random.shuffle(current_edges)
            for i in range(len(current_edges)):
                if time.time() > deadline - 5.0:
                    break
                u, v = current_edges[i]
                
                # Temporarily remove edge (u, v)
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
                # Try to add all possible edges now
                added = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if not (adj_bits[a] >> b) & 1:
                            if not creates_c6(adj_bits, a, b):
                                adj_bits[a] |= (1 << b)
                                adj_bits[b] |= (1 << a)
                                added.append((a, b))
                
                new_count = current_count - 1 + len(added)
                
                if len(added) >= 1:
                    # Accept this move
                    current_edges.pop(i)
                    for a, b in added:
                        current_edges.append([a, b])
                    current_count = new_count
                    improved = True
                    break
                else:
                    # Revert
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        return current_edges, current_count

    def simulated_annealing(adj_bits, current_edges, current_count):
        """Simulated Annealing with structured moves."""
        if not current_edges:
            return current_edges, current_count
            
        T = 2.0
        cooling = 0.998
        min_T = 0.1
        max_steps = 3000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
                
            # Pick a random edge to remove
            i = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[i]
            
            # Remove edge
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            current_edges.pop(i)
            new_count = current_count - 1
            
            # Try to add up to 3 random missing edges
            added_edges = []
            attempts = 0
            max_add_attempts = 30
            
            while attempts < max_add_attempts and len(added_edges) < 3:
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    attempts += 1
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                    
                if (adj_bits[ru] >> rv) & 1:
                    attempts += 1
                    continue
                    
                if not creates_c6(adj_bits, ru, rv):
                    adj_bits[ru] |= (1 << rv)
                    adj_bits[rv] |= (1 << ru)
                    current_edges.append([ru, rv])
                    added_edges.append((ru, rv))
                    new_count += 1
                attempts += 1
            
            delta = new_count - current_count
            if delta >= 0 or random.random() < pow(2.718, delta / T):
                current_count = new_count
            else:
                # Revert: remove added edges, restore original edge
                for a, b in added_edges:
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
                    # Remove from current_edges (find and remove)
                    for j in range(len(current_edges) - 1, -1, -1):
                        if current_edges[j] == [a, b]:
                            current_edges.pop(j)
                            break
                current_edges.append([u, v])
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            T *= cooling
            
        return current_edges, current_count

    # Main loop
    best_edges = []
    best_count = 0
    
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with 8 K5s
        adj_bits = [0] * n
        current_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    current_edges.append([u, v])
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        current_count = len(current_edges)
        
        # Greedy phase
        current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        # Local search phase
        if time.time() < deadline - 8.0:
            current_edges, current_count = local_search(adj_bits, current_edges, current_count)
        
        # SA phase
        if time.time() < deadline - 5.0:
            current_edges, current_count = simulated_annealing(adj_bits, current_edges, current_count)
        
        # Final greedy pass
        if time.time() < deadline - 3.0:
            current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edges)
    
    return best_edges
# EVOLVE-BLOCK-END
