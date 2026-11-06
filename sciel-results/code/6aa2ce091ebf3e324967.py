import time
import random
import sys

def construct():
    # Set a hard deadline
    deadline = time.time() + 290.0
    
    # 1. Generate Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = q
    
    # Helper to get K4s and Alpha4s (K4 in complement)
    def get_k4s(adj, n):
        k4s = []
        # Iterate over all combinations of 4 vertices? O(n^4) is 37^4 ~ 1.8M. Feasible.
        # But we can do it faster using bitmasks.
        # A K4 is a set of 4 vertices {a,b,c,d} such that all pairs are edges.
        # We can iterate a < b < c < d.
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n):
                    if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if ((adj[a] >> d) & 1 and (adj[b] >> d) & 1 and (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            k4s.append(mask)
        return k4s

    def get_alpha4s(adj, n):
        # Independent set of size 4 in G is K4 in complement.
        # Complement adjacency: ~adj[i] & ((1<<n)-1) & ~(1<<i)
        alpha4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if (adj[a] >> b) & 1: # If edge exists, not in independent set
                    continue
                for c in range(b + 1, n):
                    if ((adj[a] >> c) & 1 or (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if not ((adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            alpha4s.append(mask)
        return alpha4s

    # Initial lists
    k4_list = get_k4s(adj, n)
    alpha4_list = get_alpha4s(adj, n)
    
    # Function to check if a neighbor set N is valid for a new vertex
    # N is a bitmask of neighbors in the current graph (0..n-1)
    def is_valid_neighbor_set(N, n, k4_list, alpha4_list):
        # Check K4 in N
        for k4 in k4_list:
            if (N & k4) == k4:
                return False
        # Check Alpha4 in ~N
        # ~N within n bits
        not_N = ((1 << n) - 1) & ~N
        for a4 in alpha4_list:
            if (not_N & a4) == a4:
                return False
        return True

    # Local search to find a valid N
    def find_valid_N(n, k4_list, alpha4_list, time_limit=5.0):
        start_time = time.time()
        best_score = float('inf')
        best_N = None
        full_mask = (1 << n) - 1
        
        # Try a few random restarts
        for restart in range(20):
            # Initialize N randomly, roughly half bits set
            N = 0
            for i in range(n):
                if random.random() < 0.5:
                    N |= (1 << i)
            
            current_score = 0
            # Calculate initial score
            for k4 in k4_list:
                if (N & k4) == k4:
                    current_score += 1
            not_N = full_mask & ~N
            for a4 in alpha4_list:
                if (not_N & a4) == a4:
                    current_score += 1
            
            if current_score == 0:
                return N
            
            if current_score < best_score:
                best_score = current_score
                best_N = N
            
            # Hill climbing
            T = 1.0
            while time.time() - start_time < time_limit:
                if current_score == 0:
                    return N
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                delta_bit = 1 << bit
                
                # Calculate change in score
                # This requires checking only K4s/Alpha4s involving this bit?
                # For simplicity, recalculate score or use incremental update.
                # Incremental is complex. Let's just recalculate if n is small enough?
                # n is up to 42. k4_list size grows.
                # Recalculating score is O(len(k4_list) + len(alpha4_list)).
                # len(k4_list) for P(37) is ~?
                # Let's assume it's small enough (thousands).
                
                new_N = N ^ delta_bit
                
                new_score = 0
                for k4 in k4_list:
                    if (new_N & k4) == k4:
                        new_score += 1
                new_not_N = full_mask & ~new_N
                for a4 in alpha4_list:
                    if (new_not_N & a4) == a4:
                        new_score += 1
                
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    N = new_N
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_N = N
                T *= 0.99 # Cool down slightly
            
            if best_score == 0:
                return best_N

        return best_N if best_score == 0 else None

    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 10.0:
            break
        
        # Time budget per vertex
        time_per_vertex = 5.0
        if target_n > 40:
            time_per_vertex = 10.0 # Give more time for harder steps
        
        found_N = find_valid_N(n, k4_list, alpha4_list, time_per_vertex)
        
        if found_N is None:
            # Try a few more times with different random seeds or longer time?
            # If we fail, we might be stuck.
            # We can try to modify the existing graph? Too complex.
            # Just break.
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha4_list
        # New K4s: Any K3 in the new vertex's neighborhood + the new vertex
        # New Alpha4s: Any Alpha3 in the new vertex's non-neighborhood + the new vertex
        
        # Get K3s in adj[new_vertex]
        new_k4s = []
        neigh_mask = adj[new_vertex]
        # Iterate over triples in neigh_mask
        # Efficient way: iterate a < b < c in neigh_mask
        # We can just iterate all triples in 0..n-2 and check if in neigh_mask
        # n is small, so O(n^3) is fine.
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        # Get Alpha3s in ~neigh_mask (within 0..n-2)
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1: # Edge exists, so not independent
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Optional: Prune lists? No, just keep them.

    # Convert adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    
    return n, edges
