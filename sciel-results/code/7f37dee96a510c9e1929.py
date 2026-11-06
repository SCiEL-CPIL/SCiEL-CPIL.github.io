"""K4-free graphs without independent 6-sets (Ramsey number R(4,6))."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # --- Bitmask-based graph utilities ---
    # Represent adjacency as a list of integers (bitmasks) for fast operations
    
    def count_bits(x):
        return bin(x).count('1')
    
    def has_k4(adj, n):
        """Check if graph with adjacency bitmasks has a K4."""
        for a in range(n):
            # Get neighbors of a
            na = adj[a]
            # For each pair of neighbors b, c of a, check if b~c and they share a common neighbor
            # More efficient: for each b in neighbors of a, check triangles in N(a)
            # A K4 containing a means there's a triangle in N(a)
            # Triangle in N(a): b,c,d all in N(a) with b~c, c~d, d~b
            # Simpler: for each edge (b,c) in N(a), check if they have a common neighbor in N(a)
            nb_list = [b for b in range(n) if (na >> b) & 1]
            for i in range(len(nb_list)):
                b = nb_list[i]
                for j in range(i + 1, len(nb_list)):
                    c = nb_list[j]
                    if not (adj[b] >> c) & 1:
                        continue
                    # b~c, both in N(a). Check common neighbor d in N(a)
                    common = adj[b] & adj[c] & na  # common neighbors of b and c that are also neighbors of a
                    # Remove b and c themselves
                    common &= ~(1 << b)
                    common &= ~(1 << c)
                    if common:
                        return True
        return False
    
    def has_independent_set(adj, n, size):
        """Check if graph has an independent set of given size using backtracking with bitmasks."""
        if size <= 1:
            return True
        if n < size:
            return False
        
        # Use a recursive approach: pick a vertex, either include it (and remove its neighbors)
        # or exclude it
        # We work with the set of available vertices as a bitmask
        available = (1 << n) - 1
        
        def backtrack(avail, remaining):
            if remaining == 0:
                return True
            if count_bits(avail) < remaining:
                return False
            if not avail:
                return False
            
            # Pick the lowest set bit
            v = (avail & (-avail)).bit_length() - 1
            rest = avail & ~(1 << v)
            
            # Include v: remove v and all its neighbors from available
            new_avail = rest & ~adj[v]
            if backtrack(new_avail, remaining - 1):
                return True
            
            # Exclude v
            if backtrack(rest, remaining):
                return True
            
            return False
        
        return backtrack(available, size)
    
    def is_valid(adj, n):
        """Check if graph is K4-free and has no independent set of size 6."""
        if has_k4(adj, n):
            return False
        if has_independent_set(adj, n, 6):
            return False
        return True
    
    def try_add_vertex(adj, n, neighbors_mask):
        """Try adding a new vertex n with given neighbor set. Returns new adj or None."""
        new_n = n + 1
        new_adj = adj[:] + [0]
        
        # Set up edges for new vertex
        new_adj[n] = neighbors_mask
        for v in range(n):
            if (neighbors_mask >> v) & 1:
                new_adj[v] |= (1 << n)
        
        # Check K4: new vertex connects to neighbors. K4 involving new vertex means
        # there's a triangle among the neighbors.
        # Check if the subgraph induced by neighbors has a triangle
        nb_list = [v for v in range(n) if (neighbors_mask >> v) & 1]
        for i in range(len(nb_list)):
            a = nb_list[i]
            for j in range(i + 1, len(nb_list)):
                b = nb_list[j]
                if not (adj[a] >> b) & 1:
                    continue
                # a~b, both neighbors of new. Check for c in neighbors adjacent to both
                common = adj[a] & adj[b] & neighbors_mask
                common &= ~(1 << a)
                common &= ~(1 << b)
                if common:
                    return None  # K4 formed
        
        # Check independent set of 6: 
        # Non-neighbors of new vertex (in old graph) + new vertex
        non_neighbors_mask = ((1 << n) - 1) & ~neighbors_mask
        # Case 1: 5 from non_neighbors + new vertex
        # Need independent set of size 5 in non_neighbors
        if count_bits(non_neighbors_mask) >= 5:
            # Check if non_neighbors has independent set of size 5
            avail = non_neighbors_mask
            if _has_independent_set_bitmask(adj, avail, 5):
                return None
        
        # Case 2: 6 from non_neighbors (new not involved)
        if count_bits(non_neighbors_mask) >= 6:
            avail = non_neighbors_mask
            if _has_independent_set_bitmask(adj, avail, 6):
                return None
        
        return new_adj
    
    def _has_independent_set_bitmask(adj, available, size):
        """Check if there's an independent set of given size within 'available' vertices."""
        if size <= 1:
            return True
        if count_bits(available) < size:
            return False
        if not available:
            return False
        
        def backtrack(avail, remaining):
            if remaining == 0:
                return True
            if count_bits(avail) < remaining:
                return False
            if not avail:
                return False
            
            # Pick vertex with fewest neighbors in avail (for better pruning)
            best_v = -1
            best_count = n + 1
            temp = avail
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                temp &= ~(1 << v)
                # Count neighbors of v in avail
                nb_in_avail = count_bits(adj[v] & avail)
                if nb_in_avail < best_count:
                    best_count = nb_in_avail
                    best_v = v
                    if nb_in_avail == 0:
                        break
            
            v = best_v
            rest = avail & ~(1 << v)
            
            # Include v: remove v and all its neighbors
            new_avail = rest & ~adj[v]
            if backtrack(new_avail, remaining - 1):
                return True
            
            # Exclude v
            if backtrack(rest, remaining):
                return True
            
            return False
        
        return backtrack(available, size)
    
    # --- Local search: try to improve by edge flips ---
    def local_search(adj, n, max_iter=2000):
        """Try to fix violations by flipping edges. Returns improved adj or None."""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Check if valid
            if is_valid(adj, n):
                return adj
            
            # Find a violation and try to fix it
            # Try random edge flips
            for _ in range(50):
                if time.time() > deadline:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge (u,v)
                old_adj = adj[:]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                if is_valid(adj, n):
                    return adj
                
                # Revert
                adj = old_adj
        
        return None
    
    # --- Strategy 1: Start with known good graph and grow ---
    # Paley graph on 17 vertices is K4-free and has no independent set of 6? 
    # Actually Paley(17) has independence number 4, so it's fine.
    
    def make_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    # Strategy 2: Use a structured construction
    # The known 35-vertex graph by Exoo can be approached via random construction + local search
    
    best_n = 0
    best_adj = None
    
    # Start from Paley(17)
    adj = make_paley_17()
    n = 17
    
    # Verify it's valid
    if not is_valid(adj, n):
        # Try to fix
        adj = local_search(adj, n)
        if adj is None:
            # Fallback to empty graph
            adj = [0] * 1
            n = 1
    
    best_n = n
    best_adj = adj[:]
    
    # Try to grow the graph
    while n < 38:  # Aim for 35+
        if time.time() > deadline:
            break
        
        added = False
        # Try multiple random neighbor sets
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random neighbor set with bias toward moderate density
            k = random.randint(n // 3, 2 * n // 3)
            neighbors = random.sample(range(n), k)
            neighbors_mask = 0
            for v in neighbors:
                neighbors_mask |= (1 << v)
            
            new_adj = try_add_vertex(adj, n, neighbors_mask)
            if new_adj is not None:
                adj = new_adj
                n += 1
                added = True
                if n > best_n:
                    best_n = n
                    best_adj = adj[:]
                break
        
        if not added:
            # Try local search to see if we can fix the current graph to allow growth
            # Or try a different approach: start fresh with random graph
            if n < best_n:
                adj = best_adj[:]
                n = best_n
            
            # Try random restart with local search
            success = False
            for restart in range(5):
                if time.time() > deadline:
                    break
                
                # Random graph with density ~0.5
                random.seed(42 + restart)
                rand_adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            rand_adj[i] |= (1 << j)
                            rand_adj[j] |= (1 << i)
                
                # Local search to make it valid
                fixed_adj = local_search(rand_adj, n, max_iter=500)
                if fixed_adj is not None:
                    adj = fixed_adj
                    if n > best_n:
                        best_n = n
                        best_adj = adj[:]
                    success = True
                    break
            
            if not success:
                break
    
    # Try to push further with more aggressive local search
    if time.time() < deadline - 10:
        for target_n in range(n + 1, min(38, n + 5)):
            if time.time() > deadline:
                break
            
            # Try to add vertices one by one with more attempts
            adj2 = adj[:]
            n2 = n
            for _ in range(target_n - n):
                if time.time() > deadline:
                    break
                added = False
                for attempt in range(200):
                    if time.time() > deadline:
                        break
                    k = random.randint(n2 // 3, 2 * n2 // 3)
                    neighbors = random.sample(range(n2), k)
                    neighbors_mask = 0
                    for v in neighbors:
                        neighbors_mask |= (1 << v)
                    
                    new_adj = try_add_vertex(adj2, n2, neighbors_mask)
                    if new_adj is not None:
                        adj2 = new_adj
                        n2 += 1
                        added = True
                        break
                
                if not added:
                    break
            
            if n2 > best_n:
                best_n = n2
                best_adj = adj2[:]
    
    # Convert best_adj back to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
