import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    def bits_to_list(mask):
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            result.append(idx)
            temp ^= lsb
        return result

    def has_k3_in_mask(adj, mask):
        """Check if induced subgraph on mask has a K3 (triangle)"""
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            # For each other vertex v in mask that is a neighbor of u
            neigh_in_mask = adj[u] & mask & ~(1 << u)
            v_list = bits_to_list(neigh_in_mask)
            for v in v_list:
                # Check if v has any common neighbor in mask (other than u)
                if adj[v] & neigh_in_mask & ~(1 << v) & ~(1 << u):
                    return True
        return False

    def has_i3_in_mask(adj, mask):
        """Check if induced subgraph on mask has an independent set of size 3"""
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            non_neigh_in_mask = mask & ~adj[u] & ~(1 << u)
            v_list = bits_to_list(non_neigh_in_mask)
            for v in v_list:
                # Check if v has any common non-neighbor in mask (other than u)
                if non_neigh_in_mask & ~adj[v] & ~(1 << v) & ~(1 << u):
                    return True
        return False

    def has_k4_in_mask(adj, mask):
        """Check if induced subgraph on mask has a K4"""
        verts = bits_to_list(mask)
        for u in verts:
            neigh = adj[u] & mask & ~(1 << u)
            if has_k3_in_mask(adj, neigh):
                return True
        return False

    def has_i4_in_mask(adj, mask):
        """Check if induced subgraph on mask has an independent set of size 4"""
        verts = bits_to_list(mask)
        for u in verts:
            non_neigh = mask & ~adj[u] & ~(1 << u)
            if has_i3_in_mask(adj, non_neigh):
                return True
        return False

    def is_valid(adj, n):
        """Check if the graph is (5,5)-free: no K5 and no independent 5-set"""
        all_mask = (1 << n) - 1
        for u in range(n):
            if has_k4_in_mask(adj, adj[u]):
                return False
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4_in_mask(adj, non_neigh):
                return False
        return True

    # Verify seed
    if not is_valid(adj_masks, n):
        return n, []

    best_n = n
    best_adj = adj_masks[:]
    current_n = n
    current_adj = adj_masks[:]
    all_mask = (1 << current_n) - 1

    def try_add_vertex(adj, n, neighbors_mask, all_mask):
        """Try to add a new vertex connected to vertices in neighbors_mask"""
        if has_k4_in_mask(adj, neighbors_mask):
            return None
        non_neighbors_mask = all_mask & ~neighbors_mask
        if has_i4_in_mask(adj, non_neighbors_mask):
            return None
        
        new_n = n + 1
        new_adj = adj + [0]
        for v in range(n):
            if neighbors_mask & (1 << v):
                new_adj[v] |= (1 << new_n)
                new_adj[new_n] |= (1 << v)
        return new_n, new_adj

    def try_flip_edge(adj, n, i, j, all_mask):
        """Try to flip the edge between i and j"""
        if i == j:
            return None
        if adj[i] & (1 << j):
            # Remove edge: check no new I3 in common non-neighbors
            mask = (all_mask & ~adj[i] & ~adj[j]) & ~(1 << i) & ~(1 << j)
            if has_i3_in_mask(adj, mask):
                return None
            new_adj = adj[:]
            new_adj[i] &= ~(1 << j)
            new_adj[j] &= ~(1 << i)
            return new_adj
        else:
            # Add edge: check no new K3 in common neighbors
            mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
            if has_k3_in_mask(adj, mask):
                return None
            new_adj = adj[:]
            new_adj[i] |= (1 << j)
            new_adj[j] |= (1 << i)
            return new_adj

    # Generate diverse candidate neighbor sets for vertex addition
    def generate_candidates(adj, n, all_mask):
        candidates = []
        
        # Strategy 1: Neighborhoods of existing vertices
        for v in range(n):
            candidates.append(adj[v])
            candidates.append(all_mask & ~adj[v] & ~(1 << v))
        
        # Strategy 2: Structured patterns based on Paley symmetry
        # Try all shifts of a base pattern
        base_size = n // 2
        for shift in range(n):
            mask = 0
            for k in range(base_size):
                v = (shift + k * 2) % n  # Even-spaced
                mask |= (1 << v)
            candidates.append(mask)
        
        # Strategy 3: Random balanced subsets
        for _ in range(20):
            size = random.randint(n // 3, 2 * n // 3)
            neighbors = random.sample(range(n), size)
            mask = 0
            for v in neighbors:
                mask |= (1 << v)
            candidates.append(mask)
        
        # Strategy 4: Neighborhoods of random pairs (intersection/union)
        for _ in range(10):
            u, v = random.sample(range(n), 2)
            candidates.append(adj[u] & adj[v])  # Common neighbors
            candidates.append(adj[u] | adj[v])  # Union of neighbors
        
        random.shuffle(candidates)
        return candidates

    # Main search: alternate between vertex addition and edge flipping
    stuck_count = 0
    add_attempts = 0
    
    while time.time() < deadline:
        # Try to add a vertex
        added = False
        if add_attempts % 5 == 0:  # Try vertex addition every 5th iteration
            candidates = generate_candidates(current_adj, current_n, all_mask)
            
            for mask in candidates:
                if time.time() > deadline:
                    break
                result = try_add_vertex(current_adj, current_n, mask, all_mask)
                if result is not None:
                    current_n, current_adj = result
                    best_n = current_n
                    best_adj = current_adj[:]
                    all_mask = (1 << current_n) - 1
                    added = True
                    stuck_count = 0
                    break
        
        if added:
            add_attempts = 0
            continue
        
        add_attempts += 1
        stuck_count += 1
        
        # If stuck for a while, reset to best and try different approach
        if stuck_count > 200:
            current_n = best_n
            current_adj = best_adj[:]
            all_mask = (1 << current_n) - 1
            stuck_count = 0
            add_attempts = 0
        
        # Flip an edge using simulated annealing-like approach
        i = random.randint(0, current_n - 1)
        j = random.randint(0, current_n - 1)
        
        new_adj = try_flip_edge(current_adj, current_n, i, j, all_mask)
        if new_adj is not None:
            current_adj = new_adj

    # Final polish: try a more thorough local search
    if time.time() < deadline - 30:
        polish_deadline = time.time() + 25
        # Try all single-edge flips
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if time.time() > polish_deadline:
                    break
                new_adj = try_flip_edge(current_adj, current_n, i, j, all_mask)
                if new_adj is not None:
                    current_adj = new_adj
                    # Try to add a vertex after each successful flip
                    candidates = generate_candidates(current_adj, current_n, all_mask)
                    for mask in candidates[:5]:
                        result = try_add_vertex(current_adj, current_n, mask, all_mask)
                        if result is not None:
                            current_n, current_adj = result
                            best_n = current_n
                            best_adj = current_adj[:]
                            all_mask = (1 << current_n) - 1
                            break
            if time.time() > polish_deadline:
                break

    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
