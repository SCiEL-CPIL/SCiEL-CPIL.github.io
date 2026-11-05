import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
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
        """Convert bitmask to list of set bit positions."""
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            result.append(idx)
            temp ^= lsb
        return result

    def has_k4_in_mask(adj, mask):
        """Check if there's a K4 in the subgraph induced by vertices in mask."""
        if mask == 0:
            return False
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            neigh_in_mask = adj[u] & mask & ~(1 << u)
            neigh_list = bits_to_list(neigh_in_mask)
            for j in range(len(neigh_list)):
                v = neigh_list[j]
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v] & mask & ~(1 << u) & ~(1 << v)
                    if common == 0:
                        continue
                    c_list = bits_to_list(common)
                    for a in c_list:
                        if adj[a] & common & ~(1 << a):
                            return True
        return False

    def has_i4_in_mask(adj, mask):
        """Check if there's an independent set of size 4 in the subgraph induced by vertices in mask."""
        if mask == 0:
            return False
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            non_neigh_in_mask = mask & ~adj[u] & ~(1 << u)
            non_neigh_list = bits_to_list(non_neigh_in_mask)
            for j in range(len(non_neigh_list)):
                v = non_neigh_list[j]
                common_non_neigh = mask & ~adj[u] & ~adj[v] & ~(1 << u) & ~(1 << v)
                if common_non_neigh == 0:
                    continue
                c_list = bits_to_list(common_non_neigh)
                for a in c_list:
                    for b in c_list:
                        if a == b:
                            continue
                        if not (adj[a] & (1 << b)):
                            return True
        return False

    def can_add_vertex(adj, n, neighbors_mask):
        """Check if we can add a new vertex connected to exactly the vertices in neighbors_mask."""
        if has_k4_in_mask(adj, neighbors_mask):
            return False
        all_mask = (1 << n) - 1
        non_neighbors_mask = all_mask & ~neighbors_mask
        if has_i4_in_mask(adj, non_neighbors_mask):
            return False
        return True

    def is_valid(adj, n):
        """Check if the graph has no K5 and no I5."""
        for u in range(n):
            if has_k4_in_mask(adj, adj[u]):
                return False
        all_mask = (1 << n) - 1
        for u in range(n):
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4_in_mask(adj, non_neigh):
                return False
        return True

    # Verify seed
    if not is_valid(adj_masks, n):
        return n, []

    best_n = n
    best_adj = adj_masks[:]

    # 2. Try to extend using backtracking search with smarter heuristics
    current_n = n
    current_adj = adj_masks[:]
    
    # Helper: try to add a vertex with given neighbor set
    def try_add_vertex(adj, n, neighbors_mask):
        if can_add_vertex(adj, n, neighbors_mask):
            new_n = n + 1
            new_adj = adj + [0]
            for v in range(n):
                if neighbors_mask & (1 << v):
                    new_adj[v] |= (1 << new_n)
                    new_adj[new_n] |= (1 << v)
            return new_n, new_adj
        return None

    # Helper: flip edge (i,j) and check validity
    def try_flip_edge(adj, n, i, j):
        # Check if flipping would create K5 or I5
        # If we add edge (i,j): check if i and j have 3 common neighbors that form a triangle
        # If we remove edge (i,j): check if i and j have 3 common non-neighbors that form an independent set
        
        # For now, just try the flip and check full validity (expensive but correct)
        new_adj = adj[:]
        if adj[i] & (1 << j):
            # Remove edge
            new_adj[i] &= ~(1 << j)
            new_adj[j] &= ~(1 << i)
        else:
            # Add edge
            new_adj[i] |= (1 << j)
            new_adj[j] |= (1 << i)
        
        if is_valid(new_adj, n):
            return new_adj
        return None

    # Main search loop
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline:
            break
            
        found = False
        
        # Strategy 1: Try adding a vertex with various neighbor sets
        # Try neighborhoods of existing vertices, complements, and random sets
        neighbor_candidates = []
        
        # Add neighborhoods of existing vertices
        for v in range(current_n):
            neighbor_candidates.append(current_adj[v])
        
        # Add complements of neighborhoods
        all_mask = (1 << current_n) - 1
        for v in range(current_n):
            neighbor_candidates.append(all_mask & ~current_adj[v] & ~(1 << v))
        
        # Add random neighbor sets
        for _ in range(50):
            size = random.randint(current_n // 3, 2 * current_n // 3)
            neighbors = random.sample(range(current_n), size)
            mask = 0
            for v in neighbors:
                mask |= (1 << v)
            neighbor_candidates.append(mask)
        
        random.shuffle(neighbor_candidates)
        
        for mask in neighbor_candidates:
            if time.time() > deadline:
                break
            if found:
                break
                
            result = try_add_vertex(current_adj, current_n, mask)
            if result is not None:
                found = True
                current_n, current_adj = result
                best_n = current_n
                best_adj = current_adj[:]
                break
        
        # Strategy 2: If we couldn't add a vertex, try edge flipping
        if not found:
            # Try flipping some random edges
            for _ in range(100):
                if time.time() > deadline:
                    break
                    
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j:
                    continue
                    
                new_adj = try_flip_edge(current_adj, current_n, i, j)
                if new_adj is not None and new_adj != current_adj:
                    # Accept the flip
                    current_adj = new_adj
                    # Try to add a vertex again
                    for mask in neighbor_candidates[:20]:
                        if time.time() > deadline:
                            break
                        result = try_add_vertex(current_adj, current_n, mask)
                        if result is not None:
                            found = True
                            current_n, current_adj = result
                            best_n = current_n
                            best_adj = current_adj[:]
                            break
                    if found:
                        break
        
        if not found:
            # If we're stuck, try a few more random restarts
            for restart in range(3):
                if time.time() > deadline:
                    break
                    
                # Reset to best state
                current_n = best_n
                current_adj = best_adj[:]
                
                # Try random edge flips and vertex additions
                for _ in range(50):
                    if time.time() > deadline:
                        break
                        
                    # Random edge flip
                    i = random.randint(0, current_n - 1)
                    j = random.randint(0, current_n - 1)
                    if i == j:
                        continue
                        
                    new_adj = try_flip_edge(current_adj, current_n, i, j)
                    if new_adj is not None:
                        current_adj = new_adj
                        
                        # Try to add a vertex
                        size = random.randint(current_n // 3, 2 * current_n // 3)
                        neighbors = random.sample(range(current_n), size)
                        mask = 0
                        for v in neighbors:
                            mask |= (1 << v)
                        
                        result = try_add_vertex(current_adj, current_n, mask)
                        if result is not None:
                            current_n, current_adj = result
                            best_n = current_n
                            best_adj = current_adj[:]
                            break
        
        if time.time() > deadline:
            break

    # Reconstruct edges from best_adj
    final_edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
# EVOLVE-BLOCK-END
