import time
import random

def construct():
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper: Check if graph is 3-colorable using bitmask adjacency
    # neighbors is a list of ints, neighbors[i] has bit j set if (i,j) is an edge
    def is_3_colorable(neighbors):
        colors = [-1] * n
        # Order by degree
        deg = [bin(neighbors[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = [False, False, False]
            nb = neighbors[v]
            # Iterate over neighbors of v
            w = 0
            while nb:
                if nb & 1:
                    if colors[w] != -1:
                        forbidden[colors[w]] = True
                nb >>= 1
                w += 1
            
            for c in range(3):
                if not forbidden[c]:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Initialize with seed graph
    # Seed edges:
    # Cycle 0-1-2-3-4-0
    # 0-1, 1-2, 2-3, 3-4, 4-0
    # V' (5-9) connected to i+1, i-1
    # 0-6, 0-9 (5+1=6, 5-1=4->9? No, 5+(0-1)%5 = 5+4=9)
    # 1-7, 1-5
    # 2-8, 2-6
    # 3-9, 3-7
    # 4-5, 4-8
    # u=10 connected to 5,6,7,8,9
    # 11-15 connected to 5,6,7,8,9
    
    seed_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 6), (0, 9),
        (1, 7), (1, 5),
        (2, 8), (2, 6),
        (3, 9), (3, 7),
        (4, 5), (4, 8),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9),
        (12, 5), (12, 6), (12, 7), (12, 8), (12, 9),
        (13, 5), (13, 6), (13, 7), (13, 8), (13, 9),
        (14, 5), (14, 6), (14, 7), (14, 8), (14, 9),
        (15, 5), (15, 6), (15, 7), (15, 8), (15, 9)
    ]
    
    neighbors = [0] * n
    for u, v in seed_edges:
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)
        
    best_neighbors = neighbors[:]
    best_count = len(seed_edges)
    
    start_time = time.time()
    deadline = start_time + 145
    
    def try_add_edges(neighbors, max_time):
        neighbors = neighbors[:]
        # Get non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (neighbors[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > max_time:
                break
            # Check triangle: no common neighbor
            if neighbors[u] & neighbors[v]:
                continue
            # Check 3-colorability
            # Temporarily add
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            
            if is_3_colorable(neighbors):
                # Revert
                neighbors[u] ^= (1 << v)
                neighbors[v] ^= (1 << u)
                continue
            else:
                # Keep
                pass
                
        return neighbors

    current_neighbors = best_neighbors[:]
    
    while time.time() < deadline:
        # Try to add edges
        new_neighbors = try_add_edges(current_neighbors, deadline)
        new_count = sum(bin(neighbors[i]).count('1') for i in range(n)) // 2
        
        if new_count > len(current_neighbors): # Wait, len is not count of edges
            # Calculate count properly
            pass
            
        # Calculate count of edges in new_neighbors
        new_count = sum(bin(neighbors[i]).count('1') for i in range(n)) // 2
        cur_count = sum(bin(current_neighbors[i]).count('1') for i in range(n)) // 2
        
        if new_count > cur_count:
            current_neighbors = new_neighbors
            if new_count > best_count:
                best_count = new_count
                best_neighbors = current_neighbors[:]
        
        # If no improvement, try a kick
        # Remove a random edge
        if cur_count > 0:
            # Find an edge to remove
            edge_found = False
            for _ in range(100): # Try a few times to find an edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and (current_neighbors[u] & (1 << v)):
                    # Remove
                    current_neighbors[u] ^= (1 << v)
                    current_neighbors[v] ^= (1 << u)
                    edge_found = True
                    break
            
            if edge_found:
                # Try to add 2 edges
                new_neighbors = current_neighbors[:]
                # Use try_add_edges but limit to adding 2?
                # Just call try_add_edges which will add as many as possible
                # But we removed 1, so if it adds 2, we gain 1.
                # If it adds 1, we gain 0.
                # If it adds 0, we lose 1.
                
                # To be safe, we can check if the result is better
                filled_neighbors = try_add_edges(new_neighbors, deadline)
                filled_count = sum(bin(filled_neighbors[i]).count('1') for i in range(n)) // 2
                
                if filled_count > cur_count:
                    current_neighbors = filled_neighbors
                    if filled_count > best_count:
                        best_count = filled_count
                        best_neighbors = current_neighbors[:]
                # If filled_count <= cur_count, we discard the kick (keep current_neighbors as it was before kick? 
                # No, we modified current_neighbors in place for the kick. 
                # We should have copied it.
                # Let's fix the logic: 
                # We modified current_neighbors. If the kick fails, we are worse off.
                # We should restore current_neighbors if the kick doesn't improve it.
                # But we don't have the original current_neighbors saved easily if we modified it.
                # Let's save it before the kick.
                
                # This logic is getting messy. 
                # Simpler: 
                # 1. Save current state.
                # 2. Perform kick.
                # 3. If better, accept. Else, revert.
                
    # Re-implement loop with proper state management
    
    # Reset current_neighbors to best
    current_neighbors = best_neighbors[:]
    
    while time.time() < deadline:
        # 1. Greedy Add
        new_neighbors = try_add_edges(current_neighbors, deadline)
        new_count = sum(bin(new_neighbors[i]).count('1') for i in range(n)) // 2
        cur_count = sum(bin(current_neighbors[i]).count('1') for i in range(n)) // 2
        
        if new_count > cur_count:
            current_neighbors = new_neighbors
            if new_count > best_count:
                best_count = new_count
                best_neighbors = current_neighbors[:]
        
        # 2. Kick
        # Save state
        saved_neighbors = current_neighbors[:]
        saved_count = cur_count
        
        # Remove random edge
        edge_removed = False
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (current_neighbors[u] & (1 << v)):
                current_neighbors[u] ^= (1 << v)
                current_neighbors[v] ^= (1 << u)
                edge_removed = True
                break
        
        if edge_removed:
            # Try to fill
            new_neighbors = try_add_edges(current_neighbors, deadline)
            new_count = sum(bin(new_neighbors[i]).count('1') for i in range(n)) // 2
            
            if new_count > saved_count:
                current_neighbors = new_neighbors
                if new_count > best_count:
                    best_count = new_count
                    best_neighbors = current_neighbors[:]
            else:
                # Revert
                current_neighbors = saved_neighbors

    # Convert best_neighbors to list of edges
    edges = []
    for i in range(n):
        nb = best_neighbors[i]
        j = i + 1
        while nb:
            if nb & 1:
                edges.append([i, j])
            nb >>= 1
            j += 1
            
    return edges
