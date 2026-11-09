import random
import time

def construct():
    n = 20
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v, masks):
        """
        Check if adding edge (u, v) creates a K3,3.
        K3,3 partitions: {u, x1, x2} and {v, y1, y2}
        Conditions:
        - x1, x2 in N(v)
        - y1, y2 in N(u)
        - Edges (x1, y1), (x1, y2), (x2, y1), (x2, y2) exist.
        
        This is equivalent to:
        There exist x1, x2 in N(v) such that |(N(x1) intersect N(x2) intersect N(u))| >= 2.
        """
        if u == v:
            return False
        if masks[u] & (1 << v):
            return False
            
        N_v = masks[v]
        N_u = masks[u]
        
        # Extract neighbors of v
        neighbors_v = []
        temp = N_v
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            if idx != v:
                neighbors_v.append(idx)
            temp ^= lsb
            
        if len(neighbors_v) < 2:
            return True
            
        # Check all pairs in neighbors_v
        for i in range(len(neighbors_v)):
            x1 = neighbors_v[i]
            for j in range(i + 1, len(neighbors_v)):
                x2 = neighbors_v[j]
                # Common neighbors of x1 and x2 that are also neighbors of u
                common = masks[x1] & masks[x2] & N_u
                if popcount(common) >= 2:
                    return False
        return True

    def add_edge(u, v, masks):
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)

    def remove_edge(u, v, masks):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)

    def count_edges(masks):
        total = 0
        for i in range(n):
            total += popcount(masks[i])
        return total // 2

    def get_missing_edges(masks):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (masks[i] & (1 << j)):
                    missing.append((i, j))
        return missing

    def get_existing_edges(masks):
        existing = []
        for i in range(n):
            for j in range(i + 1, n):
                if masks[i] & (1 << j):
                    existing.append((i, j))
        return existing

    def greedy_build():
        masks = [0] * n
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if can_add(u, v, masks):
                add_edge(u, v, masks)
        return masks

    def local_search(masks, time_limit):
        start_time = time.time()
        current_edges = count_edges(masks)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any valid edge
            missing = get_missing_edges(masks)
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, masks):
                    add_edge(u, v, masks)
                    current_edges += 1
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try 1-for-2 swaps: Remove (u,v), add two new edges
            existing = get_existing_edges(masks)
            if not existing:
                break
            random.shuffle(existing)
            
            found_swap = False
            for u, v in existing:
                if time.time() - start_time > time_limit:
                    break
                
                # Remove (u,v)
                remove_edge(u, v, masks)
                
                # Try to add 2 edges
                added = []
                missing_now = get_missing_edges(masks)
                random.shuffle(missing_now)
                
                for x, y in missing_now:
                    if len(added) >= 2:
                        break
                    if can_add(x, y, masks):
                        add_edge(x, y, masks)
                        added.append((x, y))
                        
                if len(added) == 2:
                    current_edges += 1
                    improved = True
                    found_swap = True
                    break
                else:
                    # Undo
                    for x, y in added:
                        remove_edge(x, y, masks)
                    add_edge(u, v, masks)
                    
            if found_swap:
                continue
                
            # 3. Try 2-for-3 swaps: Remove 2 edges, add 3 new edges
            if len(existing) >= 2:
                found_swap2 = False
                # Sample a few pairs of existing edges to remove
                for _ in range(min(20, len(existing) // 2)):
                    if time.time() - start_time > time_limit:
                        break
                    
                    e1, e2 = random.sample(existing, 2)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    remove_edge(u1, v1, masks)
                    remove_edge(u2, v2, masks)
                    
                    added = []
                    missing_now = get_missing_edges(masks)
                    random.shuffle(missing_now)
                    
                    for x, y in missing_now:
                        if len(added) >= 3:
                            break
                        if can_add(x, y, masks):
                            add_edge(x, y, masks)
                            added.append((x, y))
                            
                    if len(added) == 3:
                        current_edges += 1
                        improved = True
                        found_swap2 = True
                        break
                    else:
                        # Undo
                        for x, y in added:
                            remove_edge(x, y, masks)
                        add_edge(u1, v1, masks)
                        add_edge(u2, v2, masks)
                        
                if found_swap2:
                    continue
                    
            # 4. Kick: Remove 1-3 random edges to escape local optimum
            if random.random() < 0.5:
                kick_size = random.randint(1, 3)
                if len(existing) >= kick_size:
                    edges_to_remove = random.sample(existing, kick_size)
                    for u, v in edges_to_remove:
                        remove_edge(u, v, masks)
                    current_edges -= kick_size
                    improved = True # Force re-evaluation

        return masks, current_edges

    # Main Loop
    best_mask = [0] * n
    best_edge_count = 0
    
    while time.time() < deadline:
        # Greedy build
        mask = greedy_build()
        edges = count_edges(mask)
        
        if edges > best_edge_count:
            best_edge_count = edges
            best_mask = mask[:]
            
        # Apply local search if we have time
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            local_time = min(3.0, remaining_time * 0.3)
            mask, edges = local_search(mask, local_time)
            
            if edges > best_edge_count:
                best_edge_count = edges
                best_mask = mask[:]
                
        # If we've found a very good solution, reduce search effort
        if best_edge_count >= 85:
            break

    # Final polish: try to add any remaining edges
    if time.time() < deadline - 2:
        for u, v in all_pairs:
            if not (best_mask[u] & (1 << v)):
                if can_add(u, v, best_mask):
                    add_edge(u, v, best_mask)
    
    # Convert best_mask to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_mask[i] & (1 << j):
                result.append([i, j])
    return result
