import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_edges)
    
    # Fast C6 check using bitmasks
    # adj_mask[v] is a 40-bit integer with bit i set if edge (v,i) exists
    def would_create_c6_fast(u, v, adj_mask):
        """Check if adding edge (u,v) creates a C6.
        Need a simple path u-a-b-c-d-v of length 5.
        Equivalently: is there a path of length 4 from u to some neighbor of v (excluding u and v)?
        """
        # We need: exists a,b,c,d such that:
        # u~a, a~b, b~c, c~d, d~v, all distinct, a!=v, d!=u
        # Path: u -> a -> b -> c -> d -> v (6 vertices, 5 edges + new edge = C6)
        
        # BFS/DFS from u up to depth 4, tracking visited
        # If we reach a node d at depth 4 that is adjacent to v (and d != u, d != v), we found C6
        
        # Use iterative DFS with bitmask for visited
        # Start: at u, depth 0
        # After 4 more steps (depth 4), check if current node is neighbor of v
        
        # neighbors of u (excluding v)
        start_nodes = []
        mask_u = adj_mask[u]
        # Remove v from consideration
        if v < 40:
            mask_u &= ~(1 << v)
        
        # Depth 0: we are at u
        # Depth 1: neighbors of u (excluding v and u)
        # Depth 2: neighbors of depth-1 nodes (excluding u, v, and path nodes)
        # Depth 3: neighbors of depth-2 nodes
        # Depth 4: check if any is neighbor of v
        
        # Use sets for simplicity but with early termination
        # Since n=40 and depth=4, worst case is manageable
        
        # Let's do it level by level
        # Level 1: neighbors of u (not v)
        level1 = set()
        m = mask_u
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            level1.add(idx)
            m ^= lb
        
        if not level1:
            return False
        
        # Level 2: for each node in level1, get its neighbors excluding u, v, and other level1 nodes
        level2 = set()
        for a in level1:
            mask_a = adj_mask[a]
            # Exclude u, v
            mask_a &= ~(1 << u)
            mask_a &= ~(1 << v)
            # Exclude other level1 nodes
            for b in level1:
                if b != a:
                    mask_a &= ~(1 << b)
            m = mask_a
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                level2.add(idx)
                m ^= lb
        
        if not level2:
            return False
        
        # Level 3: for each node in level2, get neighbors excluding u, v, level1, and other level2 nodes
        level3 = set()
        for b in level2:
            mask_b = adj_mask[b]
            mask_b &= ~(1 << u)
            mask_b &= ~(1 << v)
            for a in level1:
                mask_b &= ~(1 << a)
            for c in level2:
                if c != b:
                    mask_b &= ~(1 << c)
            m = mask_b
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                level3.add(idx)
                m ^= lb
        
        if not level3:
            return False
        
        # Level 4: for each node in level3, get neighbors excluding u, v, level1, level2, and other level3 nodes
        # Then check if any such neighbor is adjacent to v
        mask_v = adj_mask[v]
        mask_v &= ~(1 << u)  # d cannot be u
        
        for c in level3:
            mask_c = adj_mask[c]
            mask_c &= ~(1 << u)
            mask_c &= ~(1 << v)
            for a in level1:
                mask_c &= ~(1 << a)
            for b in level2:
                mask_c &= ~(1 << b)
            for d in level3:
                if d != c:
                    mask_c &= ~(1 << d)
            # Now check if any node in mask_c is also in mask_v
            if (mask_c & mask_v) != 0:
                return True
        
        return False
    
    # Greedy construction to get initial solution
    def greedy_construct(seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        edges = []
        
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        
        for u, v in shuffled:
            if would_create_c6_fast(u, v, adj_mask):
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append((u, v))
        
        return adj_mask, edges
    
    # Local search: try to add edges or swap edges
    def local_search(adj_mask, edges, rng, time_limit):
        current_edges = edges[:]
        current_mask = adj_mask[:]
        current_count = len(current_edges)
        best_count = current_count
        best_edges = current_edges[:]
        best_mask = current_mask[:]
        
        edge_to_pos = {e: i for i, e in enumerate(current_edges)}
        
        # Build adjacency list from mask for quick access
        def get_neighbors(mask_arr, node):
            m = mask_arr[node]
            neighbors = []
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                neighbors.append(idx)
                m ^= lb
            return neighbors
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding a missing edge
            missing = [e for e in all_edges if e not in edge_to_pos]
            rng.shuffle(missing)
            
            for u, v in missing:
                if time.time() >= time_limit:
                    break
                if would_create_c6_fast(u, v, current_mask):
                    continue
                # Add edge
                current_mask[u] |= (1 << v)
                current_mask[v] |= (1 << u)
                current_edges.append((u, v))
                edge_to_pos[(u, v)] = len(current_edges) - 1
                current_count += 1
                improved = True
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges = current_edges[:]
                    best_mask = current_mask[:]
            
            if not improved:
                # Try edge swaps: remove one edge, add another
                # Sample a few random swaps
                for _ in range(50):
                    if time.time() >= time_limit:
                        break
                    if not current_edges:
                        break
                    # Pick random edge to remove
                    rem_idx = rng.randint(0, len(current_edges) - 1)
                    rem_u, rem_v = current_edges[rem_idx]
                    
                    # Try to add a random missing edge after removal
                    missing = [e for e in all_edges if e not in edge_to_pos and e != (rem_u, rem_v)]
                    if not missing:
                        break
                    rng.shuffle(missing)
                    
                    found_swap = False
                    for add_u, add_v in missing[:20]:  # Try up to 20 candidates
                        if time.time() >= time_limit:
                            break
                        # Temporarily remove rem edge
                        current_mask[rem_u] &= ~(1 << rem_v)
                        current_mask[rem_v] &= ~(1 << rem_u)
                        
                        if not would_create_c6_fast(add_u, add_v, current_mask):
                            # Add the new edge
                            current_mask[add_u] |= (1 << add_v)
                            current_mask[add_v] |= (1 << add_u)
                            
                            # Update edge list
                            del current_edges[rem_idx]
                            del edge_to_pos[(rem_u, rem_v)]
                            # Rebuild edge_to_pos for remaining edges
                            edge_to_pos = {e: i for i, e in enumerate(current_edges)}
                            current_edges.append((add_u, add_v))
                            edge_to_pos[(add_u, add_v)] = len(current_edges) - 1
                            found_swap = True
                            break
                        
                        # Restore rem edge if not swapped
                        current_mask[rem_u] |= (1 << rem_v)
                        current_mask[rem_v] |= (1 << rem_u)
                    
                    if found_swap:
                        improved = True
                        # Count stays the same, but structure changed
                        # Try to add more edges after swap
                        missing_after = [e for e in all_edges if e not in edge_to_pos]
                        rng.shuffle(missing_after)
                        for add_u, add_v in missing_after[:30]:
                            if time.time() >= time_limit:
                                break
                            if would_create_c6_fast(add_u, add_v, current_mask):
                                continue
                            current_mask[add_u] |= (1 << add_v)
                            current_mask[add_v] |= (1 << add_u)
                            current_edges.append((add_u, add_v))
                            edge_to_pos[(add_u, add_v)] = len(current_edges) - 1
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_edges = current_edges[:]
                                best_mask = current_mask[:]
            
            if not improved:
                break
        
        return best_mask, best_edges, best_count
    
    # Main loop: multiple greedy + local search iterations
    best_count = 0
    best_edges = []
    
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        seed = seed_counter * 12345 + 67890
        
        # Get initial greedy solution
        adj_mask, edges = greedy_construct(seed)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        # Do local search if we have time
        remaining = deadline - time.time()
        if remaining > 5.0:
            ls_limit = min(remaining - 3.0, 30.0)  # Spend up to 30s on local search per iteration
            rng = random.Random(seed + 1)
            best_mask, best_edges_ls, best_count_ls = local_search(adj_mask, edges, rng, ls_limit)
            
            if best_count_ls > best_count:
                best_count = best_count_ls
                best_edges = best_edges_ls[:]
        
        # Early stop if we hit known optimum
        if best_count >= 105:
            break
    
    return [[u, v] for u, v in best_edges]
