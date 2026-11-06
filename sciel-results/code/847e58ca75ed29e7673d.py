"""Combined program: greedy C4-free construction with local search using bitmasks."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_c4_free(adj, n):
        """Check C4-freeness: no two vertices share 2+ common neighbors."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common = ai & adj[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def is_c4_free_after_add(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4. Fast check."""
        common = adj[u] & adj[v]
        return common.bit_count() < 2

    def is_c4_free_after_swap(adj, n, u, x, y):
        """Check if swapping (u,x)->(u,y) maintains C4-freeness.
        After removing (u,x) and adding (u,y), we need to verify:
        1. No new C4 involving (u,y)
        2. Removing (u,x) doesn't break anything (it can only help)
        So we just need to check that (u,y) doesn't create a C4 in the
        graph with (u,x) removed."""
        # Temporarily remove (u,x)
        adj[u] &= ~(1 << x)
        adj[x] &= ~(1 << u)
        
        # Check if (u,y) would create C4
        ok = is_c4_free_after_add(adj, n, u, y)
        
        # Restore (u,x)
        adj[u] |= (1 << x)
        adj[x] |= (1 << u)
        
        return ok

    def min_degree(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def repair_c4s(adj, n, max_iter=200):
        """Repair C4s by removing edges. Returns True if C4-free."""
        for _ in range(max_iter):
            if elapsed() > deadline - 2:
                break
            found = False
            for i in range(n):
                if elapsed() > deadline - 2:
                    break
                ai = adj[i]
                for j in range(i + 1, n):
                    common = ai & adj[j]
                    if common.bit_count() >= 2:
                        # Find which common neighbors cause the C4
                        # Remove edge from i to one of the common neighbors
                        tmp = common
                        lsb = tmp & -tmp
                        w = lsb.bit_length() - 1
                        adj[i] &= ~(1 << w)
                        adj[w] &= ~(1 << i)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return check_c4_free(adj, n)

    def add_edges_to_boost_min_degree(adj, n, target, max_rounds=50):
        """Try to add edges to boost minimum degree while maintaining C4-freeness."""
        for _ in range(max_rounds):
            if elapsed() > deadline - 2:
                break
            
            degs = [adj[i].bit_count() for i in range(n)]
            min_deg = min(degs)
            if min_deg >= target:
                return True
            
            # Find vertices below target
            low_verts = [i for i in range(n) if degs[i] < target]
            if not low_verts:
                return True
            
            # Try to add edges for low-degree vertices
            improved = False
            random.shuffle(low_verts)
            
            for u in low_verts:
                if degs[u] >= target:
                    continue
                if elapsed() > deadline - 2:
                    break
                
                mask = (1 << n) - 1
                non_neighbors = (~adj[u]) & mask
                non_neighbors &= ~(1 << u)
                
                # Get list of non-neighbors
                nn_list = []
                tmp = non_neighbors
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    nn_list.append(idx)
                    tmp ^= lsb
                
                random.shuffle(nn_list)
                
                for v in nn_list:
                    if degs[u] >= target:
                        break
                    # Check if adding (u,v) would create a C4
                    if is_c4_free_after_add(adj, n, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degs[u] += 1
                        degs[v] += 1
                        improved = True
                        if degs[u] >= target:
                            break
            
            if not improved:
                break
        
        return min_degree(adj, n) >= target

    def local_search(adj, n, target, max_iters=200):
        """Local search: swap edges to improve minimum degree."""
        degs = [adj[i].bit_count() for i in range(n)]
        
        for _ in range(max_iters):
            if elapsed() > deadline - 2:
                break
            
            min_deg = min(degs)
            if min_deg >= target:
                return True
            
            # Find a low-degree vertex
            low_verts = [i for i in range(n) if degs[i] <= min_deg + 1]
            random.shuffle(low_verts)
            
            for u in low_verts:
                if degs[u] >= target:
                    continue
                
                # Try to swap: remove edge (u, x) and add edge (u, y)
                # where x is a neighbor and y is a non-neighbor
                neighbors = []
                tmp = adj[u]
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    neighbors.append(idx)
                    tmp ^= lsb
                
                mask = (1 << n) - 1
                non_neighbors = (~adj[u]) & mask
                non_neighbors &= ~(1 << u)
                
                nn_list = []
                tmp = non_neighbors
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    nn_list.append(idx)
                    tmp ^= lsb
                
                random.shuffle(neighbors)
                random.shuffle(nn_list)
                
                for x in neighbors:
                    # Check if removing (u,x) and adding some (u,y) helps
                    # First, check which non-neighbors y would be valid
                    valid_ys = []
                    for y in nn_list:
                        if is_c4_free_after_swap(adj, n, u, x, y):
                            valid_ys.append(y)
                    
                    if not valid_ys:
                        continue
                    
                    random.shuffle(valid_ys)
                    
                    for y in valid_ys:
                        # Perform the swap
                        adj[u] &= ~(1 << x)
                        adj[x] &= ~(1 << u)
                        degs[u] -= 1
                        degs[x] -= 1
                        
                        adj[u] |= (1 << y)
                        adj[y] |= (1 << u)
                        degs[u] += 1
                        degs[y] += 1
                        
                        # Check if this improved min degree
                        new_min = min(degs)
                        if new_min >= target:
                            return True
                        
                        # Keep the swap if it doesn't hurt min degree
                        if new_min >= min_deg:
                            min_deg = new_min
                            break
                        else:
                            # Revert
                            adj[u] &= ~(1 << y)
                            adj[y] &= ~(1 << u)
                            degs[u] -= 1
                            degs[y] -= 1
                            adj[u] |= (1 << x)
                            adj[x] |= (1 << u)
                            degs[u] += 1
                            degs[x] += 1
                            break
                    break
        
        return min_degree(adj, n) >= target

    def try_construct(n, attempts=50):
        target = n - 39
        if target < 1:
            target = 1
        
        # Start with a slightly denser graph
        p = (target + 1) / n
        if p > 0.5: p = 0.5
        
        best_adj = None
        best_md = -1

        for attempt in range(attempts):
            if elapsed() > deadline - 2:
                break

            # 1. Initialize random graph using bitmasks
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # 2. Repair: Remove edges to break C4s
            if not repair_c4s(adj, n, max_iter=200):
                continue

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj[:]

            # 3. If min degree is too low, try to add edges
            if md < target:
                if add_edges_to_boost_min_degree(adj, n, target, max_rounds=50):
                    if check_c4_free(adj, n):
                        md = min_degree(adj, n)
                        if md > best_md:
                            best_md = md
                            best_adj = adj[:]
                        if md >= target:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return n, edges

            # 4. Local search to improve
            if md < target:
                if local_search(adj, n, target, max_iters=150):
                    if check_c4_free(adj, n):
                        md = min_degree(adj, n)
                        if md > best_md:
                            best_md = md
                            best_adj = adj[:]
                        if md >= target:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return n, edges

            # Final check
            if check_c4_free(adj, n):
                md = min_degree(adj, n)
                if md > best_md:
                    best_md = md
                    best_adj = adj[:]
                if md >= target:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return n, edges

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return n, edges
            
        return None

    # Try from largest n down, with time budget allocation
    # Give more time to larger n values
    time_per_n = {47: 40, 46: 40, 45: 30, 44: 20, 43: 10, 42: 5, 41: 3, 40: 2}
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if elapsed() > deadline - 5:
            break
        # Adjust deadline for this n
        old_deadline = deadline
        deadline = min(deadline, start_time + time_per_n.get(n, 10))
        
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}
        
        deadline = old_deadline

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
