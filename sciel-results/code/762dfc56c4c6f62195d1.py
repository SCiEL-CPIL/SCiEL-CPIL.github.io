import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def has_independent_set_10(adj, n):
        """Fast check: does graph have independent set of size 10?"""
        mask_all = (1 << n) - 1
        
        # Quick greedy lower bound
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if best_v == -1:
                break
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
        if size >= 10:
            return True

        # Branch and bound with strong pruning
        def bb(candidates, size):
            if size >= 10:
                return True
            if candidates == 0:
                return False
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Upper bound: try greedy on candidates to get a tighter bound
            # If even greedy can't reach 10-size, prune
            gc = candidates
            gs = 0
            while gc and gs < 10 - size:
                bv = -1
                md = n + 1
                t = gc
                while t:
                    v = (t & -t).bit_length() - 1
                    t &= t - 1
                    d = (adj[v] & gc).bit_count()
                    if d < md:
                        md = d
                        bv = v
                if bv == -1:
                    break
                gs += 1
                gc &= ~adj[bv] & ~(1 << bv)
            if size + gs < 10:
                return False

            # Pick vertex with max degree in candidates
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch: include best_v (remove its neighbors) or exclude it
            if bb(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            if bb(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return bb(mask_all, 0)

    def is_triangle_free(adj, n):
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def make_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    def mycielski(adj, n):
        new_n = 2 * n + 1
        new_adj = [0] * new_n
        for i in range(n):
            new_adj[i] = adj[i]
        for i in range(n):
            for j in range(n):
                if adj[i] & (1 << j):
                    new_adj[n + i] |= (1 << j)
                    new_adj[j] |= (1 << (n + i))
        for i in range(n):
            new_adj[2 * n] |= (1 << (n + i))
            new_adj[n + i] |= (1 << (2 * n))
        return new_adj, new_n

    def blow_up(adj, n, sizes):
        new_n = sum(sizes)
        new_adj = [0] * new_n
        offsets = [0] * (n + 1)
        for i in range(n):
            offsets[i + 1] = offsets[i] + sizes[i]
        for i in range(n):
            for j in range(n):
                if i != j and (adj[i] & (1 << j)):
                    for a in range(offsets[i], offsets[i + 1]):
                        for b in range(offsets[j], offsets[j + 1]):
                            new_adj[a] |= (1 << b)
                            new_adj[b] |= (1 << a)
        return new_adj, new_n

    def extend_andrasfai_structural(base_adj, base_n, target_n, num_trials=500):
        """Extend Andrásfai A9 to larger n using its 3-partite structure."""
        if base_n >= target_n:
            return None
        
        k = base_n // 3  # = 9 for A9
        # A9 structure: 3 parts of size k, each part is independent,
        # edges between parts form a cycle-like structure
        
        for trial in range(num_trials):
            adj = base_adj[:] + [0] * (target_n - base_n)
            new_count = target_n - base_n
            
            # Distribute new vertices among the 3 parts
            # Part i in A9: vertices {i, i+k, i+2k} for i in 0..k-1
            # New vertices will be added to extend parts
            
            # Strategy: add new vertices with adjacency based on their "position"
            # in the cyclic structure
            for idx in range(new_count):
                v_new = base_n + idx
                # Assign to a part based on trial pattern
                part = (idx + trial) % 3
                
                # Determine which existing vertices to connect to
                # In A9, vertex v in part p connects to specific vertices in other parts
                # For extension, connect to vertices that maintain triangle-freeness
                
                candidates = []
                for v_old in range(base_n):
                    old_part = v_old % 3  # rough part assignment
                    # Only connect to vertices in different parts to help triangle-free
                    if old_part != part:
                        # Check triangle constraint
                        common = 0
                        temp = adj[v_new]
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            temp &= temp - 1
                            if adj[v_old] & (1 << w):
                                common = 1
                                break
                        if common == 0:
                            candidates.append(v_old)
                
                if candidates:
                    # Connect to a random subset
                    num_connect = random.randint(max(1, len(candidates) // 4), 
                                                  min(len(candidates), max(2, len(candidates) // 2)))
                    selected = random.sample(candidates, num_connect)
                    for v_old in selected:
                        adj[v_new] |= (1 << v_old)
                        adj[v_old] |= (1 << v_new)
            
            # Quick triangle-free check
            if not is_triangle_free(adj, target_n):
                continue
            
            # Check independence number
            if not has_independent_set_10(adj, target_n):
                return adj
        
        return None

    def local_search(adj, n, max_iters=500):
        """Local search: try edge flips to reduce independence number while maintaining triangle-free."""
        adj = adj[:]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding a random non-edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or adj[i] & (1 << j):
                continue
            
            # Check triangle-free
            common = adj[i] & adj[j]
            if common != 0:
                continue
            
            # Try the edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            if not has_independent_set_10(adj, n):
                return adj
            
            # Revert
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        
        return adj

    # Try known constructions first
    best_result = None
    
    # Andrásfai And(9): n=27, alpha=9
    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
    
    # Mycielski constructions
    adj_c5 = [0] * 5
    for i in range(5):
        adj_c5[i] |= (1 << ((i + 1) % 5))
        adj_c5[i] |= (1 << ((i + 4) % 5))
    
    adj_grotzsch, n_grotzsch = mycielski(adj_c5, 5)
    if n_grotzsch <= 41 and not has_independent_set_10(adj_grotzsch, n_grotzsch):
        if best_result is None or n_grotzsch > best_result['n']:
            best_result = {'n': n_grotzsch, 'edges': build_edge_list(adj_grotzsch, n_grotzsch)}
    
    adj_m2, n_m2 = mycielski(adj_grotzsch, n_grotzsch)
    if n_m2 <= 41 and not has_independent_set_10(adj_m2, n_m2):
        if best_result is None or n_m2 > best_result['n']:
            best_result = {'n': n_m2, 'edges': build_edge_list(adj_m2, n_m2)}
    
    # Try structural extension of A9
    if time.time() < deadline:
        for target_n in range(40, max(28, (best_result['n'] + 1) if best_result else 28) - 1, -1):
            if time.time() > deadline:
                break
            
            adj_ext = extend_andrasfai_structural(adj9, n9, target_n, num_trials=300)
            if adj_ext is not None:
                best_result = {'n': target_n, 'edges': build_edge_list(adj_ext, target_n)}
                break
    
    # Try blowing up small base graphs with optimized sizes
    if time.time() < deadline:
        min_target = (best_result['n'] + 1) if best_result else 28
        
        for target_n in range(40, min_target - 1, -1):
            if time.time() > deadline:
                break
            
            found = False
            attempts = 0
            max_attempts = 2000
            
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                
                strategy = random.randint(0, 3)
                
                if strategy == 0:
                    # Blow up C5
                    sizes = [1] * 5
                    remaining = target_n - 5
                    for _ in range(remaining):
                        sizes[random.randint(0, 4)] += 1
                    adj, _ = blow_up(adj_c5, 5, sizes)
                elif strategy == 1:
                    # Blow up Grötzsch
                    sizes = [1] * n_grotzsch
                    remaining = target_n - n_grotzsch
                    for _ in range(remaining):
                        sizes[random.randint(0, n_grotzsch - 1)] += 1
                    adj, _ = blow_up(adj_grotzsch, n_grotzsch, sizes)
                elif strategy == 2:
                    # Random triangle-free
                    adj = [0] * target_n
                    edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
                    random.shuffle(edges)
                    for u, v in edges:
                        common = adj[u] & adj[v]
                        if common == 0:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                else:
                    # Blow up A9 with some expansion
                    if target_n >= n9:
                        sizes = [1] * n9
                        remaining = target_n - n9
                        for _ in range(remaining):
                            sizes[random.randint(0, n9 - 1)] += 1
                        adj, _ = blow_up(adj9, n9, sizes)
                    else:
                        continue
                
                if not has_independent_set_10(adj, target_n):
                    found = True
                    # Try local search to improve
                    if time.time() < deadline:
                        adj_ls = local_search(adj, target_n, max_iters=200)
                        if not has_independent_set_10(adj_ls, target_n):
                            adj = adj_ls
                    best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                    break
            
            if found:
                break
    
    if best_result is None:
        adj9, n9 = make_andrasfai(9)
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
    
    return best_result
