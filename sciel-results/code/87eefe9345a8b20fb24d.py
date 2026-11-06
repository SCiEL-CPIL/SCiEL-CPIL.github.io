import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_k4(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    deg_sum = 0
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        deg_sum += (adj[x] & common).bit_count()
                    count += deg_sum // 2
        return count

    def count_i6(n, adj):
        # Use backtracking to find if any I6 exists, and count them
        # For counting, use the complement: I6 in G = K6 in ~G
        # But for delta, we need exact counts. Let's use a smarter approach.
        # Actually, for the SA cost, we can use a proxy: count of "near-I6" sets
        # But for final validation, we need exact.
        # For the cost function in SA, let's count I4 in the complement of each pair's neighborhood
        # This is expensive. Instead, use a simpler proxy:
        # Cost = count_k4 + alpha * count_i6_proxy
        # where count_i6_proxy is based on independent set counting in complement
        
        # For efficiency, count triangles in complement (I3 in G) as a proxy for I6
        # This is much faster and correlated with I6 presence
        mask = (1 << n) - 1
        count = 0
        for i in range(n):
            non_neigh_i = (~adj[i]) & mask
            non_neigh_i &= ~(1 << i)
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    non_neigh_j = (~adj[j]) & mask
                    non_neigh_j &= ~(1 << j)
                    # Common non-neighbors k > j
                    mask_k = mask ^ ((1 << (j + 1)) - 1)
                    count += (non_neigh_i & non_neigh_j & mask_k).bit_count()
        return count

    def has_k6_in_complement(adj, n):
        """Check if there's an independent set of size 6 in G (K6 in complement)"""
        def bt(candidates, depth):
            if depth == 6:
                return True
            if len(candidates) < depth:
                return False
            for idx in range(len(candidates)):
                v = candidates[idx]
                new_cands = []
                for u in candidates[idx + 1:]:
                    if not ((adj[v] >> u) & 1):
                        new_cands.append(u)
                if bt(new_cands, depth + 1):
                    return True
            return False
        return bt(list(range(n)), 0)

    def is_valid(n, adj):
        if count_k4(n, adj) != 0:
            return False
        if has_k6_in_complement(adj, n):
            return False
        return True

    def delta_violations(u, v, adj, n):
        """Returns (delta_k4, delta_i6_proxy) for flipping edge (u,v)"""
        is_edge = bool((adj[u] >> v) & 1)
        
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        # Count edges in common (C) - these form K4s with (u,v)
        deg_sum = 0
        c = common
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            deg_sum += (adj[x] & common).bit_count()
        edges_in_common = deg_sum // 2
        
        # For I6 proxy: count I4 in S where S = vertices not adjacent to u or v
        # S = ~ (N(u) | N(v) | {u, v})
        mask = (1 << n) - 1
        s_mask = mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        
        # Count independent sets of size 4 in induced subgraph on S
        # This is expensive, so use a proxy: count I3 in S (triangles in complement of S)
        # Or just use the number of non-edges in S as a rough proxy
        
        # For efficiency, count triangles in the complement of the induced subgraph on S
        # i.e., count I3 in G[S]
        verts = []
        m = s_mask
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        
        k = len(verts)
        i3_count = 0
        if k >= 3:
            for i in range(k):
                vi = verts[i]
                for j in range(i + 1, k):
                    vj = verts[j]
                    if not ((adj[vi] >> vj) & 1):
                        # Count common non-neighbors in verts after j
                        for l in range(j + 1, k):
                            vl = verts[l]
                            if not ((adj[vi] >> vl) & 1) and not ((adj[vj] >> vl) & 1):
                                i3_count += 1
        
        if is_edge:
            # Removing edge: destroys K4s, creates I6s
            return -edges_in_common, i3_count
        else:
            # Adding edge: creates K4s, destroys I6s
            return edges_in_common, -i3_count

    def compute_cost(adj, n):
        """Compute total violation count for cost function"""
        k4_count = count_k4(n, adj)
        i6_proxy = count_i6(n, adj)
        return k4_count + i6_proxy

    best_n = 0
    best_adj = None

    # Try multiple target sizes and densities
    target_sizes = [35, 34, 33, 32, 31, 30, 28, 26, 24, 22, 20, 18, 17]
    densities = [0.3, 0.4, 0.5, 0.6, 0.7]

    for target_n in target_sizes:
        if time.time() > deadline - 5:
            break
        
        for density in densities:
            if time.time() > deadline - 10:
                break
            
            # Initialize with random graph
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < density:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_cost = compute_cost(adj, target_n)
            best_local_cost = current_cost
            best_local_adj = adj[:]
            
            # Simulated Annealing
            T = 2.0
            cooling = 0.9995
            min_T = 0.05
            
            step = 0
            while T > min_T and time.time() < deadline - 15:
                step += 1
                
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                # Flip edge
                is_edge = bool((adj[u] >> v) & 1)
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = compute_cost(adj, target_n)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local_adj = adj[:]
                        
                        # Check if valid
                        if current_cost == 0 and time.time() < deadline - 20:
                            if is_valid(target_n, best_local_adj):
                                return {'n': target_n, 'edges': get_edges(best_local_adj, target_n)}
                else:
                    # Revert
                    if is_edge:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                T *= cooling
            
            # After SA, do greedy repair on best_local_adj
            if best_local_cost < 50 and time.time() < deadline - 20:
                adj = best_local_adj[:]
                improved = True
                while improved and time.time() < deadline - 15:
                    improved = False
                    best_delta = 0
                    best_u = -1
                    best_v = -1
                    
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            dk4, di6 = delta_violations(u, v, adj, target_n)
                            total_delta = dk4 + di6
                            if total_delta < best_delta:
                                best_delta = total_delta
                                best_u = u
                                best_v = v
                                improved = True
                    
                    if best_u != -1:
                        if (adj[best_u] >> best_v) & 1:
                            adj[best_u] &= ~(1 << best_v)
                            adj[best_v] &= ~(1 << best_u)
                        else:
                            adj[best_u] |= (1 << best_v)
                            adj[best_v] |= (1 << best_u)
                
                if is_valid(target_n, adj):
                    return {'n': target_n, 'edges': get_edges(adj, target_n)}
                
                # Update best if valid or better
                if compute_cost(adj, target_n) < best_local_cost:
                    best_local_adj = adj[:]
                    best_local_cost = compute_cost(adj, target_n)
            
            # Track best valid graph
            if best_local_cost == 0 and is_valid(target_n, best_local_adj):
                if target_n > best_n:
                    best_n = target_n
                    best_adj = best_local_adj[:]

    if best_adj is not None:
        return {'n': best_n, 'edges': get_edges(best_adj, best_n)}
    
    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': get_edges(adj, 17)}
