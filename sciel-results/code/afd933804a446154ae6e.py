import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 20
    all_mask = (1 << n) - 1

    def common_count(adj, i, j, k):
        return (adj[i] & adj[j] & adj[k]).bit_count()

    def is_valid(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def check_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        nu = adj[u]
        nv = adj[v]
        # Triples (u, a, b): new common = old + (1 if v in N(a) and v in N(b))
        for a in range(n):
            if a == u:
                continue
            na = adj[a]
            for b in range(a + 1, n):
                if b == u:
                    continue
                nb = adj[b]
                old_cnt = (nu & na & nb).bit_count()
                if old_cnt >= 2:
                    if (nv & na & nb).bit_count() >= 1:
                        return False
        # Triples (v, a, b)
        for a in range(n):
            if a == v:
                continue
            na = adj[a]
            for b in range(a + 1, n):
                if b == v:
                    continue
                nb = adj[b]
                old_cnt = (nv & na & nb).bit_count()
                if old_cnt >= 2:
                    if (nu & na & nb).bit_count() >= 1:
                        return False
        return True

    def try_add(adj, u, v):
        if adj[u] & (1 << v):
            return False
        if not check_add(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def greedy_build(seed_order=None):
        adj = [0] * n
        order = list(range(n))
        if seed_order is None:
            random.shuffle(order)
        else:
            order = seed_order
        for u in order:
            candidates = [v for v in range(n) if v != u]
            random.shuffle(candidates)
            for v in candidates:
                try_add(adj, u, v)
        return adj

    def repair(adj, time_limit):
        """Remove edges to fix violations."""
        t_end = time.time() + time_limit
        while time.time() < t_end:
            # Find a violation
            viol = None
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    aij = ai & adj[j]
                    for k in range(j + 1, n):
                        if (aij & adj[k]).bit_count() >= 3:
                            viol = (i, j, k)
                            break
                    if viol:
                        break
                if viol:
                    break
            if not viol:
                return True
            
            i, j, k = viol
            # Remove a random edge incident to one of the triple vertices
            # that is part of the violation
            common = adj[i] & adj[j] & adj[k]
            # Pick a vertex in common to remove an edge from
            # Actually, we need to remove an edge such that the common count drops below 3
            # Remove edge (i, v) where v is in common
            v_list = []
            v = common
            while v:
                lsb = v & (-v)
                idx = lsb.bit_length() - 1
                v_list.append(idx)
                v ^= lsb
            
            if not v_list:
                break
            
            # Randomly pick which edge to remove
            v_idx = random.choice(v_list)
            # Remove edge (i, v_idx)
            adj[i] &= ~(1 << v_idx)
            adj[v_idx] &= ~(1 << i)
        
        return is_valid(adj)

    def local_search(adj, time_limit):
        """Local search to maximize edges while maintaining validity."""
        best_adj = list(adj)
        best_count = edge_count(best_adj)
        cur_adj = list(adj)
        cur_count = best_count
        
        t_end = time.time() + time_limit
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < t_end:
            if no_improve > max_no_improve:
                break
            
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            
            has_uv = bool(cur_adj[u] & (1 << v))
            
            if has_uv:
                # Try removing (u,v) and adding (x,y)
                x = random.randrange(n)
                y = random.randrange(n)
                if x == y or x == u or x == v or y == u or y == v:
                    no_improve += 1
                    continue
                if cur_adj[x] & (1 << y):
                    no_improve += 1
                    continue
                # Remove (u,v)
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                # Try adding (x,y)
                if check_add(cur_adj, x, y):
                    cur_adj[x] |= (1 << y)
                    cur_adj[y] |= (1 << x)
                    new_count = cur_count  # same count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = list(cur_adj)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    no_improve += 1
            else:
                # Try adding (u,v)
                if check_add(cur_adj, u, v):
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = list(cur_adj)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        return best_adj, best_count

    def simulated_annealing(adj, time_limit):
        """Simulated annealing to escape local optima."""
        cur_adj = list(adj)
        cur_count = edge_count(cur_adj)
        best_adj = list(cur_adj)
        best_count = cur_count
        
        t_end = time.time() + time_limit
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < t_end and T > T_min:
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            
            has_uv = bool(cur_adj[u] & (1 << v))
            
            if has_uv:
                # Try removing (u,v)
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                new_count = cur_count - 1
                delta = new_count - cur_count  # -1
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    cur_count = new_count
                else:
                    # Revert
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
            else:
                # Try adding (u,v)
                if check_add(cur_adj, u, v):
                    cur_adj[u] |= (1 << v)
                    cur_adj[v] |= (1 << u)
                    new_count = cur_count + 1
                    delta = 1
                    if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                        cur_count = new_count
                        if cur_count > best_count:
                            best_count = cur_count
                            best_adj = list(cur_adj)
                    else:
                        # Revert
                        cur_adj[u] &= ~(1 << v)
                        cur_adj[v] &= ~(1 << u)
                else:
                    # Can't add, try a swap
                    x = random.randrange(n)
                    y = random.randrange(n)
                    if x == y or x == u or x == v or y == u or y == v:
                        T *= cooling
                        continue
                    if cur_adj[x] & (1 << y):
                        T *= cooling
                        continue
                    # Remove (u,v) doesn't exist, so try adding (x,y) if valid
                    # Actually, let's just try removing a random existing edge and adding (u,v)
                    # Find an existing edge to remove
                    existing_edges = []
                    for i in range(n):
                        mask = cur_adj[i]
                        m = mask
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            if idx > i:
                                existing_edges.append((i, idx))
                            m ^= lsb
                    
                    if existing_edges:
                        rx, ry = random.choice(existing_edges)
                        # Remove (rx, ry)
                        cur_adj[rx] &= ~(1 << ry)
                        cur_adj[ry] &= ~(1 << rx)
                        # Try adding (u, v)
                        if check_add(cur_adj, u, v):
                            cur_adj[u] |= (1 << v)
                            cur_adj[v] |= (1 << u)
                            # Count stays same
                            if cur_count > best_count:
                                best_count = cur_count
                                best_adj = list(cur_adj)
                        else:
                            # Revert
                            cur_adj[rx] |= (1 << ry)
                            cur_adj[ry] |= (1 << rx)
            
            T *= cooling
        
        return best_adj, best_count

    random.seed(42)
    
    best_adj = None
    best_count = -1
    
    # Multiple restarts with different strategies
    num_restarts = 30
    restart_time = 3.0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        strategy = r % 4
        
        if strategy == 0:
            # Random greedy
            adj = greedy_build()
        elif strategy == 1:
            # Random graph then repair
            adj = [0] * n
            p = 0.35
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            repair(adj, 0.5)
        elif strategy == 2:
            # Greedy with different vertex order
            order = list(range(n))
            random.shuffle(order)
            adj = greedy_build(order)
        else:
            # Start from a known structure and perturb
            adj = [0] * n
            # Start with a random matching or sparse structure
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.25:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            repair(adj, 0.5)
        
        # Local search
        ls_adj, ls_count = local_search(adj, restart_time * 0.5)
        if ls_count > best_count:
            best_count = ls_count
            best_adj = ls_adj
        
        # Simulated annealing from local search result
        if time.time() < deadline - 5:
            sa_adj, sa_count = simulated_annealing(ls_adj, restart_time * 0.4)
            if sa_count > best_count:
                best_count = sa_count
                best_adj = sa_adj
        
        # Try to add more edges greedily
        if time.time() < deadline - 3:
            for i in range(n):
                for j in range(i + 1, n):
                    if not (best_adj[i] & (1 << j)):
                        if check_add(best_adj, i, j):
                            best_adj[i] |= (1 << j)
                            best_adj[j] |= (1 << i)
                            best_count += 1
    
    # Final greedy pass
    if best_adj is not None and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (best_adj[i] & (1 << j)):
                        if check_add(best_adj, i, j):
                            best_adj[i] |= (1 << j)
                            best_adj[j] |= (1 << i)
                            best_count += 1
                            improved = True
    
    if best_adj is None or best_count < 36:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
