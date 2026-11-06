import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def popcount(x):
        return x.bit_count()

    def is_valid_add(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # u and v common neighbors (before adding): must be <= 2
        if popcount(mask_u & mask_v) > 2:
            return False
        
        # For each neighbor w of u (w != v): check common(w, v) <= 1
        temp = mask_u
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == v:
                continue
            if popcount(adj_mask[w] & mask_v) > 1:
                return False
        
        # For each neighbor w of v (w != u): check common(u, w) <= 1
        temp = mask_v
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == u:
                continue
            if popcount(mask_u & adj_mask[w]) > 1:
                return False
        
        return True

    def add_edge_fast(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge_fast(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def edges_from_mask(adj_mask):
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def edge_count(adj_mask):
        cnt = 0
        for u in range(n):
            cnt += popcount(adj_mask[u])
        return cnt // 2

    def validate_adj_mask(adj_mask):
        """Verify K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj_mask[u] & adj_mask[v]) > 2:
                    return False
        return True

    def greedy_construct(seed_offset=0, base_type='cliques'):
        """Greedy construction with different base structures."""
        rng = random.Random(seed_offset)
        adj_mask = [0] * n
        
        if base_type == 'cliques':
            # Start with 5 cliques of size 4
            for g in range(5):
                base = 4 * g
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj_mask[base + i] |= (1 << (base + j))
                        adj_mask[base + j] |= (1 << (base + i))
        elif base_type == 'circulant':
            # 5-circulant: connect to ±1, ±2, +5 mod 20
            for i in range(n):
                for d in [1, 2, 5]:
                    j = (i + d) % n
                    if j != i and not ((adj_mask[i] >> j) & 1):
                        if is_valid_add(adj_mask, i, j):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
        elif base_type == 'random':
            # Random 4-regular base
            for i in range(n):
                candidates = [j for j in range(n) if j != i and not ((adj_mask[i] >> j) & 1)]
                rng.shuffle(candidates)
                for j in candidates[:4]:
                    if is_valid_add(adj_mask, i, j):
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
        
        # Fill remaining edges greedily in random order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not (adj_mask[u] & (1 << v)):
                if is_valid_add(adj_mask, u, v):
                    add_edge_fast(adj_mask, u, v)
        
        return adj_mask

    def local_search(adj_mask, max_time=30.0, rng=None):
        """Aggressive local search with 1-edge and 2-edge swaps."""
        if rng is None:
            rng = random.Random()
        current = list(adj_mask)
        best = list(current)
        best_cnt = edge_count(current)
        start_ls = time.time()
        ls_deadline = start_ls + max_time
        
        all_edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Phase 1: Greedy add
        for _ in range(20):
            if time.time() > ls_deadline:
                break
            rng.shuffle(all_edges_list)
            added = False
            for u, v in all_edges_list:
                if not (current[u] & (1 << v)):
                    if is_valid_add(current, u, v):
                        add_edge_fast(current, u, v)
                        added = True
            if not added:
                break
        
        cnt = edge_count(current)
        if cnt > best_cnt:
            best_cnt = cnt
            best = list(current)
        
        # Phase 2: 1-edge swap
        for _ in range(3000):
            if time.time() > ls_deadline:
                break
            
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        existing.append((u, v))
            if not existing:
                break
            
            u, v = rng.choice(existing)
            
            saved = list(current)
            remove_edge_fast(current, u, v)
            
            rng.shuffle(all_edges_list)
            for a, b in all_edges_list:
                if not (current[a] & (1 << b)):
                    if is_valid_add(current, a, b):
                        add_edge_fast(current, a, b)
            
            cnt = edge_count(current)
            if cnt > best_cnt:
                best_cnt = cnt
                best = list(current)
            else:
                current = saved
        
        # Phase 3: 2-edge swap
        for _ in range(800):
            if time.time() > ls_deadline:
                break
            
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        existing.append((u, v))
            if len(existing) < 2:
                break
            
            (u1, v1), (u2, v2) = rng.sample(existing, 2)
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            
            saved = list(current)
            remove_edge_fast(current, u1, v1)
            remove_edge_fast(current, u2, v2)
            
            rng.shuffle(all_edges_list)
            for a, b in all_edges_list:
                if not (current[a] & (1 << b)):
                    if is_valid_add(current, a, b):
                        add_edge_fast(current, a, b)
            
            cnt = edge_count(current)
            if cnt > best_cnt:
                best_cnt = cnt
                best = list(current)
            else:
                current = saved
        
        return best

    # Main loop
    best_adj = None
    best_count = 0
    
    # Try multiple initial constructions with different base types
    base_types = ['cliques', 'circulant', 'random']
    
    for base_type in base_types:
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            adj = greedy_construct(seed * 100 + hash(base_type) % 1000, base_type)
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj)
    
    # Local search on best
    if best_adj and time.time() < deadline - 5:
        improved = local_search(best_adj, max_time=min(30.0, deadline - time.time() - 3))
        cnt = edge_count(improved)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved
    
    # More local search iterations if time permits
    while time.time() < deadline - 3:
        improved = local_search(best_adj, max_time=min(15.0, deadline - time.time() - 2))
        cnt = edge_count(improved)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved
        else:
            # Try a fresh greedy + local search
            base_type = base_types[random.randint(0, 2)]
            adj = greedy_construct(random.randint(0, 10000), base_type)
            cnt2 = edge_count(adj)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = list(adj)
            if time.time() > deadline - 5:
                break
            improved2 = local_search(best_adj, max_time=min(10.0, deadline - time.time() - 2))
            cnt3 = edge_count(improved2)
            if cnt3 > best_count:
                best_count = cnt3
                best_adj = improved2
    
    # Validate and return
    if best_adj is None:
        # Fallback: just return a valid small graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if is_valid_add(adj, u, v):
                    add_edge_fast(adj, u, v)
        best_adj = adj
    
    # Final validation
    if not validate_adj_mask(best_adj):
        # Something went wrong, rebuild conservatively
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if is_valid_add(adj, u, v):
                    add_edge_fast(adj, u, v)
        best_adj = adj
    
    return edges_from_mask(best_adj)
