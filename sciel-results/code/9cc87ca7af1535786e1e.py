import random
import time

def is_k4_free(adj, n):
    """Fast K4-free check using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common_ab = adj[a] & adj[b]
            common_ab &= ~((1 << (b + 1)) - 1)
            while common_ab:
                lsb = common_ab & -common_ab
                c = lsb.bit_length() - 1
                common_abc = adj[a] & adj[b] & adj[c]
                common_abc &= ~((1 << (c + 1)) - 1)
                if common_abc:
                    return False
                common_ab ^= lsb
    return True

def has_triangle_mask(mask, adj, n):
    """Check if induced subgraph on mask has a triangle."""
    m = mask
    while m:
        lsb_u = m & -m
        u = lsb_u.bit_length() - 1
        m ^= lsb_u
        nbrs_u = adj[u] & mask & ~((1 << (u + 1)) - 1)
        while nbrs_u:
            lsb_v = nbrs_u & -nbrs_u
            v = lsb_v.bit_length() - 1
            nbrs_u ^= lsb_v
            if adj[u] & adj[v] & mask:
                return True
    return False

def compute_largest_tf(adj, n, deadline):
    """
    Find the largest triangle-free induced subgraph using branch and bound.
    Uses a decision-based approach: try to find TF sets of increasing size.
    """
    best = 0
    
    # Greedy lower bound with multiple orderings
    for attempt in range(30):
        greedy_mask = 0
        order = list(range(n))
        if attempt > 0:
            random.shuffle(order)
        
        for v in order:
            if greedy_mask & (1 << v):
                continue
            can_add = True
            nbrs_in_greedy = adj[v] & greedy_mask
            m = nbrs_in_greedy
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                if adj[u] & adj[v] & greedy_mask:
                    can_add = False
                    break
            if can_add:
                greedy_mask |= (1 << v)
        size = bin(greedy_mask).count('1')
        if size > best:
            best = size
    
    if best >= n:
        return best
    
    # Try to find TF sets of size best+1, best+2, ... using backtracking
    def exists_tf_of_size(target_size):
        if time.time() > deadline:
            return False
        
        def backtrack(idx, chosen_mask, chosen_count):
            if time.time() > deadline:
                return False
            
            if chosen_count == target_size:
                return True
            
            if idx >= n:
                return False
            
            if chosen_count + (n - idx) < target_size:
                return False
            
            v = idx
            
            # Try including v
            if not (chosen_mask & (1 << v)):
                can_add = True
                nbrs_in_chosen = adj[v] & chosen_mask
                m = nbrs_in_chosen
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m ^= lsb
                    if adj[u] & adj[v] & chosen_mask:
                        can_add = False
                        break
                
                if can_add:
                    if backtrack(idx + 1, chosen_mask | (1 << v), chosen_count + 1):
                        return True
            
            # Try excluding v
            if backtrack(idx + 1, chosen_mask, chosen_count):
                return True
            
            return False
        
        return backtrack(0, 0, 0)
    
    current_best = best
    for target in range(best + 1, n + 1):
        if time.time() > deadline:
            break
        if exists_tf_of_size(target):
            current_best = target
        else:
            break
    
    return current_best

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 18
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_l = n + 1
    best_adj = None
    
    def update_best(adj, l):
        nonlocal best_l, best_adj
        if l < best_l:
            best_l = l
            best_adj = adj[:]
    
    random.seed(42)
    
    candidates = []
    
    # Generate random 4-partite graphs with random edge deletions
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        # Random partition sizes summing to 18
        # Use various splits
        if trial % 4 == 0:
            parts = [list(range(0, 5)), list(range(5, 10)), list(range(10, 14)), list(range(14, 18))]
        elif trial % 4 == 1:
            parts = [list(range(0, 4)), list(range(4, 8)), list(range(8, 13)), list(range(13, 18))]
        elif trial % 4 == 2:
            parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 15)), list(range(15, 18))]
        else:
            parts = [list(range(0, 3)), list(range(3, 8)), list(range(8, 13)), list(range(13, 18))]
        
        adj = [0] * n
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Randomly delete some edges to break K4s
        for _ in range(2000):
            if is_k4_free(adj, n):
                break
            found_k4 = False
            # Find a K4 and remove a random edge from it
            for a in range(n):
                if found_k4:
                    break
                for b in range(a + 1, n):
                    if found_k4:
                        break
                    if not (adj[a] & (1 << b)):
                        continue
                    common_ab = adj[a] & adj[b]
                    common_ab &= ~((1 << (b + 1)) - 1)
                    c_list = []
                    m = common_ab
                    while m:
                        lsb = m & -m
                        c = lsb.bit_length() - 1
                        c_list.append(c)
                        m ^= lsb
                    for c in c_list:
                        if found_k4:
                            break
                        common_abc = adj[a] & adj[b] & adj[c]
                        common_abc &= ~((1 << (c + 1)) - 1)
                        d_list = []
                        m2 = common_abc
                        while m2:
                            lsb = m2 & -m2
                            d = lsb.bit_length() - 1
                            d_list.append(d)
                            m2 ^= lsb
                        for d in d_list:
                            # Found K4: a, b, c, d
                            edge_choices = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                            e = random.choice(edge_choices)
                            adj[e[0]] &= ~(1 << e[1])
                            adj[e[1]] &= ~(1 << e[0])
                            found_k4 = True
                            break
        
        if is_k4_free(adj, n):
            candidates.append(adj)
    
    # Also try random 3-partite (Turan-like) with deletions
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        adj = [0] * n
        parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for u in parts3[i]:
                    for v in parts3[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        for _ in range(3000):
            if is_k4_free(adj, n):
                break
            found_k4 = False
            for a in range(n):
                if found_k4:
                    break
                for b in range(a + 1, n):
                    if found_k4:
                        break
                    if not (adj[a] & (1 << b)):
                        continue
                    common_ab = adj[a] & adj[b]
                    common_ab &= ~((1 << (b + 1)) - 1)
                    c_list = []
                    m = common_ab
                    while m:
                        lsb = m & -m
                        c = lsb.bit_length() - 1
                        c_list.append(c)
                        m ^= lsb
                    for c in c_list:
                        if found_k4:
                            break
                        common_abc = adj[a] & adj[b] & adj[c]
                        common_abc &= ~((1 << (c + 1)) - 1)
                        d_list = []
                        m2 = common_abc
                        while m2:
                            lsb = m2 & -m2
                            d = lsb.bit_length() - 1
                            d_list.append(d)
                            m2 ^= lsb
                        for d in d_list:
                            edge_choices = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                            e = random.choice(edge_choices)
                            adj[e[0]] &= ~(1 << e[1])
                            adj[e[1]] &= ~(1 << e[0])
                            found_k4 = True
                            break
        
        if is_k4_free(adj, n):
            candidates.append(adj)
    
    # Evaluate candidates
    for adj in candidates:
        if time.time() > deadline - 20:
            break
        l = compute_largest_tf(adj, n, deadline)
        update_best(adj, l)
    
    # Simulated annealing refinement
    if best_adj is not None and time.time() < deadline - 15:
        current_adj = best_adj[:]
        current_l = best_l
        T = 1.0
        
        for step in range(10000):
            if time.time() > deadline - 5:
                break
            
            i, j = random.sample(range(n), 2)
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            if not is_k4_free(current_adj, n):
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
                continue
            
            new_l = compute_largest_tf(current_adj, n, deadline)
            
            if new_l < current_l:
                current_l = new_l
                update_best(current_adj, new_l)
            elif new_l == current_l:
                pass
            else:
                delta = new_l - current_l
                if random.random() < pow(2.71828, -delta / T):
                    pass
                else:
                    current_adj[i] ^= (1 << j)
                    current_adj[j] ^= (1 << i)
            
            T *= 0.999
    
    if best_adj is None:
        if candidates:
            return adj_to_edges(candidates[-1])
        else:
            # Fallback: Turan T(18,3)
            adj_turan = [0] * n
            parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
            for i in range(3):
                for j in range(i + 1, 3):
                    for u in parts3[i]:
                        for v in parts3[j]:
                            adj_turan[u] |= (1 << v)
                            adj_turan[v] |= (1 << u)
            return adj_to_edges(adj_turan)
    
    return adj_to_edges(best_adj)
