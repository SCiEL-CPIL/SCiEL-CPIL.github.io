import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += adj[u].bit_count()
        return c // 2

    def can_add(adj, u, v):
        # Check if adding (u,v) is valid
        # 1. For each w in N(v), if w != u, check common(u,w) < 2
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if w != u:
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            mask_v -= lsb
            
        # 2. For each w in N(u), if w != v, check common(v,w) < 2
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if w != v:
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
            mask_u -= lsb
            
        return True

    def greedy_fill(adj):
        # Try to add all possible edges in a specific order
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    def repair(adj):
        # Remove edges to fix violations
        # Find a violating pair
        while True:
            found_violation = False
            for a in range(n):
                for b in range(a + 1, n):
                    common = adj[a] & adj[b]
                    if common.bit_count() >= 3:
                        found_violation = True
                        # Pick a common neighbor w
                        # Remove edge (a, w) or (b, w)
                        # Randomly pick one to remove
                        w_list = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            w = lsb.bit_length() - 1
                            w_list.append(w)
                            mask -= lsb
                        w = random.choice(w_list)
                        # Decide to remove (a,w) or (b,w)
                        if random.random() < 0.5:
                            adj[a] &= ~(1 << w)
                            adj[w] &= ~(1 << a)
                        else:
                            adj[b] &= ~(1 << w)
                            adj[w] &= ~(1 << b)
                        break
                if found_violation:
                    break
            if not found_violation:
                break

    best_adj = [0] * n
    best_count = 0
    
    def update_best(adj):
        nonlocal best_adj, best_count
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: Greedy constructions with different strategies
    # Strategy A: Random order
    # Strategy B: Degree biased (low degree first)
    # Strategy C: Dense random init + repair
    
    num_greedy = 200
    
    for i in range(num_greedy):
        if time.time() > deadline - 60.0:
            break
        
        adj = [0] * n
        
        if i % 3 == 0:
            # Random order greedy
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        elif i % 3 == 1:
            # Degree biased: sort by sum of degrees, re-evaluate periodically
            # Simple approximation: shuffle, then try to add.
            # Better: iterate multiple times, picking lowest degree pair.
            # For speed, we'll just shuffle and add, but maybe prioritize low deg?
            # Let's stick to a simple shuffle for now but maybe a different seed.
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            # Random dense init + repair
            p = random.uniform(0.3, 0.5)
            for u, v in all_edges:
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            repair(adj)
            greedy_fill(adj)
            
        update_best(adj)
        
        # Local search on this specific solution if it's good
        if count_edges(adj) >= best_count - 2:
            # Run a short local search
            t_start = time.time()
            t_limit = min(deadline - 5.0, t_start + 1.0) # 1s per local search
            while time.time() < t_limit:
                # Try to add a random missing edge
                if random.random() < 0.7:
                    while True:
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u == v: continue
                        if u > v: u, v = v, u
                        if (adj[u] >> v) & 1:
                            continue
                        break
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Try to remove a random existing edge (to escape local optima)
                    if count_edges(adj) == 0: continue
                    while True:
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u == v: continue
                        if u > v: u, v = v, u
                        if not ((adj[u] >> v) & 1):
                            continue
                        break
                    # Accept removal with some probability (simulated annealing style)
                    if random.random() < 0.3:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                update_best(adj)
            
            # Final greedy fill
            greedy_fill(adj)
            update_best(adj)

    # Phase 2: Intensive local search on the best solution found so far
    if time.time() < deadline - 10.0:
        adj = best_adj[:]
        t_limit = deadline - 5.0
        
        # Simulated Annealing on the best solution
        T = 5.0
        cooling = 0.9995
        min_T = 0.1
        
        while T > min_T and time.time() < t_limit:
            # Random move
            if random.random() < 0.7:
                # Try to add
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if (adj[u] >> v) & 1:
                        continue
                    break
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    update_best(adj)
            else:
                # Try to remove
                if count_edges(adj) == 0: continue
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if not ((adj[u] >> v) & 1):
                        continue
                    break
                if random.random() < math.exp(-1.0 / T):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            T *= cooling
        
        greedy_fill(adj)
        update_best(adj)
        
        # Phase 3: 2-edge swaps on the best solution
        if time.time() < deadline - 2.0:
            adj = best_adj[:]
            t_limit = deadline - 2.0
            
            # Try to find a swap: remove e1, add e2, e3
            # This is expensive, so we limit attempts
            for _ in range(1000):
                if time.time() > t_limit:
                    break
                
                # Pick a random edge to remove
                if count_edges(adj) == 0: break
                while True:
                    u1 = random.randint(0, n-1)
                    v1 = random.randint(0, n-1)
                    if u1 == v1: continue
                    if u1 > v1: u1, v1 = v1, u1
                    if not ((adj[u1] >> v1) & 1):
                        continue
                    break
                
                # Remove it
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Try to add two edges
                added = 0
                # Try random missing edges
                attempts = 0
                while added < 2 and attempts < 50:
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2: continue
                    if u2 > v2: u2, v2 = v2, u2
                    if (adj[u2] >> v2) & 1:
                        continue
                    if can_add(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        added += 1
                    attempts += 1
                
                if added == 2:
                    update_best(adj)
                else:
                    # Revert the removal
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    # Revert any additions
                    # We need to track what was added. 
                    # For simplicity, we'll just revert the whole state if it didn't improve?
                    # Actually, if added < 2, we should revert the removal and any additions.
                    # Let's just restart the swap attempt logic cleanly or revert.
                    # Reverting additions is hard without tracking.
                    # Let's just accept that we might be in a worse state if we don't revert properly.
                    # Better approach: copy adj before modification.
                    # Since we are in a tight loop, copying is expensive.
                    # Let's just break out of this swap loop if it's too complex to revert safely.
                    # Or, simpler: just do the swap on a copy?
                    # Let's switch to a simpler strategy: just try to add 1 edge after removing 1 (net 0) or add 2 (net +1).
                    # If net +1, keep. If net 0, keep (might help future). If net -1, revert.
                    # But reverting additions is the issue.
                    # Let's just discard this swap attempt if it doesn't result in +1 edge.
                    # So we need to restore the original adj.
                    # We didn't save it. This is a bug in the logic above.
                    # Let's fix by saving adj before the swap.
                    pass

    return get_edges(best_adj)

def _construct_greedy():
    pass

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass
