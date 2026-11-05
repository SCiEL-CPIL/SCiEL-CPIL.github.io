import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_pairs = len(all_pairs)
    full_mask = (1 << n) - 1

    # Precompute edge index for quick lookup
    edge_set = set(all_pairs)

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        nu = adj[u]
        nv = adj[v]
        
        # Check triples containing u (not v)
        for x in range(n):
            if x == u or x == v:
                continue
            ax = adj[x]
            if not (nv & (1 << x)):
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                ay = adj[y]
                if not (nv & (1 << y)):
                    continue
                common_u = nu & ax & ay
                if common_u.bit_count() == 2:
                    return False
        
        # Check triples containing v (not u)
        for x in range(n):
            if x == u or x == v:
                continue
            ax = adj[x]
            if not (nu & (1 << x)):
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                ay = adj[y]
                if not (nu & (1 << y)):
                    continue
                common_v = nv & ax & ay
                if common_v.bit_count() == 2:
                    return False
        
        return True

    def greedy_construct(seed):
        random.seed(seed)
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_iters=500):
        adj = adj[:]
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            added = False
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if added:
                continue
            
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_in or not edges_out:
                break
                
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            for e_rem in edges_in[:30]:
                if swapped:
                    break
                u1, v1 = e_rem
                for e_add in edges_out[:30]:
                    u2, v2 = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    if can_add_edge(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        swapped = True
                        break
                    else:
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                if not swapped:
                    continue
            
            if not swapped:
                break
        return adj

    def simulated_annealing(adj, max_steps=10000):
        adj = adj[:]
        current_count = count_edges(adj)
        best_count = current_count
        best_adj = adj[:]
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random missing edge
                missing = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid(adj):
                    new_count = current_count + 1
                    delta = 1
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        adj[u] = old_u
                        adj[v] = old_v
                else:
                    adj[u] = old_u
                    adj[v] = old_v
                    
            elif move_type < 0.7:
                # Remove a random existing edge
                existing = [(i, j) for i, j in all_pairs if (adj[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                old_u, old_v = adj[u], adj[v]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                new_count = current_count - 1
                delta = -1
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    adj[u] = old_u
                    adj[v] = old_v
                    
            else:
                # Swap: remove one, add another
                existing = [(i, j) for i, j in all_pairs if (adj[i] & (1 << j))]
                missing = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2):
                    continue
                    
                old_u1, old_v1 = adj[u1], adj[v1]
                old_u2, old_v2 = adj[u2], adj[v2]
                
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                if is_valid(adj):
                    delta = 0
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        adj[u1] = old_u1
                        adj[v1] = old_v1
                        adj[u2] = old_u2
                        adj[v2] = old_v2
                else:
                    adj[u1] = old_u1
                    adj[v1] = old_v1
                    adj[u2] = old_u2
                    adj[v2] = old_v2
            
            T *= cooling
        
        return best_adj

    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Multiple random restarts with different strategies
    for trial in range(60):
        if time.time() > deadline - 15:
            break
        seed = 42 + trial * 17
        random.seed(seed)
        
        # Mix of greedy and random initialization
        if trial % 3 == 0:
            adj = greedy_construct(seed)
        elif trial % 3 == 1:
            # Dense random start with repair
            adj = [0] * n
            p = 0.7 + 0.1 * (trial % 5)
            for i, j in all_pairs:
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            # Repair
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                if is_valid(adj):
                    break
                # Find a violation and remove an edge
                for i in range(n):
                    for j in range(i+1, n):
                        cij = adj[i] & adj[j]
                        if cij.bit_count() < 3:
                            continue
                        for k in range(j+1, n):
                            cijk = cij & adj[k]
                            if cijk.bit_count() >= 3:
                                # Remove a random edge from the K3,3
                                # Get 3 common neighbors
                                neighbors = []
                                m = cijk
                                while len(neighbors) < 3 and m:
                                    lsb = m & (-m)
                                    idx = lsb.bit_length() - 1
                                    neighbors.append(idx)
                                    m ^= lsb
                                u = random.choice([i, j, k])
                                v = random.choice(neighbors)
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                break
                            if is_valid(adj):
                                break
                        if is_valid(adj):
                            break
                    if is_valid(adj):
                        break
        else:
            adj = greedy_construct(seed + 1000)
        
        # Local search first
        adj = local_search(adj, max_iters=100)
        update_best(adj)
        
        # Simulated annealing
        adj = simulated_annealing(adj, max_steps=5000)
        update_best(adj)
        
        # Final local search polish
        adj = local_search(adj, max_iters=200)
        update_best(adj)

    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
