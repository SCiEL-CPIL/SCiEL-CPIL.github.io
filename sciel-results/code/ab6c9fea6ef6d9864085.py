import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    full_mask = (1 << n) - 1

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
        
        # Check triples containing u (not v)
        # For x, y in V \ {u, v}:
        # if v in N(x) and v in N(y) and |N(u) & N(x) & N(y)| == 2 -> invalid
        # Note: N(u) is old N(u)
        nu = adj[u]
        nv = adj[v]
        
        # Precompute list of neighbors for faster access? 
        # Bitmask operations are fast enough.
        
        # Iterate over all pairs x, y not u, v
        # We can iterate x from 0 to n-1, y from x+1 to n-1
        for x in range(n):
            if x == u or x == v:
                continue
            ax = adj[x]
            if not (nv & (1 << x)): # v not in N(x)
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                ay = adj[y]
                if not (nv & (1 << y)): # v not in N(y)
                    continue
                # v is common neighbor of x, y
                # Check if u has 2 common neighbors with x, y
                common_u = nu & ax & ay
                if common_u.bit_count() == 2:
                    return False
        
        # Check triples containing v (not u)
        for x in range(n):
            if x == u or x == v:
                continue
            ax = adj[x]
            if not (nu & (1 << x)): # u not in N(x)
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                ay = adj[y]
                if not (nu & (1 << y)): # u not in N(y)
                    continue
                # u is common neighbor of x, y
                # Check if v has 2 common neighbors with x, y
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

    def local_search(adj, max_iters=1000):
        adj = adj[:]
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Try to add an edge
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
            
            # Try 1-swap
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_in or not edges_out:
                break
                
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            for e_rem in edges_in[:50]:
                if swapped:
                    break
                u1, v1 = e_rem
                for e_add in edges_out[:50]:
                    u2, v2 = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    # Remove e_rem
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    # Check if we can add e_add
                    if can_add_edge(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        swapped = True
                        break
                    else:
                        # Restore e_rem
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                if not swapped:
                    # If we broke out of inner loop without swapping, e_rem was restored
                    # But if we finished inner loop, e_rem is restored.
                    # Wait, if we broke inner loop with swap, e_rem is NOT restored.
                    # If we didn't swap, we restored e_rem in the else block.
                    # So state is consistent.
                    continue
            
            if not swapped:
                break
        return adj

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

    # Multiple random restarts
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        seed = 42 + trial * 13
        adj = greedy_construct(seed)
        adj = local_search(adj, max_iters=200)
        update_best(adj)
        
        # Further polish
        adj = local_search(adj, max_iters=500)
        update_best(adj)

    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
