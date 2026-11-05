import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def popcount(x):
        return bin(x).count("1")
        
    def find_triangle(adj, n):
        for u in range(n):
            neighbors = adj[u]
            neigh_list = []
            mask = neighbors
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                neigh_list.append(v)
                mask ^= lsb
            for i in range(len(neigh_list)):
                for j in range(i + 1, len(neigh_list)):
                    if (adj[neigh_list[i]] >> neigh_list[j]) & 1:
                        return (u, neigh_list[i], neigh_list[j])
        return None

    def find_independent_set_10(adj, n):
        # Simple backtracking to find one
        # Order vertices by degree (ascending) might help?
        # For now, just use bitset DFS
        
        def dfs(remaining, current_set):
            if len(current_set) == 10:
                return current_set
            if not remaining:
                return None
            
            # Pruning
            if len(current_set) + popcount(remaining) < 10:
                return None
            
            # Pick a vertex
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Try including v
            if all(not (adj[v] & (1 << u)) for u in current_set):
                res = dfs(remaining & ~(adj[v] | lsb), current_set + [v])
                if res:
                    return res
            
            # Try excluding v
            res = dfs(remaining ^ lsb, current_set)
            if res:
                return res
            return None
        
        return dfs((1 << n) - 1, [])

    def local_search(n, p_init):
        nonlocal deadline
        # Initialize
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p_init:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for _ in range(5000):
            if time.time() > deadline:
                return None
            
            # Check and fix triangles
            tri = find_triangle(adj, n)
            if tri:
                u, v, w = tri
                # Remove one random edge
                edge_to_remove = random.choice([(u, v), (u, w), (v, w)])
                a, b = edge_to_remove
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                continue
            
            # Check and fix independent sets
            ind_set = find_independent_set_10(adj, n)
            if ind_set:
                # Add a random edge within the independent set
                # The set is independent, so no edges exist between them
                a, b = random.sample(ind_set, 2)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                continue
            
            # If we get here, no triangles and no independent set of 10
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        return None

    # Try n=40
    for p in [0.3, 0.4, 0.5]:
        res = local_search(40, p)
        if res:
            return res
            
    # Try n=39
    for p in [0.3, 0.4, 0.5]:
        res = local_search(39, p)
        if res:
            return res
            
    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
