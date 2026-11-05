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
        # Exact branch and bound to check if alpha >= 10
        # Returns the set if found, else None
        # We only need to know if it exists for validity check
        
        # Order vertices by degree (ascending) to help pruning? 
        # For n=39, simple DFS with bitsets is fast enough if alpha is small.
        
        def dfs(remaining, current_size):
            if current_size == 10:
                return True
            if not remaining:
                return False
            if current_size + popcount(remaining) < 10:
                return False
            
            # Pick a vertex
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Try including v
            # If we include v, we remove v and its neighbors from remaining
            new_remaining = remaining & ~(adj[v] | lsb)
            if dfs(new_remaining, current_size + 1):
                return True
            
            # Try excluding v
            if dfs(remaining ^ lsb, current_size):
                return True
            
            return False
        
        return dfs((1 << n) - 1, 0)

    def get_paley_graph(q):
        # q must be 1 mod 4 for triangle-free Paley graph
        if q % 4 != 1:
            return None
        
        qr_set = set()
        for i in range(1, q):
            qr_set.add((i * i) % q)
        
        n = q
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = (u - v) % q
                if diff in qr_set:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def find_max_independent_set(adj, n):
        # Greedy or simple B&B to find one large independent set
        # We just need one of size 6 for P_37
        best_set = []
        
        # Simple recursive search for an independent set of size 6
        # Since alpha(P_37)=6, this should find it quickly
        
        def dfs(remaining, current_set):
            if len(current_set) == 6:
                return current_set
            if not remaining:
                return None
            if len(current_set) + popcount(remaining) < 6:
                return None
            
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

    # 1. Try to construct n=39 by extending P_37
    adj_37 = get_paley_graph(37)
    if adj_37:
        # Find an independent set I of size 6 in P_37
        I = find_max_independent_set(adj_37, 37)
        if I and len(I) == 6:
            # Extend to n=39
            n = 39
            adj = [0] * n
            # Copy P_37
            for i in range(37):
                adj[i] = adj_37[i]
            
            # Add vertices 37 and 38
            # Connect 37 and 38 to all vertices in I
            for v in I:
                adj[37] |= (1 << v)
                adj[v] |= (1 << 37)
                adj[38] |= (1 << v)
                adj[v] |= (1 << 38)
            
            # No edge between 37 and 38 to keep it simple (and triangle-free)
            
            # Verify
            if find_triangle(adj, n) is None:
                if not find_independent_set_10(adj, n):
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        # If extension failed or verification failed, return P_37
        # Verify P_37 just in case
        if find_triangle(adj_37, 37) is None:
            if not find_independent_set_10(adj_37, 37):
                edges = []
                for i in range(37):
                    for j in range(i + 1, 37):
                        if (adj_37[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': 37, 'edges': edges}

    # Fallback to seed graph n=26
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
