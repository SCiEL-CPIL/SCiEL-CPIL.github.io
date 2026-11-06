import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_triangle_free(n, adj):
        for u in range(n):
            temp = adj[u]
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def has_independent_set_k(n, adj, k):
        if n < k:
            return False
        
        # Fast greedy lower bound on alpha
        remaining = (1 << n) - 1
        greedy_size = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = popcount(remaining & adj[v])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            greedy_size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
            if greedy_size >= k:
                return True
        
        # Exact branch and bound with coloring
        def get_color_bound(candidates):
            color_bound = 0
            remaining = candidates
            while remaining:
                color_bound += 1
                clique = 0
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= ~(1 << v)
                    if (adj[v] & clique) == clique:
                        clique |= (1 << v)
                remaining &= ~clique
            return color_bound

        # Memoization for speed
        memo = {}
        
        def backtrack(candidates, size):
            if size >= k:
                return True
            if not candidates:
                return False
            if size + popcount(candidates) < k:
                return False
            
            key = (candidates, size)
            if key in memo:
                return memo[key]
            
            cb = get_color_bound(candidates)
            if size + cb < k:
                memo[key] = False
                return False
            
            # Pick vertex with minimum degree in G (maximum in complement)
            best_v = -1
            best_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = popcount(candidates & adj[v])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                memo[key] = False
                return False
            
            # Include best_v
            new_cands = candidates & ~(1 << best_v) & ~adj[best_v]
            if backtrack(new_cands, size + 1):
                memo[key] = True
                return True
            
            # Exclude best_v
            cands_without = candidates & ~(1 << best_v)
            if backtrack(cands_without, size):
                memo[key] = True
                return True
            
            memo[key] = False
            return False

        return backtrack((1 << n) - 1, 0)

    def edges_from_adj(n, adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                if v > u:
                    edges.append([u, v])
        return edges

    best_n = 0
    best_edges = []
    
    # Baseline: cycle C_19
    best_n = 19
    best_edges = [[i, (i + 1) % 19] for i in range(19)]

    # Try to find larger n using local search
    for n in range(40, 19, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_attempts = 3000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            
            # Generate random triangle-free graph using greedy method
            adj = [0] * n
            perm = list(range(n))
            random.shuffle(perm)
            
            # Greedy edge addition ensuring triangle-free
            for u in perm:
                for v in perm:
                    if v >= u:
                        continue
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not is_triangle_free(n, adj):
                continue
            
            if not has_independent_set_k(n, adj, 10):
                best_n = n
                best_edges = edges_from_adj(n, adj)
                found = True
                break
        
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
