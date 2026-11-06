import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    def greedy_alpha(g_adj, n):
        """Returns a lower bound on the independence number."""
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            # Find vertex with min degree in remaining subgraph
            min_deg = n + 1
            v = -1
            r = remaining
            while r:
                v_bit = r & (-r)
                idx = v_bit.bit_length() - 1
                deg = bin(g_adj[idx] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    v = idx
                r ^= v_bit
            
            if v == -1:
                break
                
            count += 1
            # Remove v and its neighbors
            remaining &= ~(g_adj[v] | (1 << v))
            
        return count

    def exact_alpha(g_adj, n):
        """Returns the exact independence number (max clique in complement)."""
        # Use a simple branch and bound
        # H[i] is non-neighbors in G
        mask_all = (1 << n) - 1
        h_adj = [0] * n
        for i in range(n):
            h_adj[i] = mask_all ^ g_adj[i] ^ (1 << i)
        
        best = 0
        
        def dfs(candidates, size):
            nonlocal best
            if size + len(bin(candidates).count('1')) <= best:
                return
            if candidates == 0:
                if size > best:
                    best = size
                return
            
            # Heuristic: pick vertex with max degree in candidates
            # To speed up, we can just pick the first one, but max degree is better
            max_deg = -1
            v = -1
            r = candidates
            while r:
                v_bit = r & (-r)
                idx = v_bit.bit_length() - 1
                deg = bin(h_adj[idx] & candidates).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                r ^= v_bit
            
            if v == -1:
                return

            v_bit = 1 << v
            
            # Include v
            dfs(candidates & h_adj[v], size + 1)
            
            # Exclude v
            dfs(candidates & ~v_bit, size)

        dfs(mask_all, 0)
        return best

    def local_search(n):
        # Initialize with random triangle-free graph
        g_adj = [0] * n
        mask_all = (1 << n) - 1
        
        # Add random edges
        for _ in range(n * n):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if g_adj[u] & (1 << v):
                continue
            common = g_adj[u] & g_adj[v]
            if common == 0:
                g_adj[u] |= (1 << v)
                g_adj[v] |= (1 << u)
        
        current_alpha = greedy_alpha(g_adj, n)
        
        T = 1.0
        # Run local search
        while time.time() < deadline:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            if g_adj[u] & (1 << v):
                # Remove edge
                new_g_adj = g_adj[:]
                new_g_adj[u] ^= (1 << v)
                new_g_adj[v] ^= (1 << u)
                new_alpha = greedy_alpha(new_g_adj, n)
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < pow(2.718, -delta / max(T, 0.01)):
                    g_adj = new_g_adj
                    current_alpha = new_alpha
                    if current_alpha <= 9:
                        # Verify with exact
                        if exact_alpha(g_adj, n) <= 9:
                            return g_adj
            else:
                # Add edge
                common = g_adj[u] & g_adj[v]
                if common != 0:
                    continue
                
                new_g_adj = g_adj[:]
                new_g_adj[u] |= (1 << v)
                new_g_adj[v] |= (1 << u)
                new_alpha = greedy_alpha(new_g_adj, n)
                delta = new_alpha - current_alpha
                if delta < 0 or random.random() < pow(2.718, -delta / max(T, 0.01)):
                    g_adj = new_g_adj
                    current_alpha = new_alpha
                    if current_alpha <= 9:
                        if exact_alpha(g_adj, n) <= 9:
                            return g_adj
            
            T *= 0.999
            if T < 0.01:
                T = 0.01
        return None

    # Try n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try a few restarts
        for _ in range(10):
            if time.time() > deadline:
                break
            g_adj = local_search(n)
            if g_adj is not None:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if g_adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

    # Fallback to Andrasfai n=26
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
