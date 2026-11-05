import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Helper: extract edges from adjacency bitmasks
    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Fast K4 check with early exit
    def has_k4(n, adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Check if any two vertices in common are adjacent
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if (adj[w] & c) != 0:
                            return True
        return False
    
    # Check for independent set of size 6
    def has_ind6(n, adj):
        mask = (1 << n) - 1
        # Compute complement adjacency (non-edges, excluding self)
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        
        # Quick check: if any vertex has complement degree < 5, no ind6 can include it
        # But we need to check all possibilities
        
        # Use branch and bound to find clique of size 6 in complement
        def find_clique(candidates, size, current):
            if size == 0:
                return True
            if candidates.bit_count() < size:
                return False
            
            # Get lowest set bit
            v = (candidates & -candidates).bit_length() - 1
            new_cands = candidates & ~(1 << v) & comp_adj[v]
            
            if find_clique(new_cands, size - 1, current | (1 << v)):
                return True
            
            return find_clique(candidates & ~(1 << v), size, current)
        
        return find_clique((1 << n) - 1, 6, 0)
    
    def is_valid(n, adj):
        if has_k4(n, adj):
            return False
        if has_ind6(n, adj):
            return False
        return True
    
    # Penalty function for local search
    def get_penalty(n, adj):
        # Count K4s (or just check existence for speed)
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        k4_count += (adj[w] & c).bit_count()
        k4_count //= 6
        
        ind6_exists = has_ind6(n, adj)
        
        if k4_count > 0:
            return k4_count * 10
        if ind6_exists:
            return 100
        return 0
    
    # Local search using simulated annealing
    def local_search(n, adj, time_limit):
        current_adj = adj[:]
        current_penalty = get_penalty(n, current_adj)
        
        if current_penalty == 0:
            return current_adj
        
        temp = 10.0
        cooling = 0.998
        best_adj = current_adj[:]
        best_penalty = current_penalty
        
        while time.time() < time_limit:
            # Random edge flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            if (current_adj[u] >> v) & 1:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            new_penalty = get_penalty(n, current_adj)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = current_adj[:]
                    if best_penalty == 0:
                        return best_adj
            else:
                delta = new_penalty - current_penalty
                if random.random() < pow(2.71828, -delta / max(temp, 0.001)):
                    current_penalty = new_penalty
                else:
                    # Revert
                    if (current_adj[u] >> v) & 1:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                    else:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
            
            temp *= cooling
        
        return best_adj if best_penalty <= current_penalty else current_adj
    
    # Try to construct the Schläfli graph (n=35)
    # The Schläfli graph is a strongly regular graph with parameters (35, 16, 7, 8)
    # It is K4-free and has independence number 5.
    # We'll use a known construction or try to find it via search.
    
    # For now, let's try various n values with local search
    # Start with larger n and work down
    
    best_n = 0
    best_adj = None
    
    # Try n=35 first (Schläfli graph)
    # Use multiple restarts
    for target_n in [35, 34, 33, 32, 31, 30, 28, 25, 22, 20, 18, 17]:
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 5 if target_n >= 30 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random graph
            adj = [0] * target_n
            # Use a density that might work
            # For K4-free, we want sparse graphs
            # For no independent 6-set, we want dense graphs
            # Balance: try different densities
            density = 0.3 + 0.1 * (restart % 3)
            
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            slice_end = time.time() + 8
            if slice_end > deadline - 5:
                slice_end = deadline - 5
            
            result_adj = local_search(target_n, adj, slice_end)
            
            if is_valid(target_n, result_adj):
                best_n = target_n
                best_adj = result_adj
                found = True
                break
        
        if found:
            break
    
    # Fallback: Paley graph on 17 vertices (known to be K4-free with alpha=5)
    if best_n == 0 or best_adj is None:
        n = 17
        adj = [0] * n
        squares = {x * x % n for x in range(1, n)}
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_n = n
        best_adj = adj
    
    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
