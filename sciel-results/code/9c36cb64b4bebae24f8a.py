import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False  # already exists
        
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
        
        # 2. For all w in N(v) \ {u}: pair (u,w) gains v as common neighbor
        #    Need popcount(adj[u] & adj[w]) <= 1 (since v not in adj[u] yet, and v in adj[w])
        w = v
        mask_w = adj[v]
        while mask_w:
            lsb = mask_w & (-mask_w)
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            if w == u:
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False
        
        # 3. For all w in N(u) \ {v}: pair (v,w) gains u as common neighbor
        #    Need popcount(adj[v] & adj[w]) <= 1
        mask_w = adj[u]
        while mask_w:
            lsb = mask_w & (-mask_w)
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
        
    def local_search(adj):
        """Iteratively add any possible edge until no more can be added."""
        changed = True
        while changed:
            changed = False
            # Try all pairs; prefer pairs with fewer common neighbors first
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        common = popcount(adj[u] & adj[v])
                        if common <= 2:
                            candidates.append((common, u, v))
            candidates.sort()
            for _, u, v in candidates:
                if not ((adj[u] >> v) & 1):
                    if can_add(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
        return adj

    def greedy_construct():
        """Build a K2,3-free graph greedily by trying to add edges."""
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add(adj, u, v):
                add_edge(adj, u, v)
        return adj
    
    def simulated_annealing(adj_init, max_steps=50000):
        """Optimize a graph using simulated annealing with edge-addition focus."""
        adj = list(adj_init)
        best_adj = list(adj)
        best_count = edge_count(adj)
        current_count = best_count
        
        T = 3.0
        T_min = 0.05
        cooling = 0.999
        
        step = 0
        while step < max_steps:
            if time.time() > deadline:
                break
            
            # 80% chance to try adding an edge, 20% to try removing
            if random.random() < 0.8:
                # Try adding a random non-existent edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                if not ((adj[u] >> v) & 1):
                    if can_add(adj, u, v):
                        add_edge(adj, u, v)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = list(adj)
            else:
                # Try removing a random existing edge
                # Find an existing edge
                found = False
                for _ in range(20):  # try up to 20 times to find an edge
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (adj[u] >> v) & 1:
                        found = True
                        break
                
                if found:
                    remove_edge(adj, u, v)
                    current_count -= 1
                    # Accept with probability based on temperature
                    if random.random() < min(1.0, T / (T + 0.5)):
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = list(adj)
                    else:
                        add_edge(adj, u, v)
                        current_count += 1
            
            step += 1
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_adj, best_count

    # Multi-start: try several greedy constructions and optimize each
    best_adj = None
    best_count = 0
    
    # First, a deterministic greedy
    adj0 = greedy_construct()
    adj0 = local_search(adj0)
    c0 = edge_count(adj0)
    if c0 > best_count:
        best_count = c0
        best_adj = list(adj0)
    
    # Try multiple random seeds for greedy + SA
    num_starts = 25  # Increased for more exploration
    for seed in range(num_starts):
        if time.time() > deadline - 12:
            break
        
        random.seed(seed * 12345 + 678)
        adj_greedy = greedy_construct()
        adj_greedy = local_search(adj_greedy)
        c_greedy = edge_count(adj_greedy)
        
        if c_greedy > best_count:
            best_count = c_greedy
            best_adj = list(adj_greedy)
        
        # Run SA on this with moderate steps
        remaining = deadline - time.time()
        if remaining > 20:
            steps = min(30000, int(remaining * 200))
            adj_sa, c_sa = simulated_annealing(adj_greedy, max_steps=steps)
            adj_sa = local_search(adj_sa)
            c_sa = edge_count(adj_sa)
            if c_sa > best_count:
                best_count = c_sa
                best_adj = list(adj_sa)
        
        # Also run SA on the current best if we have time
        remaining = deadline - time.time()
        if remaining > 25 and best_adj is not None:
            steps = min(25000, int(remaining * 150))
            adj_sa2, c_sa2 = simulated_annealing(best_adj, max_steps=steps)
            adj_sa2 = local_search(adj_sa2)
            c_sa2 = edge_count(adj_sa2)
            if c_sa2 > best_count:
                best_count = c_sa2
                best_adj = list(adj_sa2)
    
    # Final SA polish on the best
    if best_adj is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        steps = min(50000, int(remaining * 300))
        adj_final, c_final = simulated_annealing(best_adj, max_steps=steps)
        adj_final = local_search(adj_final)
        c_final = edge_count(adj_final)
        if c_final > best_count:
            best_count = c_final
            best_adj = list(adj_final)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
