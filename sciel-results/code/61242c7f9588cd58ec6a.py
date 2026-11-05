import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    # Helper: Check if bitmask represents an independent set
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        # Common neighbors must form an independent set
        return is_independent(common, adj)

    # Exact B&B for largest triangle-free induced subgraph
    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound
        # Heuristic: Order by degree descending, add if valid
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        if best[0] == n:
            return n

        # B&B
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            # Upper bound
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v
            # Check if S U {v} is triangle-free
            # This requires N(v) & S to be independent
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    random.seed(42)
    best_adj = [0] * n
    best_score = n # Max possible is n

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial candidates
    # 1. Turan Graph T(18,3)
    adj_turan = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if (i // 6) != (j // 6):
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
                
    # 2. Random Greedy
    def random_greedy():
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    candidates = [adj_turan, random_greedy(), random_greedy(), random_greedy()]
    
    for adj in candidates:
        if time.time() > deadline - 5.0:
            break
        score = largest_tf(adj, time_limit=deadline - 2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Initial candidate score: {best_score}")

    # Local Search
    adj = best_adj[:]
    
    while time.time() < deadline - 2.0:
        # Randomly perturb
        if random.random() < 0.5:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            # Remove edge
            if random.random() < 0.3:
                continue # Skip removal if graph is sparse?
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and ((adj[u] >> v) & 1):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

        # Evaluate
        score = largest_tf(adj, time_limit=deadline - 2.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            print(f"Improved! Score: {best_score} at {time.time()-start_time:.1f}s")
        else:
            # Revert if not improved, to stay in good region
            # Or keep with some probability for diversity?
            # Reverting is safer to not drift to bad regions.
            adj = best_adj[:]

    edges = get_edge_list(best_adj)
    return edges
