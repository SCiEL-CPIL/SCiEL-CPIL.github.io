import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s for safety
    
    def has_triangle(adj, u, v):
        # adj is a list of int bitsets
        # Check if N(u) and N(v) intersect
        return (adj[u] & adj[v]) != 0

    def max_independent_set_size(adj, n):
        # Exact branch and bound for alpha using bitsets
        # adj: list of int bitsets
        # n: number of vertices
        
        # Order vertices by degree (popcount)
        # For bitsets, we can just iterate
        
        best_alpha = 0
        
        # Recursive helper
        # avail: int bitset of available vertices
        # current_size: size of current independent set
        def search(avail, current_size):
            nonlocal best_alpha
            if avail == 0:
                return
            # Upper bound: current_size + popcount(avail)
            if current_size + bin(avail).count('1') <= best_alpha:
                return
            
            # Heuristic: pick vertex with max degree in induced subgraph
            # To speed up, we can just pick the first one or use a greedy heuristic
            # Let's pick the vertex with max degree in the subgraph induced by avail
            # This requires iterating over set bits of avail
            max_deg = -1
            v = -1
            temp = avail
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                # Degree of u in avail is popcount(adj[u] & avail)
                deg = bin(adj[u] & avail).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = u
                temp -= lsb
            
            if v == -1:
                return

            # Option 1: Include v
            # New avail: avail - {v} - N(v)
            # N(v) in avail is adj[v] & avail
            # {v} is 1 << v
            new_avail = avail & ~( (1 << v) | (adj[v] & avail) )
            search(new_avail, current_size + 1)
            
            if current_size + 1 > best_alpha:
                best_alpha = current_size + 1

            # Option 2: Exclude v
            new_avail = avail & ~(1 << v)
            search(new_avail, current_size)

        search((1 << n) - 1, 0)
        return best_alpha

    def try_construct(n, seed):
        random.seed(seed)
        
        # Initialize empty graph
        adj = [0] * n
        
        # Phase 1: Fill with random edges to maximize density (random maximal triangle-free)
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if not has_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Phase 2: Local Search / ILS
        # We will try to add edges to reduce alpha
        # If stuck, we perturb by removing edges and re-filling
        
        for iteration in range(50):
            if time.time() > deadline:
                return None
            
            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                return adj
            
            # Find an independent set of size >= 10
            # We can use a greedy approach to find a large IS
            is_set = []
            avail = (1 << n) - 1
            while avail:
                # Pick vertex with min degree in avail
                min_deg = n + 1
                v = -1
                temp = avail
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    deg = bin(adj[u] & avail).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        v = u
                    temp -= lsb
                if v == -1:
                    break
                is_set.append(v)
                avail &= ~( (1 << v) | (adj[v] & avail) )
                
            if len(is_set) < 10:
                # Greedy found < 10, but exact said >= 10?
                # If exact alpha >= 10, greedy usually finds something close.
                # If we can't find an IS of size 10, we might be stuck.
                # Perturb: remove some edges and re-add
                break

            # Try to add edges within is_set
            added = False
            for i in range(len(is_set)):
                for j in range(i+1, len(is_set)):
                    u = is_set[i]
                    v = is_set[j]
                    if not (adj[u] & (1 << v)):
                        if not has_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
            if not added:
                # Perturb: remove a random subset of edges
                # This allows escaping local optima
                num_edges_to_remove = random.randint(5, 15)
                all_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            all_edges.append((u, v))
                if all_edges:
                    random.shuffle(all_edges)
                    for u, v in all_edges[:num_edges_to_remove]:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    
                    # Re-fill with random edges
                    edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                    random.shuffle(edges)
                    for u, v in edges:
                        if not has_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

        return None

    # Try n=40
    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_construct(40, seed)
        if res is not None:
            edges = []
            for i in range(40):
                for j in range(i+1, 40):
                    if res[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 40, 'edges': edges}

    # Try n=39
    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_construct(39, seed)
        if res is not None:
            edges = []
            for i in range(39):
                for j in range(i+1, 39):
                    if res[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 39, 'edges': edges}

    # Fallback: Return a small valid graph
    # C5 blow-up with m=4 gives n=20, alpha=8
    n = 20
    adj = [0] * n
    for g1 in range(5):
        for g2 in range(g1+1, 5):
            if (g2 - g1) % 5 in [1, 4]:
                for u in range(g1*4, (g1+1)*4):
                    for v in range(g2*4, (g2+1)*4):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
