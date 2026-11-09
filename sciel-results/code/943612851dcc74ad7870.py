import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for evaluation overhead

    def make_paley(n):
        """Create a Paley-type graph on n vertices using quadratic residues."""
        # Find a prime p >= n
        def is_prime(x):
            if x < 2: return False
            if x < 4: return True
            if x % 2 == 0 or x % 3 == 0: return False
            i = 5
            while i * i <= x:
                if x % i == 0 or x % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        p = n
        while not is_prime(p):
            p += 1
        if p < 5:
            p = 5
        
        # Quadratic residues mod p
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        
        # Build adjacency as bitmask
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in residues or (p - diff) in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def count_bad_6_bitmask(n, adj):
        """Count 6-vertex subsets inducing regular subgraphs using bitmask ops."""
        count = 0
        for comb in itertools.combinations(range(n), 6):
            # Compute degrees within the subset using bitmasks
            degs = []
            for i in comb:
                d = 0
                for j in comb:
                    if i != j and (adj[i] >> j) & 1:
                        d += 1
                degs.append(d)
            if len(set(degs)) == 1:
                count += 1
        return count

    def check_validity_bitmask(n, adj):
        """Check if graph has no regular induced subgraph of size >= 6."""
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                degs = []
                for i in comb:
                    d = 0
                    for j in comb:
                        if i != j and (adj[i] >> j) & 1:
                            d += 1
                    degs.append(d)
                if len(set(degs)) == 1:
                    return False
        return True

    def get_violating_edges_bitmask(n, adj, comb_list=None):
        """Return set of edges in violating 6-sets."""
        if comb_list is None:
            comb_list = list(itertools.combinations(range(n), 6))
        violating = set()
        for comb in comb_list:
            degs = []
            for i in comb:
                d = 0
                for j in comb:
                    if i != j and (adj[i] >> j) & 1:
                        d += 1
                degs.append(d)
            if len(set(degs)) == 1:
                for idx_i in range(6):
                    for idx_j in range(idx_i + 1, 6):
                        u, v = comb[idx_i], comb[idx_j]
                        violating.add((min(u, v), max(u, v)))
        return violating

    def local_search_bitmask(n, max_iters=200, seed_adj=None):
        """Simulated annealing with bitmask representation."""
        if seed_adj is not None:
            adj = list(seed_adj)
        else:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        comb_list = list(itertools.combinations(range(n), 6))
        current_score = count_bad_6_bitmask(n, adj)
        if current_score == 0:
            if check_validity_bitmask(n, adj):
                return adj
        
        best_adj = list(adj)
        best_score = current_score
        temp = 1.0
        cooling = 0.999
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose edge to flip
            if current_score > 0 and it % 15 == 0:
                viol_edges = get_violating_edges_bitmask(n, adj, comb_list)
                if viol_edges:
                    u, v = random.choice(list(viol_edges))
                else:
                    u, v = random.choice(edges)
            else:
                u, v = random.choice(edges)
            
            # Flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            new_score = count_bad_6_bitmask(n, adj)
            
            delta = new_score - current_score
            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = list(adj)
                if current_score == 0:
                    if check_validity_bitmask(n, adj):
                        return adj
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            
            temp *= cooling
        
        if best_score < current_score:
            adj = best_adj
            current_score = best_score
            if current_score == 0:
                if check_validity_bitmask(n, adj):
                    return adj
        return None

    def try_find_graph(target_n, time_budget, num_restarts=5):
        """Try multiple strategies to find valid graph on target_n vertices."""
        t_start = time.time()
        
        # Strategy 1: Paley initialization
        if time.time() - t_start < time_budget * 0.4:
            paley_adj = make_paley(target_n)
            result = local_search_bitmask(target_n, max_iters=150, seed_adj=paley_adj)
            if result is not None:
                return result
        
        # Strategy 2: Random restarts
        for r in range(num_restarts):
            if time.time() > t_start + time_budget:
                break
            result = local_search_bitmask(target_n, max_iters=100)
            if result is not None:
                return result
        
        return None

    # Try n=21 first with more time
    result = try_find_graph(21, 40, num_restarts=3)
    if result is not None:
        return {'n': 21, 'edges': [[u, v] for u in range(21) for v in range(u + 1, 21) if (result[u] >> v) & 1]}

    # Try n=20
    result = try_find_graph(20, 40, num_restarts=5)
    if result is not None:
        return {'n': 20, 'edges': [[u, v] for u in range(20) for v in range(u + 1, 20) if (result[u] >> v) & 1]}

    # Try n=18
    result = try_find_graph(18, 20, num_restarts=5)
    if result is not None:
        return {'n': 18, 'edges': [[u, v] for u in range(18) for v in range(u + 1, 18) if (result[u] >> v) & 1]}

    # Try n=15
    result = try_find_graph(15, 15, num_restarts=5)
    if result is not None:
        return {'n': 15, 'edges': [[u, v] for u in range(15) for v in range(u + 1, 15) if (result[u] >> v) & 1]}

    # Fallback: small random graph
    n = 10
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    adj[0] = 0
    for j in range(1, n):
        adj[j] &= ~(1 << 0)
    
    comb_10 = list(itertools.combinations(range(n), 6))
    for _ in range(50):
        if count_bad_6_bitmask(n, adj) == 0:
            if check_validity_bitmask(n, adj):
                return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]}
        u = random.randint(1, n - 1)
        v = random.randint(1, n - 1)
        if u != v:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)

    # Ultimate fallback: K6 minus an edge
    edges = []
    for u in range(6):
        for v in range(u + 1, 6):
            if not (u == 0 and v == 1):
                edges.append([u, v])
    return {'n': 6, 'edges': edges}
