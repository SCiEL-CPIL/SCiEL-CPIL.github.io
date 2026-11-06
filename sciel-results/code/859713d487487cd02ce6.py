"""
K4-free graph maximizer (no independent 6-set) via Max-SAT-style simulated annealing.
Combines bitset-based fast constraint checking with a global energy landscape search.
"""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def _build_adj_bits(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def _clebsch_edges():
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                d = bin(i ^ j).count('1')
                if d == 1 or d == 3:
                    edges.append([i, j])
        return edges

    # Base graph: Clebsch (n=16)
    best_n = 16
    best_edges = _clebsch_edges()

    # Fast exact K4 check
    def _has_k4(adj, n):
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common_ab = adj[a] & adj[b]
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                c_mask = common_ab
                while c_mask:
                    c = (c_mask & -c_mask).bit_length() - 1
                    c_mask &= c_mask - 1
                    if c <= b:
                        continue
                    common_abc = common_ab & adj[c]
                    common_abc &= ~(1 << a)
                    common_abc &= ~(1 << b)
                    common_abc &= ~(1 << c)
                    if common_abc:
                        return True
        return False

    # Fast exact independent 6-set check
    def _has_ind6(adj, n):
        # Complement adjacency
        full_mask = (1 << n) - 1
        comp = [full_mask ^ adj[i] ^ (1 << i) for i in range(n)]
        
        # Check for clique of size 6 in complement
        # Use recursive branching with bitsets
        def _clique_search(vertices_mask, size, adj_list):
            if size == 0:
                return True
            if size > vertices_mask.bit_count():
                return False
            if size == 1:
                return vertices_mask != 0
            
            # Pick a vertex with high degree in the subgraph
            # For speed, just iterate
            m = vertices_mask
            while m:
                b = m & (-m)
                v = b.bit_length() - 1
                m ^= b
                
                # Common neighbors of v in the remaining mask
                common = adj_list[v] & vertices_mask
                common &= ~(1 << v)
                
                if _clique_search(common, size - 1, adj_list):
                    return True
            return False

        return _clique_search((1 << n) - 1, 6, comp)

    def _is_valid(adj, n):
        if _has_k4(adj, n):
            return False
        if _has_ind6(adj, n):
            return False
        return True

    # Verify base
    adj_base = _build_adj_bits(best_n, best_edges)
    if not _is_valid(adj_base, best_n):
        # Fallback: empty graph on 5 vertices
        best_n = 5
        best_edges = []

    # Simulated Annealing for larger n
    def _energy(adj, n):
        """Count violations: K4s + Ind6s (approximate count for speed)"""
        # Count K4s (exact)
        k4_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common_ab = adj[a] & adj[b]
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                c_mask = common_ab
                while c_mask:
                    c = (c_mask & -c_mask).bit_length() - 1
                    c_mask &= c_mask - 1
                    if c <= b:
                        continue
                    common_abc = common_ab & adj[c]
                    common_abc &= ~(1 << a)
                    common_abc &= ~(1 << b)
                    common_abc &= ~(1 << c)
                    k4_count += common_abc.bit_count()
        
        # Count Ind6s (sampled for speed)
        ind6_count = 0
        full_mask = (1 << n) - 1
        comp = [full_mask ^ adj[i] ^ (1 << i) for i in range(n)]
        
        # Sample some 6-sets to estimate
        samples = 50
        for _ in range(samples):
            combo = random.sample(range(n), 6)
            valid = True
            for i in range(6):
                for j in range(i + 1, 6):
                    if (adj[combo[i]] >> combo[j]) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                ind6_count += 1
        
        # Scale ind6 count to estimate total
        total_6sets = 1.0
        for i in range(6):
            total_6sets *= (n - i)
        total_6sets /= 720.0
        est_ind6 = ind6_count * (total_6sets / samples)
        
        return k4_count * 10 + est_ind6 * 5

    def _anneal(target_n, time_limit):
        t0 = time.time()
        random.seed(12345 + target_n)
        
        # Initialize with random graph
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_energy = _energy(adj, target_n)
        best_energy = current_energy
        best_adj = adj[:]
        
        temp = 100.0
        cooling = 0.999
        
        while time.time() - t0 < time_limit:
            # Pick random edge to flip
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_energy = _energy(adj, target_n)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_adj = adj[:]
            else:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            temp *= cooling
            if temp < 1e-6:
                temp = 1e-6
        
        return best_adj, best_energy

    # Try to find valid graphs for increasing n
    for target_n in range(17, 36):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        # Try a few annealing runs
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            
            adj, energy = _anneal(target_n, time_limit=10.0)
            
            if energy == 0:
                # Verify exactly
                if _is_valid(adj, target_n):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
            elif energy < 1.0:
                # Close to valid, try local repair
                adj_rep = adj[:]
                for step in range(100):
                    if time.time() > deadline - 5.0:
                        break
                    if _is_valid(adj_rep, target_n):
                        edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if (adj_rep[i] >> j) & 1:
                                    edges.append([i, j])
                        best_n = target_n
                        best_edges = edges
                        found = True
                        break
                    # Flip random edge
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (adj_rep[u] >> v) & 1:
                        adj_rep[u] &= ~(1 << v)
                        adj_rep[v] &= ~(1 << u)
                    else:
                        adj_rep[u] |= (1 << v)
                        adj_rep[v] |= (1 << u)
                if found:
                    break
        
        if not found:
            # Heuristic: if we failed here, likely to fail for larger n
            # But try one more step with more aggressive search
            if target_n >= 20:
                break

    return {'n': best_n, 'edges': best_edges}
