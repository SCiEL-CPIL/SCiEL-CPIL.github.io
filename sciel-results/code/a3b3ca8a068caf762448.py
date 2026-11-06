"""Combined program: greedy C4-free construction with local search using bitmasks."""
import time
import random

def _prime_power(p, max_n=50):
    """Find a prime power >= max_n. Return (q, is_prime) where q is the prime power."""
    # Check primes first
    for p in range(max_n, max_n + 200):
        if p < 2:
            continue
        is_p = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_p = False
                break
        if is_p:
            return p, True
    # Check prime powers
    for base in range(2, max_n + 20):
        for power in range(2, 8):
            q = base ** power
            if q >= max_n:
                return q, False
            if q > 1000:
                break
    return max_n, True

def _polarity_graph_46():
    """
    Construct a C4-free graph on 46 vertices using a polarity of PG(2,4).
    PG(2,4) has 21 points. We use a different construction:
    
    Actually, let's use the standard Erdos-Renyi polarity graph on q+1 vertices
    where q=43 (prime), giving 44 vertices, then extend.
    
    Better: use the polarity graph on q=43 (44 vertices) as a seed,
    or construct directly on 46 vertices.
    
    For n=46, we need min degree >= 7. A polarity graph on q+1 vertices
    has degree roughly sqrt(q). For q=43, degree ~ 6-7.
    
    Let's just use the greedy approach with a better seed.
    """
    return None

def _polarity_graph(q):
    """
    Construct a polarity graph on q+1 vertices (for prime q).
    Vertices: {0, 1, ..., q-1} ∪ {∞}
    Edge (i, j) for i,j in {0,...,q-1}: i*j ≡ 1 (mod q)
    Edge (i, ∞): i = 0
    This gives a C4-free graph with degree ~ sqrt(q).
    """
    n = q + 1
    adj = [0] * n
    for i in range(q):
        for j in range(i + 1, q):
            if (i * j) % q == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    # Vertex q is infinity
    # Edge (0, q)
    adj[0] |= (1 << q)
    adj[q] |= (1 << 0)
    return adj, n

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_c4_free(adj, n):
        """Check C4-freeness: no two vertices share 2+ common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def min_degree(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def greedy_construct(n, target):
        """
        Randomized greedy construction: build graph vertex by vertex,
        adding edges that don't create a C4, prioritizing low-degree vertices.
        """
        adj = [0] * n
        degs = [0] * n
        mask = (1 << n) - 1

        # Process vertices in random order
        order = list(range(n))
        random.shuffle(order)

        for u in order:
            # Find all non-neighbors of u
            non_neighbors = (~adj[u]) & mask
            non_neighbors &= ~(1 << u)
            
            nn_list = []
            tmp = non_neighbors
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                nn_list.append(idx)
                tmp ^= lsb
            
            random.shuffle(nn_list)
            
            # Add edges to u that don't create C4
            for v in nn_list:
                if degs[u] >= target:
                    break
                # Check if adding (u,v) creates a C4
                common = adj[u] & adj[v]
                if common.bit_count() < 2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degs[u] += 1
                    degs[v] += 1

        return adj, degs

    def local_search(adj, n, target, max_iters=100):
        """
        Local search: try to increase minimum degree by adding edges
        that don't create C4s.
        """
        degs = [adj[i].bit_count() for i in range(n)]
        mask = (1 << n) - 1

        for _ in range(max_iters):
            if elapsed() > deadline - 2:
                break
            
            min_deg = min(degs)
            if min_deg >= target:
                break
            
            # Find vertices with lowest degree
            low_verts = [i for i in range(n) if degs[i] <= min_deg + 1]
            random.shuffle(low_verts)
            
            improved = False
            for u in low_verts:
                if degs[u] >= target:
                    continue
                
                non_neighbors = (~adj[u]) & mask
                non_neighbors &= ~(1 << u)
                
                nn_list = []
                tmp = non_neighbors
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    nn_list.append(idx)
                    tmp ^= lsb
                
                random.shuffle(nn_list)
                
                for v in nn_list:
                    common = adj[u] & adj[v]
                    if common.bit_count() < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degs[u] += 1
                        degs[v] += 1
                        improved = True
                        if degs[u] >= target:
                            break
                
                if not improved:
                    continue
            
            if not improved:
                break

        return adj, degs

    def edge_repair(adj, n, target, max_iters=50):
        """
        Edge removal local search: if graph has C4s, remove edges to break them
        while trying to keep degrees as high as possible.
        """
        degs = [adj[i].bit_count() for i in range(n)]
        mask = (1 << n) - 1
        
        for _ in range(max_iters):
            if elapsed() > deadline - 2:
                break
            
            if check_c4_free(adj, n):
                break
            
            # Find a C4 and remove an edge from it
            # Find a pair of vertices with >= 2 common neighbors
            found = False
            for i in range(n):
                if found:
                    break
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if common.bit_count() >= 2:
                        # Find two common neighbors k, l
                        c_list = []
                        tmp = common
                        while tmp:
                            lsb = tmp & -tmp
                            idx = lsb.bit_length() - 1
                            c_list.append(idx)
                            tmp ^= lsb
                        k, l = c_list[0], c_list[1]
                        # C4: i-k-j-l-i
                        # Remove one of the four edges to break the C4
                        # Choose the edge whose removal causes least damage
                        candidates = [(i, k), (k, j), (j, l), (l, i)]
                        random.shuffle(candidates)
                        for a, b in candidates:
                            # Check if removing (a,b) helps
                            # Just remove it
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                            degs[a] -= 1
                            degs[b] -= 1
                            found = True
                            break
                        break
        
            if not found:
                break

        return adj, degs

    def try_construct(n, attempts=50):
        target = n - 39
        if target < 1:
            target = 1
        
        best_adj = None
        best_md = -1

        # Try polarity graph seed first (for n=44, use q=43)
        q, is_prime = _prime_power(n - 1)
        if q >= n - 1 and q <= n + 10:
            pol_adj, pol_n = _polarity_graph(q)
            if pol_n == n:
                if check_c4_free(pol_adj, n):
                    md = min_degree(pol_adj, n)
                    if md > best_md:
                        best_md = md
                        best_adj = pol_adj[:]
                    if md < target:
                        pol_adj2, _ = local_search(pol_adj[:], n, target, max_iters=50)
                        if check_c4_free(pol_adj2, n):
                            md2 = min_degree(pol_adj2, n)
                            if md2 > best_md:
                                best_md = md2
                                best_adj = pol_adj2[:]
                            if md2 >= target:
                                edges = []
                                for i in range(n):
                                    for j in range(i + 1, n):
                                        if best_adj[i] & (1 << j):
                                            edges.append([i, j])
                                return n, edges

        # Randomized greedy construction attempts
        for attempt in range(attempts):
            if elapsed() > deadline - 2:
                break

            # 1. Greedy construction
            adj, degs = greedy_construct(n, target)

            # 2. Verify C4-free
            if not check_c4_free(adj, n):
                # Try edge repair
                adj, degs = edge_repair(adj, n, target, max_iters=30)
                if not check_c4_free(adj, n):
                    continue

            md = min(degs)
            if md > best_md:
                best_md = md
                best_adj = adj[:]

            # 3. If min degree is too low, try local search
            if md < target:
                adj2, degs2 = local_search(adj, n, target, max_iters=50)
                if check_c4_free(adj2, n):
                    md2 = min(degs2)
                    if md2 > best_md:
                        best_md = md2
                        best_adj = adj2[:]
                    if md2 >= target:
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if best_adj[i] & (1 << j):
                                    edges.append([i, j])
                        return n, edges

            if md >= target:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return n, edges

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return n, edges
            
        return None

    # Try from largest n down, spending more time on larger n
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if elapsed() > deadline - 5:
            break
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
