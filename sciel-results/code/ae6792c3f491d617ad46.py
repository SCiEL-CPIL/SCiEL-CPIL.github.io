import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (evaluator kills at 150s)
    
    def check_valid(adj, n):
        """Check if graph has no K4 and no independent set of 6."""
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Need to check if any edge exists in common
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[w] & common:
                            return False  # Found K4
        # Check independent set of 6 using backtracking
        def dfs_is(candidates, count):
            if count == 6:
                return True
            if not candidates:
                return False
            if count + candidates.bit_count() < 6:
                return False
            # Pick vertex with min degree in candidates for better pruning
            best_v = -1
            best_deg = n + 1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & candidates).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                return False
            v = best_v
            # Exclude v
            if dfs_is(candidates & ~(1 << v), count):
                return True
            # Include v
            new_cands = candidates & ~(1 << v) & ~(adj[v])
            if dfs_is(new_cands, count + 1):
                return True
            return False
        if dfs_is((1 << n) - 1, 0):
            return False
        return True

    def count_k4(adj, n):
        """Count K4s using bitmask."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        if w > v and (adj[w] & common):
                            # Count edges in common subgraph
                            c2 = common
                            ec = 0
                            while c2:
                                lsb2 = c2 & -c2
                                w2 = lsb2.bit_length() - 1
                                c2 ^= lsb2
                                ec += (adj[w2] & common).bit_count()
                            count += ec // 2
                            break
        return count

    def has_is6(adj, n):
        """Check for independent set of size 6."""
        def dfs_is(candidates, count):
            if count == 6:
                return True
            if not candidates:
                return False
            if count + candidates.bit_count() < 6:
                return False
            # Pick vertex with min degree
            best_v = -1
            best_deg = n + 1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj[v] & candidates).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                return False
            v = best_v
            if dfs_is(candidates & ~(1 << v), count):
                return True
            new_cands = candidates & ~(1 << v) & ~(adj[v])
            if dfs_is(new_cands, count + 1):
                return True
            return False
        return dfs_is((1 << n) - 1, 0)

    def generate_paley(n):
        """Generate Paley graph if n is a prime power ≡ 1 (mod 4)."""
        # Check if n is prime
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(math.isqrt(x)) + 1):
                if x % i == 0:
                    return False
            return True
        if not is_prime(n) or n % 4 != 1:
            return None
        # Compute quadratic residues
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def generate_complement_paley(n):
        """Generate complement of Paley graph."""
        adj = generate_paley(n)
        if adj is None:
            return None
        # Complement
        for i in range(n):
            adj[i] = ((1 << n) - 1) & ~(1 << i) & ~adj[i]
        return adj

    def local_search(adj, n, time_limit=10.0):
        """Run local search to fix K4s and IS6 violations."""
        start = time.time()
        T = 1.0
        T_min = 0.05
        alpha = 0.9995
        
        def get_cost(a):
            k4 = count_k4(a, n)
            is6 = 1 if has_is6(a, n) else 0
            return 100 * k4 + is6
        
        current_cost = get_cost(adj)
        best_adj = adj[:]
        best_cost = current_cost
        
        while time.time() - start < time_limit:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_cost = get_cost(adj)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                    if best_cost == 0:
                        break
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            T = max(T * alpha, T_min)
        
        return best_adj, best_cost

    def try_extend(adj_old, n_old, n_new, time_limit=8.0):
        """Try to extend a valid graph from n_old to n_new vertices."""
        # Start with the old graph plus random new vertices
        adj = adj_old[:] + [0] * (n_new - n_old)
        for i in range(n_old, n_new):
            for j in range(i + 1, n_new):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            # Connect new vertices to old ones randomly
            for j in range(n_old):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Run local search
        adj, cost = local_search(adj, n_new, time_limit)
        return adj, cost == 0

    # Main algorithm
    best_n = 0
    best_adj = None
    
    # Try Paley graphs for known prime powers ≡ 1 mod 4
    paley_candidates = [5, 13, 17, 29, 37]
    for n in paley_candidates:
        if time.time() > deadline - 15.0:
            break
        adj = generate_paley(n)
        if adj is not None:
            if check_valid(adj, n):
                best_n = n
                best_adj = adj[:]
            # Also try complement
            adj_c = generate_complement_paley(n)
            if adj_c is not None and check_valid(adj_c, n):
                if n > best_n:
                    best_n = n
                    best_adj = adj_c[:]
    
    # If we have a valid graph, try to extend it
    if best_adj is not None:
        for n_try in range(best_n + 1, 36):
            if time.time() > deadline - 10.0:
                break
            success = False
            for attempt in range(3):
                if time.time() > deadline - 5.0:
                    break
                adj_new, valid = try_extend(best_adj, best_n, n_try, time_limit=6.0)
                if valid:
                    best_n = n_try
                    best_adj = adj_new
                    success = True
                    break
            if not success:
                break
    
    # If no valid graph found, try random initialization with local search
    if best_adj is None:
        for n_try in range(20, 36):
            if time.time() > deadline - 10.0:
                break
            for attempt in range(5):
                if time.time() > deadline - 5.0:
                    break
                # Random graph
                adj = [0] * n_try
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                adj, cost = local_search(adj, n_try, time_limit=5.0)
                if cost == 0:
                    best_n = n_try
                    best_adj = adj
                    break
            if best_n >= n_try:
                continue
            else:
                break
    
    # Final verification and conversion
    if best_adj is not None and best_n > 0:
        # Verify
        if not check_valid(best_adj, best_n):
            # Try to fix
            best_adj, cost = local_search(best_adj, best_n, time_limit=5.0)
            if cost != 0:
                # Fall back to smallest valid
                for n_fallback in range(best_n, 1, -1):
                    adj_f = best_adj[:n_fallback]
                    if check_valid(adj_f, n_fallback):
                        best_n = n_fallback
                        best_adj = adj_f
                        break
    
    if best_adj is None or best_n == 0:
        # Return trivial valid graph
        best_n = 1
        best_adj = [0]
    
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
