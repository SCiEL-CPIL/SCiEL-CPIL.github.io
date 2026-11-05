import random
import itertools
import time

# Pre-compute popcount for all masks up to 2^22
# 2^22 = 4,194,304. List of 4M integers. Memory ~ 32MB. Time ~ 1-2s.
# This is a significant algorithmic change: Pre-computation vs On-the-fly.
MAX_N = 22
MAX_MASK = 1 << MAX_N
# We only need popcount for masks up to 2^22.
# But adj[v] & mask can be any subset of 22 bits.
# So we need popcount for all 2^22 values.
try:
    POPCOUNT = [0] * MAX_MASK
    for i in range(1, MAX_MASK):
        POPCOUNT[i] = POPCOUNT[i >> 1] + (i & 1)
except MemoryError:
    POPCOUNT = None

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead

    def get_popcount(x):
        if POPCOUNT is not None and x < MAX_MASK:
            return POPCOUNT[x]
        return bin(x).count('1')

    def fast_check(n, adj, samples=500):
        """
        Randomized check. Returns True if violation found.
        """
        for _ in range(samples):
            k = random.randint(6, n)
            subset = set(random.sample(range(n), k))
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                deg = get_popcount(adj[v] & mask)
                degs.append(deg)
            
            if len(set(degs)) == 1:
                return True
        return False

    def full_check(n, adj):
        """
        Exhaustive check. Returns True if valid.
        """
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in comb:
                    deg = get_popcount(adj[v] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    def optimize(n, time_limit):
        """
        Local search with pre-computed popcount.
        """
        start_time = time.time()
        
        # Try multiple restarts with different densities
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            p = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_adj = adj[:]
            
            # Initial check
            if not fast_check(n, current_adj, 100):
                if full_check(n, current_adj):
                    return current_adj

            # Local Search
            # We use a "walk" strategy: flip an edge, if it reduces violations (or keeps same), keep it.
            # But we don't have a fast violation counter.
            # Instead, we just do random flips and check periodically.
            
            for _ in range(1000):
                if time.time() - start_time > time_limit:
                    break
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge (u, v)
                has_edge = (current_adj[u] & (1 << v)) != 0
                if has_edge:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                
                # Check
                if not fast_check(n, current_adj, 50):
                    if full_check(n, current_adj):
                        return current_adj

        return None

    # Try n=22
    adj_22 = optimize(22, 30)
    if adj_22 is not None:
        edges = []
        for i in range(22):
            for j in range(i + 1, 22):
                if (adj_22[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 22, 'edges': edges}

    # Try n=21
    adj_21 = optimize(21, 40)
    if adj_21 is not None:
        edges = []
        for i in range(21):
            for j in range(i + 1, 21):
                if (adj_21[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 21, 'edges': edges}

    # Try n=20
    adj_20 = optimize(20, 30)
    if adj_20 is not None:
        edges = []
        for i in range(20):
            for j in range(i + 1, 20):
                if (adj_20[i] & (1 << j)) != 0:
                    edges.append([i, j])
        return {'n': 20, 'edges': edges}

    # Fallback: Return a small valid graph
    # A graph with 5 vertices is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}
