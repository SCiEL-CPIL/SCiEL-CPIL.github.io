import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 512
    
    # Precompute LCM matrix (only upper triangle needed, but 512x512 = 262144 entries, fine)
    # Actually, let's just compute on the fly since gcd is fast
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Build adjacency: adj[x] = set of y where lcm(x,y) <= N
    # This is O(N^2) = 262144 operations, very fast
    
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= N:
                adj[a].add(b)
                adj[b].add(a)
    
    # Greedy: pick vertex with max degree, add it, remove non-neighbors
    # But this might not be optimal. Let's try multiple heuristics.
    
    # Heuristic 1: Greedy by degree
    def greedy_by_degree():
        S = set()
        remaining = set(range(1, N + 1))
        while remaining:
            # Pick vertex with max degree in remaining
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            S.add(best_v)
            # Remove best_v and its non-neighbors
            neighbors = adj[best_v]
            remaining = remaining & neighbors
            remaining.discard(best_v)
        return S
    
    # Heuristic 2: Start from divisor sets
    def get_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs
    
    best_set = set()
    
    # Try divisor sets of numbers with many divisors
    for L in range(1, N + 1):
        divs = get_divisors(L)
        if len(divs) > len(best_set):
            best_set = divs.copy()
    
    # Try greedy
    g = greedy_by_degree()
    if len(g) > len(best_set):
        best_set = g
    
    # Local search: try to add elements
    def can_add(S, x):
        for y in S:
            if y != x and lcm(x, y) > N:
                return False
        return True
    
    # Greedy add to best_set
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if x not in best_set and can_add(best_set, x):
                best_set.add(x)
                improved = True
    
    # Try swap improvements
    for _ in range(500):
        if time.time() > deadline:
            break
        improved = False
        S_list = list(best_set)
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            x_rem = S_list[i]
            S_new = best_set - {x_rem}
            # Try to add as many as possible
            added = 0
            for x in range(1, N + 1):
                if x not in S_new and can_add(S_new, x):
                    S_new.add(x)
                    added += 1
            if added > 1:
                best_set = S_new
                improved = True
                break
        if not improved:
            break
    
    return list(best_set)
