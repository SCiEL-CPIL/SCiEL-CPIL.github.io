import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_VAL = 10000
    # We will search for cliques in the range 1..N
    # Sums will be in 2..2N
    
    # Precompute smallest prime factor for fast factorization
    MAX_S = 20000 # Max sum possible is 19999
    spf = list(range(MAX_S + 1))
    for i in range(2, int(MAX_S**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_S + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def is_smooth(n, max_prime):
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p > max_prime:
                return False
            while n % p == 0:
                n //= p
        return True

    def find_clique_brone_kerbosch(N, P_max):
        """
        Find a clique of size 12 in graph 1..N where edge (u,v) exists if u+v is P_max-smooth.
        Uses Bron-Kerbosch with pivot.
        """
        # Precompute smoothness for sums up to 2N
        smooth = [False] * (2 * N + 1)
        for s in range(2, 2 * N + 1):
            smooth[s] = is_smooth(s, P_max)
        
        # Build adjacency sets
        adj = [set() for _ in range(N + 1)]
        for u in range(1, N + 1):
            for v in range(u + 1, N + 1):
                if smooth[u + v]:
                    adj[u].add(v)
                    adj[v].add(u)
        
        best_clique = []
        
        def bronk(R, P, X):
            nonlocal best_clique
            
            if time.time() > deadline:
                return
                
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = R.copy()
                return
            
            # Choose pivot u from P union X to minimize branching
            P_union_X = P.union(X)
            if not P_union_X:
                return
            
            pivot = None
            max_deg = -1
            for u in P_union_X:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if pivot is None:
                candidates = list(P)
            else:
                candidates = list(P.difference(adj[pivot]))
                
            for v in candidates:
                new_R = R + [v]
                new_P = P.intersection(adj[v])
                new_X = X.intersection(adj[v])
                bronk(new_R, new_P, new_X)
                
                P.discard(v)
                X.add(v)
                
                if len(best_clique) >= 12:
                    return

        initial_P = set(range(1, N + 1))
        bronk([], initial_P, set())
        
        return best_clique

    # Try to find clique for P=2 (Score 1)
    clique = find_clique_brone_kerbosch(100, 2)
    if len(clique) >= 12:
        return clique[:12]

    # Try P=3 (Score 2)
    clique = find_clique_brone_kerbosch(200, 3)
    if len(clique) >= 12:
        return clique[:12]

    # Try P=5 (Score 3)
    clique = find_clique_brone_kerbosch(300, 5)
    if len(clique) >= 12:
        return clique[:12]

    # Try P=7 (Score 4)
    clique = find_clique_brone_kerbosch(400, 7)
    if len(clique) >= 12:
        return clique[:12]

    # Fallback: Local search
    factor_cache = {}
    def factorize(n):
        if n in factor_cache:
            return factor_cache[n]
        if n <= 1:
            factor_cache[n] = set()
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        factor_cache[n] = factors
        return factors

    def evaluate(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)

    if len(clique) > 6:
        seed_list = list(clique)
        while len(seed_list) < 12:
            val = random.randint(1, 10000)
            if val not in seed_list:
                seed_list.append(val)
        best = seed_list
    else:
        best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        
    best_score = evaluate(best)
    
    random.seed(42)
    candidate_pool = list(set(random.randint(1, 10000) for _ in range(2000)))
    for m in [2, 3, 4, 5, 6, 10, 12]:
        for k in range(1, 10000 // m + 1):
            candidate_pool.append(k * m)
    candidate_pool = list(set(candidate_pool))
    random.shuffle(candidate_pool)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = best[i]
            for new_val in candidate_pool:
                if new_val == old_val or new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break

    return best
