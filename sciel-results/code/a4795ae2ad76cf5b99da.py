import time
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    N = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # Generate primes up to MAX_SUM
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(MAX_SUM)
    prime_set = set(primes)
    
    # Precompute smallest prime factor (SPF) for fast factorization
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def is_smooth(n, allowed_primes):
        # Check if all prime factors of n are in allowed_primes
        while n > 1:
            p = spf[n]
            if p not in allowed_primes:
                return False
            while n % p == 0:
                n //= p
        return True

    # We will try sets of primes of increasing size
    # Start with smallest primes
    # k=2: {2,3}
    # k=3: {2,3,5}, {2,3,7}, {2,5,7}, ...
    # We prioritize sets with smaller primes as they are more likely to form smooth sums
    
    # Generate candidate prime sets
    # For k=2, 3, 4, 5
    # We only need to check a limited number of combinations
    # Specifically, combinations of the first few primes
    
    # Backtracking function to find a clique of size N in the graph defined by P
    def find_clique(allowed_primes):
        # Generate smooth sums
        smooth_sums = []
        for s in range(2, MAX_SUM + 1):
            if is_smooth(s, allowed_primes):
                smooth_sums.append(s)
        
        # If there are too few smooth sums, it's impossible
        # Each pair (a,b) needs a unique sum? No, sums can repeat.
        # But a and b are distinct.
        
        # Build adjacency structure implicitly
        # For each a in 1..MAX_VAL, find neighbors b
        # To speed up, we only consider 'active' nodes that have high connectivity
        # Let's compute degrees first
        degrees = [0] * (MAX_VAL + 1)
        neighbors = [set() for _ in range(MAX_VAL + 1)]
        
        # This loop is O(MAX_VAL * len(smooth_sums))
        # len(smooth_sums) for {2,3} is ~100.
        # 10000 * 100 = 1,000,000 ops. Very fast.
        for s in smooth_sums:
            # a + b = s => b = s - a
            # a must be in [1, MAX_VAL]
            # b must be in [1, MAX_VAL] and b != a
            min_a = max(1, s - MAX_VAL)
            max_a = min(MAX_VAL, s - 1)
            for a in range(min_a, max_a + 1):
                b = s - a
                if b != a:
                    # Add edge a-b
                    if b not in neighbors[a]:
                        neighbors[a].add(b)
                        degrees[a] += 1
                    if a not in neighbors[b]:
                        neighbors[b].add(a)
                        degrees[b] += 1
        
        # Filter nodes with degree >= N-1
        candidates = [i for i in range(1, MAX_VAL + 1) if degrees[i] >= N - 1]
        
        if len(candidates) < N:
            return None
            
        # Sort candidates by degree descending
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        
        # Backtracking search
        # We want to find a clique of size N
        # We can restrict the search to the first M candidates to avoid explosion
        # If a clique exists, it's likely among high-degree nodes
        # Let's try with top 500 candidates
        search_space = candidates[:500]
        
        # Map candidate to index for faster lookup? Or just use sets.
        # We'll use a recursive backtracking
        found_clique = []
        
        def backtrack(start_idx, current_clique):
            if len(current_clique) == N:
                found_clique.extend(current_clique)
                return True
            
            if start_idx >= len(search_space):
                return False
            
            # Pruning: if not enough nodes left to complete the clique
            if len(search_space) - start_idx < N - len(current_clique):
                return False
                
            for i in range(start_idx, len(search_space)):
                node = search_space[i]
                # Check if node connects to all in current_clique
                valid = True
                for c in current_clique:
                    if c not in neighbors[node]:
                        valid = False
                        break
                if valid:
                    current_clique.append(node)
                    if backtrack(i + 1, current_clique):
                        return True
                    current_clique.pop()
            
            return False
        
        if backtrack(0, []):
            return found_clique[:N]
        return None

    # Iterate over prime sets
    # k=2
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
    # We try combinations of small primes
    
    # Try k=2
    p2 = [2, 3]
    res = find_clique(p2)
    if res: return res
    
    if time.time() > deadline: return list(range(1, 13))
    
    # Try k=3
    # Combinations of {2,3,5}, {2,3,7}, {2,5,7}, {3,5,7}...
    # Prioritize those with 2
    k3_combos = [
        [2, 3, 5],
        [2, 3, 7],
        [2, 3, 11],
        [2, 5, 7],
        [2, 3, 13],
        [2, 5, 11],
        [3, 5, 7]
    ]
    for combo in k3_combos:
        res = find_clique(combo)
        if res: return res
        if time.time() > deadline: return list(range(1, 13))
        
    # Try k=4
    # Combinations of {2,3,5,7}, {2,3,5,11}, ...
    k4_combos = [
        [2, 3, 5, 7],
        [2, 3, 5, 11],
        [2, 3, 7, 11],
        [2, 3, 5, 13],
        [2, 5, 7, 11]
    ]
    for combo in k4_combos:
        res = find_clique(combo)
        if res: return res
        if time.time() > deadline: return list(range(1, 13))
        
    # If all fail, return a heuristic set
    # The trivial set 1..12 has cost 9.
    # Maybe we can find a set with cost < 9 by random search?
    # Given the time constraint, we return the trivial set if search fails.
    # But we should try one more thing:
    # Maybe a set of powers of 2? 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048.
    # Sums: 1+2=3, 1+4=5, 2+4=6(2,3), 4+8=12(2,3)...
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
    # Too many primes.
    
    # Let's stick to the search result. If it fails, return 1..12.
    return list(range(1, 13))

# EVOLVE-BLOCK-END
