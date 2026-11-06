import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe

    N = 123457

    # We want to find a, b such that a+b=N and max(P(a), P(b)) is minimized.
    # This is equivalent to finding the smallest prime K such that there exists
    # a K-smooth a and a K-smooth b with a+b=N.
    # K-smooth means all prime factors are <= K.
    
    # We will iterate through primes K = 2, 3, 5, 7, 11, ...
    # For each K, we generate all K-smooth numbers <= N.
    # Then we check if any pair sums to N.
    
    # To generate K-smooth numbers, we can use a recursive/iterative approach
    # or a simple loop over exponents.
    # Since N is small (123k), the number of smooth numbers is small.
    
    def generate_smooth_numbers(limit, primes):
        """
        Generates all numbers <= limit whose prime factors are in 'primes'.
        Returns a set of such numbers.
        """
        smooths = set([1])
        # We can use a queue or just iterative expansion
        # Since the count is small, a simple recursive generation or
        # iterating over combinations of powers works.
        
        # Efficient way:
        # Start with 1.
        # For each prime p in primes, multiply existing smooths by p^k
        # such that result <= limit.
        
        current_smooths = [1]
        for p in primes:
            new_smooths = []
            for s in current_smooths:
                val = s
                while val <= limit:
                    new_smooths.append(val)
                    val *= p
            # We need to combine these correctly.
            # Actually, it's easier to just generate all combinations.
            # But the number of smooth numbers is small.
            # Let's use a different method:
            # Iterate i from 1 to limit? No, that's O(N).
            # We want to be faster than O(N) if possible, but O(N) is fine for 123k.
            # However, generating the set of smooth numbers is O(S) where S is count of smooths.
            
            # Let's stick to a robust generator.
            pass

    # A better generator for small N:
    def get_smooth_set(limit, max_prime):
        # Sieve to find primes up to max_prime? No, we pass the list of primes.
        # We want numbers composed ONLY of these primes.
        # We can do a DFS.
        smooths = set()
        
        def dfs(current, idx):
            if current > limit:
                return
            smooths.add(current)
            if idx >= len(primes_list):
                return
            p = primes_list[idx]
            next_val = current
            while next_val <= limit:
                dfs(next_val, idx)
                next_val *= p
        
        # primes_list is passed in closure or argument
        # We need to pass the list of primes to use.
        pass

    # Let's implement the check for specific K values.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97...
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We will check K = 2, then K=3, etc.
    # For a given set of primes P_K, generate all P_K-smooth numbers <= N.
    
    def generate_p_smooth_numbers(limit, primes):
        """
        Returns a set of all numbers <= limit with prime factors in primes.
        """
        smooths = set([1])
        # We can iteratively add factors.
        # This specific method works well for small sets of primes.
        # However, a simple DFS is safer.
        
        res = set()
        
        def dfs(val, idx):
            if val > limit:
                return
            res.add(val)
            if idx == len(primes):
                return
            p = primes[idx]
            next_val = val
            while next_val <= limit:
                dfs(next_val, idx)
                next_val *= p
                
        dfs(1, 0)
        return res

    # Check K = 2
    # Primes: [2]
    # Check K = 3
    # Primes: [2, 3]
    # Check K = 5
    # Primes: [2, 3, 5]
    # ...
    
    for i in range(len(primes)):
        current_primes = primes[:i+1]
        max_p = current_primes[-1]
        
        # Check time
        if time.time() > deadline:
            break
            
        smooths = generate_p_smooth_numbers(N, current_primes)
        
        # Check if any pair sums to N
        found = False
        for a in smooths:
            b = N - a
            if b in smooths:
                # Found a pair with max prime factor max_p
                # Since we iterate K in increasing order, this is the minimum.
                return [a, b]
                
    # Fallback: if no smooth pair found with small primes (unlikely for this N),
    # use the sieve method from the original program to be safe.
    # This ensures we always return a valid answer.
    
    lpf = [0] * (N + 1)
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    for p in range(2, N + 1):
        if is_prime[p]:
            for j in range(p, N + 1, p):
                lpf[j] = p
    lpf[1] = 1
    
    best_val = float('inf')
    best_pair = [1, N]
    
    for a in range(1, N // 2 + 1):
        b = N - a
        p_a = lpf[a]
        p_b = lpf[b]
        current_max = p_a if p_a > p_b else p_b
        if current_max < best_val:
            best_val = current_max
            best_pair = [a, b]
            
    return best_pair

