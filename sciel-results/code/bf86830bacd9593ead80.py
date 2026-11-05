import time
import random
from itertools import combinations

def get_primes(n):
    sieve = [True] * (n + 1)
    if n >= 0:
        sieve[0] = False
    if n >= 1:
        sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Primes up to 20000
PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)

def factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            factors.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            all_primes.update(factorize(s))
    return len(all_primes)

# Precompute prime factors for all sums up to 20000
MAX_SUM = 20000
prime_factors_cache = [set() for _ in range(MAX_SUM + 1)]
for s in range(2, MAX_SUM + 1):
    prime_factors_cache[s] = factorize(s)

def generate_smooth_numbers(max_val, primes):
    nums = {1}
    for p in primes:
        current = list(nums)
        for x in current:
            y = x
            while y * p <= max_val:
                y *= p
                nums.add(y)
    return sorted(nums)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Strong precomputed candidate: powers of 2 and 3 combinations
    # Sums of elements from {1,2,4,8,16,32,64,128,256,512,1024,2048} are 
    # of form 2^a + 2^b = 2^min(a,b) * (1 + 2^|a-b|)
    # The odd factors (1+2^k) for k=1..11: 3,5,9(3^2),17,33(3*11),65(5*13),129(3*43),257,513(3^3*19),1025(5^2*41),2049(3*683)
    # So primes: 2,3,5,11,13,17,41,43,19,683 -> 10 primes. Not great.
    
    # Better: use {2,3}-smooth numbers where sums are also {2,3}-smooth
    # Known good set from inspiration 2: [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    # Let's verify: sums like 1+2=3, 1+4=5 (not 2,3-smooth!)
    # Wait, 5 is not in {2,3}. So that set has prime 5.
    
    # Let's use the odd numbers approach from inspiration 1: [1,3,5,7,9,11,13,15,17,19,21,23]
    # Sums are even. 1+3=4=2^2, 1+5=6=2*3, 1+7=8=2^3, 1+9=10=2*5, 1+11=12=2^2*3,
    # 1+13=14=2*7, 1+15=16=2^4, 1+17=18=2*3^2, 1+19=20=2^2*5, 1+21=22=2*11, 1+23=24=2^3*3
    # 3+5=8, 3+7=10, 3+9=12, 3+11=14, 3+13=16, 3+15=18, 3+17=20, 3+19=22, 3+21=24, 3+23=26=2*13
    # 5+7=12, 5+9=14, 5+11=16, 5+13=18, 5+15=20, 5+17=22, 5+19=24, 5+21=26, 5+23=28=2^2*7
    # 7+9=16, 7+11=18, 7+13=20, 7+15=22, 7+17=24, 7+19=26, 7+21=28, 7+23=30=2*3*5
    # 9+11=20, 9+13=22, 9+15=24, 9+17=26, 9+19=28, 9+21=30, 9+23=32
    # 11+13=24, 11+15=26, 11+17=28, 11+19=30, 11+21=32, 11+23=34=2*17
    # 13+15=28, 13+17=30, 13+19=32, 13+21=34, 13+23=36=2^2*3^2
    # 15+17=32, 15+19=34, 15+21=36, 15+23=38=2*19
    # 17+19=36, 17+21=38, 17+23=40=2^3*5
    # 19+21=40, 19+23=42=2*3*7
    # 21+23=44=2^2*11
    # Primes: 2,3,5,7,11,13,17,19 -> 8 primes
    
    # Let's try to find a set with fewer primes using the lattice approach
    # Strategy: pick small primes P, Q, and find numbers a_i = u_i*P + v_i*Q
    # such that a_i + a_j = (u_i+u_j)*P + (v_i+v_j)*Q has few prime factors
    
    # Alternative: use the clique-finding approach with a better candidate set
    # Focus on {2,3,5}-smooth numbers and find a clique of size 12 where all sums are {2,3,5}-smooth
    
    candidate_prime_sets = [
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7],
        [2, 3, 5],
        [2, 3],
    ]
    
    best_set = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]  # fallback: 8 primes
    best_score = count_distinct_primes(best_set)  # should be 8
    
    for prime_set in candidate_prime_sets:
        if time.time() > deadline:
            break
        
        prime_set_set = set(prime_set)
        
        # Generate smooth numbers
        smooth_nums = generate_smooth_numbers(10000, prime_set)
        
        # Limit candidates for efficiency
        max_candidates = min(len(smooth_nums), 300)
        candidates = smooth_nums[:max_candidates]
        
        # Build compatibility graph
        adj = {a: set() for a in candidates}
        for i in range(len(candidates)):
            a = candidates[i]
            for j in range(i + 1, len(candidates)):
                b = candidates[j]
                s = a + b
                if s <= MAX_SUM:
                    pf = prime_factors_cache[s]
                    if pf.issubset(prime_set_set):
                        adj[a].add(b)
                        adj[b].add(a)
        
        # Find clique of size 12 using backtracking with degree-based ordering
        def find_clique(min_size, candidates_list, adj):
            if len(candidates_list) < min_size:
                return None
            
            candidates_list = sorted(candidates_list, key=lambda x: len(adj[x]), reverse=True)
            
            clique = []
            
            def backtrack(remaining):
                if len(clique) == min_size:
                    return True
                if not remaining:
                    return False
                if len(remaining) < min_size - len(clique):
                    return False
                
                for idx, node in enumerate(remaining):
                    if all(node in adj[c] for c in clique):
                        clique.append(node)
                        new_remaining = remaining[idx+1:]
                        if backtrack(new_remaining):
                            return True
                        clique.pop()
                
                return False
            
            if backtrack(candidates_list):
                return clique
            return None
        
        clique = find_clique(12, candidates, adj)
        
        if clique:
            score = count_distinct_primes(clique)
            if score < best_score:
                best_score = score
                best_set = clique[:]
                if best_score <= 3:
                    break
        
        # Local search refinement
        if time.time() > deadline:
            break
        
        for restart in range(3):
            if time.time() > deadline:
                break
            
            if len(candidates) >= 12:
                init_set = random.sample(candidates, 12)
            else:
                init_set = candidates[:]
                while len(init_set) < 12:
                    init_set.append(random.randint(1, 10000))
            
            s = init_set[:]
            score = count_distinct_primes(s)
            
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, 11)
                old_val = s[i]
                new_val = random.choice(candidates)
                if new_val in s:
                    continue
                
                s[i] = new_val
                new_score = count_distinct_primes(s)
                
                if new_score <= score:
                    score = new_score
                else:
                    s[i] = old_val
            
            if score < best_score:
                best_score = score
                best_set = s[:]
                if best_score <= 3:
                    break
    
    # Final local search on the best set found
    if time.time() < deadline - 5.0:
        arr = best_set[:]
        cur_score = count_distinct_primes(arr)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5.0:
                    break
                # Try replacing with smooth numbers
                smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7, 11])
                for v in smooth_nums:
                    if v in arr:
                        continue
                    arr[i] = v
                    new_score = count_distinct_primes(arr)
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                        break
                    else:
                        arr[i] = best_set[i]
                if time.time() > deadline - 5.0:
                    break
            if cur_score < best_score:
                best_score = cur_score
                best_set = arr[:]
    
    return best_set
