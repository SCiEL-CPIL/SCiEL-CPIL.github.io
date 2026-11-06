import time
import random
import math

MAX_VAL = 10000
MAX_SUM = 20000

# Precompute Smallest Prime Factor (SPF) up to 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factors for all numbers up to MAX_SUM
pf = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    pf[i] = get_prime_factors(i)

def calc_prime_set(nums):
    """Returns the set of all prime factors of pairwise sums."""
    primes = set()
    n = len(nums)
    for i in range(n):
        xi = nums[i]
        for j in range(i+1, n):
            s = xi + nums[j]
            primes |= pf[s]
    return primes

def calc_score(nums):
    return len(calc_prime_set(nums))

def greedy_construct():
    """
    Greedily build a set of 12 numbers.
    At each step, choose the number that minimizes the total number of distinct primes.
    """
    current_set = []
    current_primes = set()
    
    # Candidate pool: mix of random and "smooth" numbers
    smooth_candidates = []
    for i in range(1, 10001):
        if len(pf[i]) <= 3:
            smooth_candidates.append(i)
            
    if len(smooth_candidates) < 1000:
        candidates = list(range(1, 10001))
    else:
        candidates = smooth_candidates
        
    for step in range(12):
        best_val = None
        best_score = float('inf')
        best_primes = None
        
        # Sample candidates to keep it fast
        sample_size = 500
        sampled = random.sample(candidates, min(sample_size, len(candidates)))
        
        for val in sampled:
            if val in current_set:
                continue
            
            new_primes = set(current_primes)
            for x in current_set:
                s = x + val
                new_primes |= pf[s]
                
            sc = len(new_primes)
            if sc < best_score:
                best_score = sc
                best_val = val
                best_primes = new_primes
                
        if best_val is None:
            while True:
                val = random.randint(1, MAX_VAL)
                if val not in current_set:
                    break
            best_val = val
            best_primes = set(current_primes)
            for x in current_set:
                best_primes |= pf[x + val]
                
        current_set.append(best_val)
        current_primes = best_primes
        
    return current_set

def simulated_annealing(initial_set, time_limit):
    """
    Run Simulated Annealing starting from initial_set.
    """
    current = initial_set[:]
    current_score = calc_score(current)
    best = current[:]
    best_score = current_score
    
    start_time = time.time()
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, 11)
        old_val = current[i]
        
        new_val = random.randint(1, MAX_VAL)
        
        if new_val == old_val or new_val in current:
            continue
            
        current[i] = new_val
        new_score = calc_score(current)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    current[i] = old_val
            else:
                current[i] = old_val
                
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_score = float('inf')
    
    # 1. Try Greedy Construction
    g_set = greedy_construct()
    g_score = calc_score(g_set)
    if g_score < best_score:
        best_score = g_score
        best_set = g_set[:]
        
    # 2. Run Simulated Annealing from Greedy result
    if time.time() < deadline - 10:
        sa_set, sa_score = simulated_annealing(g_set, deadline - time.time() - 5)
        if sa_score < best_score:
            best_score = sa_score
            best_set = sa_set[:]
            
    # 3. If time remains, try other inits
    if time.time() < deadline - 10:
        r_set = random.sample(range(1, MAX_VAL + 1), 12)
        r_score = calc_score(r_set)
        if r_score < best_score:
            best_score = r_score
            best_set = r_set[:]
            
        if time.time() < deadline - 5:
            sa_set2, sa_score2 = simulated_annealing(r_set, deadline - time.time() - 2)
            if sa_score2 < best_score:
                best_score = sa_score2
                best_set = sa_set2[:]
                
    # 4. If best_score is still high, try structured inits
    if best_score > 5 and time.time() < deadline - 5:
        structured = [2**i for i in range(12)]
        s_score = calc_score(structured)
        if s_score < best_score:
            best_score = s_score
            best_set = structured[:]
            
    return best_set
