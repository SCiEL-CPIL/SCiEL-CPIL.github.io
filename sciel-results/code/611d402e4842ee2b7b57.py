import math
import random
import time
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _coverage_length(primes, residues, max_check=10000):
    """
    Returns the largest L such that every integer 1..L is covered.
    primes: list of primes
    residues: list of residues corresponding to primes
    """
    covered = [False] * (max_check + 1)
    # Mark covered numbers
    for p, r in zip(primes, residues):
        # Start from the smallest number >= 1 that is congruent to r mod p
        # n = k*p + r. We need n >= 1.
        # If r == 0, n can be p, 2p... (since n=0 is not in 1..L)
        # If r > 0, n can be r, p+r...
        start = r if r > 0 else p
        if start > max_check:
            continue
        for n in range(start, max_check + 1, p):
            covered[n] = True
            
    # Find the first gap
    L = 0
    for i in range(1, max_check + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _score_covered_count(primes, residues, N):
    """
    Returns the count of covered integers in 1..N.
    Used as a smoother objective function for optimization.
    """
    count = 0
    # Using a set or boolean array is faster for large N if we just count
    # But for optimization, we can just iterate 1..N and check.
    # To speed up, we can precompute masks or use numpy, but standard lib is safer for portability unless numpy is confirmed. 
    # The prompt says NumPy is available. Let's use it for speed if N is large.
    pass # Will implement in the main block

def _is_valid(primes, residues):
    # Check distinct primes, 0 <= r < p, sum 1/p <= 1
    if len(set(primes)) != len(primes):
        return False
    for p, r in zip(primes, residues):
        if r < 0 or r >= p:
            return False
    total = sum(Fraction(1, p) for p in primes)
    if total > 1:
        return False
    return True

def _seed_solution():
    primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    return [[p, (i + 1) % p] for i, p in enumerate(primes)]

def _local_search(primes, residues, time_limit, initial_temp=1.0, cooling=0.999):
    """
    Optimizes residues for a fixed set of primes.
    Objective: Maximize coverage in 1..N_check.
    """
    N_check = 5000 # Check coverage up to 5000 to guide the search
    best_residues = residues[:]
    best_score = _score_covered_count(primes, best_residues, N_check)
    current_residues = residues[:]
    current_score = best_score
    temp = initial_temp
    
    start_time = time.time()
    random.shuffle(current_residues) # Randomize start? No, keep initial.
    
    while time.time() - start_time < time_limit:
        # Pick a random prime to change residue
        idx = random.randint(0, len(primes) - 1)
        old_r = current_residues[idx]
        new_r = random.randint(0, primes[idx] - 1)
        if new_r == old_r:
            continue
        
        current_residues[idx] = new_r
        new_score = _score_covered_count(primes, current_residues, N_check)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_residues[idx] = new_r
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            current_residues[idx] = old_r
            
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    return best_residues

def _score_covered_count(primes, residues, N):
    # Efficient counting
    # We can use a boolean array
    covered = bytearray(N + 1) # 1 byte per element, fast
    for p, r in zip(primes, residues):
        start = r if r > 0 else p
        if start > N:
            continue
        for n in range(start, N + 1, p):
            covered[n] = 1
    return sum(covered)

def construct():
    deadline = time.time() + 140 # Safety margin
    
    all_primes = _primes_upto(1000)
    
    # Define candidate sets of primes
    # We want sets where sum(1/p) <= 1
    # Strategy: Mix of small and medium primes.
    
    candidate_sets = []
    
    # Set 1: Greedy smallest
    sum_recip = 0
    set1 = []
    for p in all_primes:
        if sum_recip + Fraction(1, p) <= 1:
            set1.append(p)
            sum_recip += Fraction(1, p)
    candidate_sets.append(set1)
    
    # Set 2: Skip 2, take next
    sum_recip = 0
    set2 = []
    for p in all_primes:
        if p == 2: continue
        if sum_recip + Fraction(1, p) <= 1:
            set2.append(p)
            sum_recip += Fraction(1, p)
    candidate_sets.append(set2)
    
    # Set 3: Skip 2,3
    sum_recip = 0
    set3 = []
    for p in all_primes:
        if p in [2, 3]: continue
        if sum_recip + Fraction(1, p) <= 1:
            set3.append(p)
            sum_recip += Fraction(1, p)
    candidate_sets.append(set3)
    
    # Set 4: Randomized greedy with replacement? No, distinct.
    # Let's try a set with 2,5,7...
    sum_recip = 0
    set4 = []
    for p in all_primes:
        if p in [3]: continue
        if sum_recip + Fraction(1, p) <= 1:
            set4.append(p)
            sum_recip += Fraction(1, p)
    candidate_sets.append(set4)
    
    best_L = 0
    best_classes = _seed_solution()
    
    # Evaluate seed
    seed_primes = [c[0] for c in best_classes]
    seed_residues = [c[1] for c in best_classes]
    best_L = _coverage_length(seed_primes, seed_residues, 10000)
    
    # Try optimizing each candidate set
    for primes in candidate_sets:
        if time.time() > deadline - 10:
            break
            
        # Initialize residues randomly
        residues = [random.randint(0, p - 1) for p in primes]
        
        # Run local search / SA
        time_alloc = min(20, deadline - time.time() - 5)
        if time_alloc <= 0:
            continue
            
        opt_residues = _local_search(primes, residues, time_alloc)
        
        # Evaluate exact L for this solution
        L = _coverage_length(primes, opt_residues, 10000)
        
        if L > best_L:
            best_L = L
            best_classes = [[p, r] for p, r in zip(primes, opt_residues)]
            
    return best_classes
