import time
import random
from itertools import combinations, product as iter_product

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def is_smooth(n, target_primes):
    """Check if n is smooth with respect to target_primes."""
    if n <= 1:
        return True
    return factors_cache[n].issubset(target_primes)

def find_compatible_residues(target_primes, max_val=10000):
    """
    Find residue classes modulo M=product(target_primes) such that
    for any two numbers a,b with a≡r1, b≡r2 (mod M), a+b is P-smooth.
    Returns list of (residue, list_of_numbers_in_that_residue_class)
    """
    M = 1
    for p in target_primes:
        M *= p
    
    # For each pair of residues (r1, r2), check if all numbers a≡r1, b≡r2 have a+b smooth
    # Actually, a+b mod M = (r1+r2) mod M. If a+b is not smooth, then it has a prime factor not in P.
    # But a+b can be any number ≡ (r1+r2) mod M. So we need to check if ALL numbers ≡ (r1+r2) mod M 
    # are P-smooth? No, that's too strong. We need that for the specific a,b we pick, a+b is smooth.
    
    # Better approach: For a given residue r, collect all numbers ≤ max_val with that residue.
    # Then do local search on which 12 numbers to pick from these residue classes.
    
    # Actually, the key insight: if we pick all numbers from residue classes r1, r2, ..., rk,
    # then any sum a+b where a≡ri, b≡rj will be ≡ (ri+rj) mod M.
    # We need to ensure that for each pair (ri, rj) we use, the numbers we pick have sums that are smooth.
    
    # Let's just return the candidate pool grouped by residue for later use.
    residue_to_numbers = {}
    for num in range(1, max_val + 1):
        r = num % M
        if r not in residue_to_numbers:
            residue_to_numbers[r] = []
        residue_to_numbers[r].append(num)
    
    return M, residue_to_numbers

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 999
    best_arr = list(range(1, 13))
    
    # Strategy: Try small prime sets and use modular arithmetic to guide search
    # For a prime set P, find residue classes where sums tend to be P-smooth
    # Then pick 12 numbers from compatible residue classes
    
    prime_sets_to_try = [
        frozenset([2, 3, 5, 7, 11, 13]),      # k=6
        frozenset([2, 3, 5, 7, 11, 13, 17]),   # k=7
        frozenset([2, 3, 5, 7, 11, 13, 17, 19]), # k=8
    ]
    
    for target_primes in prime_sets_to_try:
        if time.time() > deadline:
            break
        
        M, residue_to_numbers = find_compatible_residues(target_primes, 10000)
        
        # Find which residues have "good" sums: for residue r, check if numbers ≡ r mod M 
        # tend to produce smooth sums when paired with other "good" residues
        
        # Heuristic: A residue r is "good" if there exist at least 12 numbers ≡ r mod M 
        # that are themselves P-smooth (or at least their pairwise sums are likely smooth)
        
        # Actually, let's identify "promising" residues: those where many numbers are P-smooth
        promising_residues = []
        for r, nums in residue_to_numbers.items():
            smooth_count = sum(1 for n in nums if is_smooth(n, target_primes))
            if smooth_count >= 12:
                promising_residues.append((r, smooth_count, nums))
        
        if not promising_residues:
            continue
        
        # Sort by smooth_count descending
        promising_residues.sort(key=lambda x: -x[1])
        
        # Try combinations of 1-3 promising residues
        # For each combination, pick 12 numbers from the union of those residue classes
        # and do local search
        
        best_for_this_p = 999
        best_arr_for_this_p = None
        
        for num_residues in [1, 2, 3]:
            if time.time() > deadline:
                break
            
            # Try top few residue combinations
            top_residues = promising_residues[:min(10, len(promising_residues))]
            
            for combo in combinations(top_residues, num_residues):
                if time.time() > deadline:
                    break
                
                # Pool: all numbers from these residue classes that are P-smooth
                pool = []
                seen = set()
                for r, sc, nums in combo:
                    for n in nums:
                        if n not in seen and is_smooth(n, target_primes):
                            pool.append(n)
                            seen.add(n)
                
                if len(pool) < 12:
                    continue
                
                # Local search: pick 12 from pool to minimize bad primes in sums
                random.shuffle(pool)
                current_arr = pool[:12]
                
                def calc_cost(arr):
                    bad_primes = set()
                    for i in range(12):
                        for j in range(i+1, 12):
                            s = arr[i] + arr[j]
                            for p in factors_cache[s]:
                                if p not in target_primes:
                                    bad_primes.add(p)
                    return len(bad_primes)
                
                current_cost = calc_cost(current_arr)
                
                if current_cost == 0:
                    best_for_this_p = len(target_primes)
                    best_arr_for_this_p = current_arr[:]
                    break
                
                # Hill climbing with restarts
                best_local_cost = current_cost
                best_local_arr = current_arr[:]
                
                for restart in range(5):
                    if time.time() > deadline:
                        break
                    
                    # Restart with random selection
                    random.shuffle(pool)
                    current_arr = pool[:12]
                    current_cost = calc_cost(current_arr)
                    
                    if current_cost == 0:
                        best_for_this_p = len(target_primes)
                        best_arr_for_this_p = current_arr[:]
                        break
                    
                    # Hill climbing
                    stuck = False
                    while not stuck and time.time() < deadline:
                        stuck = True
                        for idx in range(12):
                            old_val = current_arr[idx]
                            for new_val in random.sample(pool, min(50, len(pool))):
                                if new_val == old_val:
                                    continue
                                if new_val in current_arr:
                                    continue
                                
                                current_arr[idx] = new_val
                                new_cost = calc_cost(current_arr)
                                
                                if new_cost < current_cost:
                                    current_cost = new_cost
                                    stuck = False
                                    if new_cost < best_local_cost:
                                        best_local_cost = new_cost
                                        best_local_arr = current_arr[:]
                                    if new_cost == 0:
                                        best_for_this_p = len(target_primes)
                                        best_arr_for_this_p = current_arr[:]
                                        break
                                else:
                                    current_arr[idx] = old_val
                            
                            if best_for_this_p == len(target_primes):
                                break
                        if best_for_this_p == len(target_primes):
                            break
                    
                    if best_for_this_p == len(target_primes):
                        break
                
                if best_for_this_p == len(target_primes):
                    break
            
            if best_for_this_p == len(target_primes):
                break
        
        if best_for_this_p < best_score:
            best_score = best_for_this_p
            best_arr = best_arr_for_this_p
        
        if best_score == len(target_primes):
            # Can't do better than the size of the prime set we used
            if len(target_primes) <= 6:
                return best_arr
    
    # Fallback: general local search
    if best_score == 999:
        best_arr = list(range(1, 13))
        best_score = count_distinct_primes(best_arr)
    
    # Final improvement attempt
    random.seed(42)
    current_arr = best_arr[:]
    current_score = best_score
    
    steps = 0
    while time.time() < deadline and steps < 2000:
        steps += 1
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        if new_val in current_arr and new_val != old_val:
            continue
        current_arr[idx] = new_val
        score = count_distinct_primes(current_arr)
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best_arr = current_arr[:]
        else:
            current_arr[idx] = old_val
    
    return best_arr
