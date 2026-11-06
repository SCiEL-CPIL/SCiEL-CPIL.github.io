import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
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

    def evaluate_set(nums):
        """Calculate the number of distinct prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(all_primes)

    def evaluate_with_prime_set(nums):
        """Return (prime_count, prime_set) for a set of numbers."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(all_primes), all_primes

    # Strategy 1: Exact Clique Search for Prime Set {2, 3}
    # We look for 12 numbers x_i such that x_i + x_j is only divisible by 2 and 3.
    # This implies x_i + x_j = 2^a * 3^b.
    # If we set x_i = 6*k_i + 3, then x_i + x_j = 6(k_i + k_j) + 6 = 6(k_i + k_j + 1).
    # We need k_i + k_j + 1 to be of the form 2^a * 3^b.
    
    max_k = 10000 // 6
    # Generate all 2^a * 3^b <= 2*max_k + 1
    smooth_sums = set()
    for a in range(15):
        pow2 = 1 << a
        if pow2 > 2 * max_k + 1:
            break
        for b in range(15):
            val = pow2 * (3 ** b)
            if val > 2 * max_k + 1:
                break
            smooth_sums.add(val)
            
    # k_i must be such that x_i <= 10000 => 6*k_i + 3 <= 10000 => k_i <= 1666
    # Also k_i >= 0.
    
    # Build adjacency: for each k, store which k' have k+k'+1 in smooth_sums
    # This is expensive, so we'll use a smarter approach
    
    # Heuristic Clique Search with better candidate selection
    found_clique = False
    best_clique = None
    
    # Precompute: for each k in [0, 500], find all k' in [0, 1666] such that k+k'+1 is smooth
    # Then find a clique using greedy + backtracking
    
    # Try multiple random starting points
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        # Start with a small k (sums are smaller, more likely to be smooth)
        start_k = random.randint(0, 200)
        clique = [start_k]
        clique_set = {start_k}
        
        # Greedy: at each step, pick the k' that maximizes connections to existing clique
        for _ in range(11):
            best_candidate = None
            best_score = -1
            best_connections = 0
            
            # Sample a subset of candidates to keep it fast
            sample_size = min(100, 1667 - len(clique_set))
            candidates_to_try = random.sample(range(0, 1667), sample_size)
            
            for k_prime in candidates_to_try:
                if k_prime in clique_set:
                    continue
                # Check if k_prime connects to all in clique
                ok = True
                connections = 0
                for k in clique:
                    if (k + k_prime + 1) in smooth_sums:
                        connections += 1
                    else:
                        ok = False
                        break
                if ok:
                    # Score by how many other potential candidates it connects to
                    # (approximate: just use connections count)
                    if connections > best_score:
                        best_score = connections
                        best_candidate = k_prime
            
            if best_candidate is None:
                # Try a broader search
                for k_prime in range(0, 1667):
                    if k_prime in clique_set:
                        continue
                    ok = True
                    for k in clique:
                        if (k + k_prime + 1) not in smooth_sums:
                            ok = False
                            break
                    if ok:
                        best_candidate = k_prime
                        break
            
            if best_candidate is None:
                break
            clique.append(best_candidate)
            clique_set.add(best_candidate)
        
        if len(clique) == 12:
            found_clique = True
            best_clique = [6 * k + 3 for k in clique]
            break

    if found_clique:
        # Verify
        if len(set(best_clique)) == 12 and all(1 <= x <= 10000 for x in best_clique):
            if evaluate_set(best_clique) == 2:
                return best_clique

    # Strategy 2: Improved Local Search with targeted candidate generation
    # Start with diverse structured initial guesses
    
    # Generate multiple diverse initializations
    initializations = []
    
    # Pattern 1: Multiples of 6 (sums are even, divisible by 2)
    initializations.append([6 * i for i in range(1, 13)])
    
    # Pattern 2: 6k+3 (sums are 6(k1+k2+1), divisible by 6)
    initializations.append([6 * i + 3 for i in range(0, 12)])
    
    # Pattern 3: 6k+1 (sums are 6(k1+k2)+2, even)
    initializations.append([6 * i + 1 for i in range(0, 12)])
    
    # Pattern 4: 6k+5 (sums are 6(k1+k2)+10, even)
    initializations.append([6 * i + 5 for i in range(0, 12)])
    
    # Pattern 5: Powers of 2 times small constants
    initializations.append([2**i for i in range(4, 16)])
    
    # Pattern 6: 3*2^k
    initializations.append([3 * (2**i) for i in range(0, 12)])
    
    # Pattern 7: Random
    initializations.append(random.sample(range(1, 10001), 12))
    
    # Pattern 8: Small numbers with structure
    initializations.append(list(range(1, 13)))
    
    # Pattern 9: Fibonacci-like
    fib = [1, 2]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    initializations.append(fib[:12])
    
    best_set = None
    best_score = float('inf')
    best_prime_set = None
    
    for init in initializations:
        if time.time() > deadline - 5:
            break
        
        # Ensure distinct and valid
        curr_set = []
        seen = set()
        for x in init:
            if 1 <= x <= 10000 and x not in seen:
                curr_set.append(x)
                seen.add(x)
        while len(curr_set) < 12:
            x = random.randint(1, 10000)
            if x not in seen:
                curr_set.append(x)
                seen.add(x)
        
        curr_score, curr_primes = evaluate_with_prime_set(curr_set)
        
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
            best_prime_set = curr_primes.copy()
            if best_score <= 2:
                return best_set

        # Targeted Local Search
        # Key insight: generate candidates that, when added to existing elements,
        # produce sums with prime factors already in the current prime set
        
        improved = True
        stall_count = 0
        max_stall = 50
        
        while improved and time.time() < deadline - 5:
            improved = False
            stall_count += 1
            
            if stall_count > max_stall:
                break
            
            # Pick an index to change - prioritize indices that contribute most "new" primes
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate targeted candidates
            candidates = set()
            
            # 1. Small perturbations
            for delta in [-5, -3, -1, 1, 3, 5]:
                nv = old_val + delta
                if 1 <= nv <= 10000:
                    candidates.add(nv)
            
            # 2. Targeted: for each other element, try values that make sum have few new primes
            for j in range(12):
                if j == idx:
                    continue
                other = curr_set[j]
                # Try values x such that x + other has prime factors mostly in curr_primes
                # Sample some candidates around values that would work
                for trial in range(5):
                    # Random value, but check if sum has few new primes
                    x = random.randint(1, 10000)
                    s = x + other
                    if s <= MAX_SUM:
                        pf = get_prime_factors(s)
                        new_primes = pf - curr_primes
                        if len(new_primes) <= 2:
                            candidates.add(x)
            
            # 3. Structured candidates based on current prime set
            # If 2 is in prime set, try even numbers
            if 2 in curr_primes:
                for _ in range(10):
                    candidates.add(random.randint(1, 5000) * 2)
            
            # If 3 is in prime set, try multiples of 3
            if 3 in curr_primes:
                for _ in range(10):
                    candidates.add(random.randint(1, 3333) * 3)
            
            # If 5 is in prime set, try multiples of 5
            if 5 in curr_primes:
                for _ in range(10):
                    candidates.add(random.randint(1, 2000) * 5)
            
            # 4. Random fallback
            candidates.add(random.randint(1, 10000))
            
            # 5. Values that are sums of small smooth numbers
            for a in range(2, 20):
                for b in range(2, 20):
                    v = a * b
                    if 1 <= v <= 10000:
                        candidates.add(v)
            
            # Evaluate candidates and pick the best improvement
            best_candidate = None
            best_new_score = curr_score
            
            for new_val in candidates:
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in curr_set:
                    continue
                
                new_set = curr_set.copy()
                new_set[idx] = new_val
                new_score, new_primes = evaluate_with_prime_set(new_set)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_candidate = new_val
                    best_candidate_primes = new_primes
            
            if best_candidate is not None:
                curr_set[idx] = best_candidate
                curr_score = best_new_score
                curr_primes = best_candidate_primes
                improved = True
                stall_count = 0
                
                if curr_score < best_score:
                    best_score = curr_score
                    best_set = curr_set.copy()
                    best_prime_set = curr_primes.copy()
                    if best_score <= 2:
                        return best_set
    
    # Final polish: try to remove one prime at a time
    if best_set is not None and best_prime_set is not None and time.time() < deadline - 2:
        # For each prime in the set, try to eliminate it
        primes_to_try = list(best_prime_set)
        random.shuffle(primes_to_try)
        
        for target_prime in primes_to_try:
            if time.time() > deadline - 2:
                break
            
            # Try to modify the set so that no sum is divisible by target_prime
            # This is a constraint satisfaction problem - use local search
            work_set = best_set.copy()
            work_score, work_primes = evaluate_with_prime_set(work_set)
            
            improved = True
            stall = 0
            while improved and stall < 20 and time.time() < deadline - 2:
                improved = False
                stall += 1
                
                idx = random.randint(0, 11)
                old_val = work_set[idx]
                
                # Try to find a new value such that for all j, (new_val + work_set[j]) % target_prime != 0
                # OR (new_val + work_set[j]) is not divisible by target_prime
                found = False
                for _ in range(50):
                    nv = random.randint(1, 10000)
                    if nv in work_set:
                        continue
                    
                    # Check if this eliminates target_prime
                    ok = True
                    for j in range(12):
                        if j == idx:
                            continue
                        s = nv + work_set[j]
                        if s <= MAX_SUM and target_prime in get_prime_factors(s):
                            # This sum still has target_prime, but that's okay if other sums don't
                            pass
                    
                    # Actually, we need to check if target_prime is still in the union
                    test_set = work_set.copy()
                    test_set[idx] = nv
                    test_score, test_primes = evaluate_with_prime_set(test_set)
                    
                    if target_prime not in test_primes and test_score <= work_score:
                        work_set = test_set
                        work_score = test_score
                        work_primes = test_primes
                        improved = True
                        found = True
                        break
                
                if not found and stall > 10:
                    break
            
            if target_prime not in work_primes and work_score < best_score:
                best_score = work_score
                best_set = work_set.copy()
                best_prime_set = work_primes.copy()
    
    return best_set
# EVOLVE-BLOCK-END
