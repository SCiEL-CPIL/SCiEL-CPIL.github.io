import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precompute primes up to a reasonable limit for mutation pool
    # We need primes up to 10^18, but for mutation we can sample from a wide range.
    # However, generating primes up to 10^18 is impossible.
    # We will generate small primes for the initial population and use "jump" mutations 
    # for large primes if needed, but typically optimal solutions use small-to-medium primes.
    # Let's generate primes up to 10^6 for a robust pool.
    
    def sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.array([i for i, v in enumerate(is_prime) if v])

    primes_pool = sieve(1000000) # Primes up to 10^6
    # Add some larger primes for diversity? 
    # Actually, for k~15, product is ~10^18. 
    # If we use primes > 10^6, we can only have few of them.
    # The optimal solution likely uses a mix or mostly small primes.
    # Let's stick to primes up to 10^5 to ensure we can fit ~15 of them.
    # 2*3*5*...*47 (15th prime) is ~6e17. 
    # So primes up to 50 are critical.
    # Let's make a pool of primes up to 10000.
    primes_pool = sieve(10000)
    
    # 2. Vectorized Score Function
    def calculate_score(prime_list):
        k = len(prime_list)
        if k == 0:
            return 0
        
        # Generate all 2^k masks
        # masks shape: (2^k, k)
        masks = np.array([list(map(int, format(i, '0{}b'.format(k))[::-1])) for i in range(1 << k)], dtype=np.int8)
        
        # Calculate divisor values
        # log_values = masks @ log(prime_list)
        # We need exact values to sort correctly, but logs are good for sorting if no overflow.
        # However, 10^18 is within float64 precision for sorting? 
        # 10^18 has 19 digits. Float64 has ~15-16 digits precision. 
        # Sorting by log might lead to ties or wrong order for close values.
        # We must use exact integer arithmetic or high precision.
        # But NumPy int64 handles 10^18.
        # Multiplication: masks (2^k, k) @ primes (k,)
        # This is a matrix-vector product.
        # primes_arr = np.array(prime_list, dtype=np.int64)
        # divisors = masks @ primes_arr  # This works if no overflow, but masks are 0/1.
        # Wait, masks @ primes gives sum of primes in mask? NO.
        # Divisor is PRODUCT of primes in mask.
        # So we need: prod(prime_list[i] for i where mask[i]==1)
        # This is not a simple linear operation.
        
        # Fast way to compute products for all subsets:
        # Use iterative doubling.
        # d[0] = 1
        # d[1] = p0
        # d[2] = p1
        # d[3] = p0*p1
        # ...
        # We can build this using broadcasting.
        
        # Let's use a loop over primes to build the array of divisors.
        # Start with [1]
        divs = np.array([1], dtype=np.int64)
        for p in prime_list:
            # New divisors are old divisors * p
            new_divs = divs * p
            divs = np.concatenate([divs, new_divs])
            
        # Now divs has 2^k elements.
        # We need to sort them and track which mask they came from?
        # The mask is the index in the construction order?
        # No, the construction order is not the mask index directly if we just append.
        # Let's track masks explicitly.
        
        # Reset
        masks = np.array([0], dtype=np.int32)
        divs = np.array([1], dtype=np.int64)
        
        for idx, p in enumerate(prime_list):
            bit = 1 << idx
            new_divs = divs * p
            new_masks = masks | bit
            divs = np.concatenate([divs, new_divs])
            masks = np.concatenate([masks, new_masks])
            
        # Sort divs, keep masks aligned
        # np.argsort is stable? We just need the order.
        sort_idx = np.argsort(divs, kind='mergesort')
        sorted_divs = divs[sort_idx]
        sorted_masks = masks[sort_idx]
        
        # Check adjacent coprime: (m_i & m_{i+1}) == 0
        # Vectorized bitwise and
        adj_masks = sorted_masks[:-1]
        next_masks = sorted_masks[1:]
        
        # Count zeros
        score = np.sum((adj_masks & next_masks) == 0)
        
        return int(score)

    # 3. Initial Population
    # Start with the first k primes.
    # Find max k such that product of first k primes <= 10^18
    max_k = 0
    prod = 1
    for p in primes_pool:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Initial candidate: first max_k primes
    init_primes = list(primes_pool[:max_k])
    
    # Generate a diverse initial population
    pop_size = 20
    population = []
    
    # Add the greedy solution
    population.append(init_primes[:])
    
    # Add random variations
    for _ in range(pop_size - 1):
        cand = init_primes[:]
        # Randomly swap some primes with other primes from the pool
        # Ensure product constraint
        for _ in range(random.randint(1, 3)):
            i = random.randint(0, len(cand) - 1)
            # Pick a random prime from pool, maybe biased towards small ones
            p_new = random.choice(primes_pool[:100]) # Bias to small primes for feasibility
            # Check feasibility
            old_prod = 1
            for p in cand:
                old_prod *= p
            new_prod = old_prod * p_new // cand[i]
            if new_prod <= 10**18:
                cand[i] = p_new
        # Sort to maintain canonical form? 
        # The set of primes defines the number. Order doesn't matter for the number, 
        # but for our mask construction, the order of primes in the list determines the bit positions.
        # However, the *values* of the divisors depend only on the set.
        # So we can sort the primes to have a canonical representation.
        cand.sort()
        population.append(cand)
        
    # Evaluate initial population
    scores = [calculate_score(p) for p in population]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_primes = population[best_idx][:]
    
    # 4. Evolutionary Loop
    # Use Nelder-Mead on the *indices* of the primes? 
    # No, discrete GA is better here.
    
    # To speed up, we will pre-compute the mask array structure once for max_k.
    # The mask values (bitmasks) depend on k.
    # We will cache the sorted order of masks? 
    # No, the sorted order of *divisors* depends on the prime values.
    
    # Optimization: 
    # If we change one prime, can we update the score faster?
    # Probably not easily with the full sort.
    # But calculate_score is vectorized and fast. 
    # Let's estimate: k=15, 2^15=32768. 
    # Sorting 32k floats/ints is ~1ms. 
    # Bitwise ops are ~0.1ms. 
    # Total ~2ms per eval. 
    # In 145s, we can do ~70,000 evaluations.
    
    generation = 0
    while time.time() < deadline - 5.0:
        generation += 1
        new_population = []
        
        # Elitism: keep the best
        new_population.append(best_primes[:])
        
        # Crossover and Mutation
        for _ in range(pop_size - 1):
            # Select parent
            # Roulette wheel or Tournament
            parent = random.choice(population)
            
            # Mutation
            child = parent[:]
            
            # 1. Swap a prime with another
            if random.random() < 0.7:
                i = random.randint(0, len(child) - 1)
                # Sample from pool
                # Bias towards primes that keep the product low
                # Calculate current product
                curr_prod = 1
                for p in child:
                    curr_prod *= p
                
                # Max allowed for new prime
                max_p = int(10**18 * child[i] / curr_prod)
                
                # Find valid candidates in pool
                valid_cands = [p for p in primes_pool if p <= max_p and p not in child]
                if valid_cands:
                    # Bias towards small primes? Or random?
                    # Small primes tend to create more divisors (wait, count is fixed).
                    # Small primes create "dense" low end. Large primes create "sparse" high end.
                    # Coprime pairs are more likely when divisors are far apart?
                    # If d_i and d_{i+1} are far apart, they are likely coprime?
                    # Not necessarily.
                    # Let's try random selection from valid.
                    new_p = random.choice(valid_cands)
                    child[i] = new_p
            
            # 2. Add/Remove prime (change k)
            elif random.random() < 0.2:
                if len(child) < max_k and random.random() < 0.5:
                    # Add a prime
                    curr_prod = 1
                    for p in child:
                        curr_prod *= p
                    max_p = int(10**18 / curr_prod)
                    valid_cands = [p for p in primes_pool if p <= max_p and p not in child]
                    if valid_cands:
                        new_p = random.choice(valid_cands)
                        child.append(new_p)
                else:
                    # Remove a prime
                    if len(child) > 2:
                        i = random.randint(0, len(child) - 1)
                        child.pop(i)
            
            child.sort()
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population
        for i, p in enumerate(population):
            s = calculate_score(p)
            if s > best_score:
                best_score = s
                best_primes = p[:]
                
        # Optional: Local search on the best candidate found in this generation
        if time.time() < deadline - 10.0:
            # Try swapping each prime in best_primes with a random valid prime
            for i in range(len(best_primes)):
                if time.time() > deadline - 10.0: break
                curr_prod = 1
                for p in best_primes:
                    curr_prod *= p
                max_p = int(10**18 * best_primes[i] / curr_prod)
                valid_cands = [p for p in primes_pool if p <= max_p and p not in best_primes]
                if not valid_cands: continue
                
                # Try a few random candidates
                for _ in range(5):
                    if time.time() > deadline - 10.0: break
                    p_new = random.choice(valid_cands)
                    trial = best_primes[:]
                    trial[i] = p_new
                    trial.sort()
                    s = calculate_score(trial)
                    if s > best_score:
                        best_score = s
                        best_primes = trial[:]
        
    # 5. Final Construction
    result = 1
    for p in best_primes:
        result *= p
        
    return result

# Run construct to test if needed, but evaluator calls it.
if __name__ == "__main__":
    n = construct()
    print(n)
