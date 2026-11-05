import time
import random

def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    L_max = 5000
    
    best_L = 0
    best_res = [0] * n_primes
    
    start_time = time.time()
    deadline = start_time + 140
    
    def compute_counts(res):
        counts = [0] * (L_max + 1)
        for i, p in enumerate(primes):
            r = res[i]
            start_x = r
            if start_x == 0:
                start_x = p
            if start_x > L_max:
                continue
            for x in range(start_x, L_max + 1, p):
                counts[x] += 1
        return counts

    def get_L(counts):
        for i in range(1, L_max + 1):
            if counts[i] < 2:
                return i - 1
        return L_max

    def evaluate(res):
        counts = compute_counts(res)
        return get_L(counts)

    # Seed configuration
    seed_res = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    seed_L = evaluate(seed_res)
    if seed_L > best_L:
        best_L = seed_L
        best_res = seed_res.copy()

    restarts = 0
    # Use a deterministic seed for reproducibility in testing, 
    # but maybe random for better exploration? 
    # Let's use a fixed seed to ensure we at least get the seed score, 
    # and then explore.
    random.seed(42)
    
    while time.time() < deadline:
        restarts += 1
        # Alternate between random init and seed init?
        if restarts % 5 == 0:
            res = seed_res.copy()
        else:
            res = [random.randint(0, p - 1) for p in primes]
            
        current_L = evaluate(res)
        
        if current_L > best_L:
            best_L = current_L
            best_res = res.copy()
        
        # Hill Climbing
        while time.time() < deadline:
            counts = compute_counts(res)
            first_fail = L_max + 1
            for i in range(1, L_max + 1):
                if counts[i] < 2:
                    first_fail = i
                    break
            
            if first_fail > L_max:
                break
                
            if random.random() < 0.5:
                # Directed move
                candidates = []
                for i, p in enumerate(primes):
                    if first_fail % p != res[i]:
                        candidates.append(i)
                
                if candidates:
                    i = random.choice(candidates)
                    new_r = first_fail % primes[i]
                    old_r = res[i]
                    res[i] = new_r
                    new_L = evaluate(res)
                    
                    if new_L > current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_res = res.copy()
                    elif new_L == current_L and random.random() < 0.05:
                        pass
                    else:
                        res[i] = old_r
                else:
                    # Should not happen
                    pass
            else:
                # Random move
                i = random.randint(0, n_primes - 1)
                old_r = res[i]
                new_r = random.randint(0, primes[i] - 1)
                res[i] = new_r
                new_L = evaluate(res)
                
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_res = res.copy()
                elif new_L == current_L and random.random() < 0.05:
                    pass
                else:
                    res[i] = old_r

    return best_res
