import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    # Primes below 512
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(512)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * 512
    weights = [0.0] * 512
    
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        n = a
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                masks[a] |= (1 << prime_to_idx[p])
                while n % p == 0:
                    n //= p
        if n > 1:
            masks[a] |= (1 << prime_to_idx[n])
            
    # 2. Greedy Initialization
    def greedy_init(skip_first_k=0):
        chosen = []
        used_mask = 0
        # Iterate from largest weight (smallest denom, largest a) to smallest
        for a in range(511, 0, -1):
            if skip_first_k > 0:
                skip_first_k -= 1
                continue
            if masks[a] & used_mask == 0:
                chosen.append(a)
                used_mask |= masks[a]
        return chosen

    def calculate_score(chosen):
        return sum(weights[a] for a in chosen)

    def get_mask_of_set(chosen):
        m = 0
        for a in chosen:
            m |= masks[a]
        return m

    # 3. Local Search
    def local_search(chosen):
        current_score = calculate_score(chosen)
        chosen_set = set(chosen)
        
        while time.time() < deadline:
            improved = False
            # Try removing each element and refilling greedily
            for a in chosen:
                if time.time() >= deadline:
                    break
                
                # Remove a
                temp_chosen = [x for x in chosen if x != a]
                temp_set = chosen_set - {a}
                temp_mask = get_mask_of_set(temp_chosen)
                
                # Refill greedily
                # Candidates are all numbers not in temp_set
                # Sorted by weight descending (which is a ascending? No, weight is 1/(512-a), so larger a is higher weight)
                # So we iterate a from 511 down to 1
                
                new_chosen = list(temp_chosen)
                new_mask = temp_mask
                new_score = current_score - weights[a] # Start with score without a
                
                # We need to check if we can add anything
                # To be efficient, we only need to check if the sum of added weights > weights[a]
                # Greedy fill:
                for b in range(511, 0, -1):
                    if b in temp_set:
                        continue
                    if masks[b] & new_mask == 0:
                        new_chosen.append(b)
                        new_mask |= masks[b]
                        new_score += weights[b]
                
                if new_score > current_score + 1e-9:
                    chosen = new_chosen
                    chosen_set = set(chosen)
                    current_score = new_score
                    improved = True
                    break # Restart loop with new state
            
            if not improved:
                break
                
        return chosen, current_score

    # 4. Main Loop with Restarts
    best_chosen = greedy_init()
    best_score = calculate_score(best_chosen)
    
    # Run local search on the initial greedy result
    best_chosen, best_score = local_search(best_chosen)
    
    # Try some random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        # Perturb the greedy: skip a few random large numbers
        # Or just skip the first K numbers in the greedy order
        k = restart_count % 10
        if k == 0:
            continue # Skip this restart type, just do another local search pass on best?
            
        init_chosen = greedy_init(skip_first_k=k)
        # It might be better to start from best_chosen and do a random swap?
        # But greedy init with different skips explores different basins.
        
        refined_chosen, refined_score = local_search(init_chosen)
        
        if refined_score > best_score:
            best_score = refined_score
            best_chosen = refined_chosen
            
        # Also try local search on best_chosen again to ensure convergence
        if time.time() < deadline - 2.0:
             refined_best, score_best = local_search(best_chosen)
             if score_best > best_score:
                 best_score = score_best
                 best_chosen = refined_best
                 
    return best_chosen
