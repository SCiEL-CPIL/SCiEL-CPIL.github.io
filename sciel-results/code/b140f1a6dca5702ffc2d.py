import math

# EVOLVE-BLOCK-START

def construct():
    N = 511
    candidates = list(range(511, 0, -1))
    weights = [1.0 / (512 - a) for a in candidates]
    
    # Generate primes < 512
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    # Map prime to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Compute bitmask for each number
    # mask[a] has bit i set if prime[i] divides a
    masks = [0] * (N + 1)
    for a in range(1, N + 1):
        x = a
        m = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                m |= (1 << i)
                while x % p == 0:
                    x //= p
        if x > 1:
            # x is a prime > sqrt(a), must be in primes list
            if x in prime_to_idx:
                m |= (1 << prime_to_idx[x])
            else:
                # Should not happen for a < 512
                pass
        masks[a] = m

    # Suffix sums for pruning
    suffix_sum = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]

    # Greedy initialization
    best_score = 0.0
    best_solution = []
    used_mask_g = 0
    greedy_sol = []
    for k in range(len(candidates)):
        a = candidates[k]
        if (masks[a] & used_mask_g) == 0:
            greedy_sol.append(a)
            used_mask_g |= masks[a]
            best_score += weights[k]
    best_solution = greedy_sol

    # Branch and Bound
    def search(idx, current_sum, used_mask, current_sol):
        nonlocal best_score, best_solution
        
        # Pruning
        if current_sum + suffix_sum[idx] <= best_score:
            return
            
        if idx == len(candidates):
            if current_sum > best_score:
                best_score = current_sum
                best_solution = list(current_sol)
            return
            
        a = candidates[idx]
        ma = masks[a]
        
        # Branch 1: Take a
        if (ma & used_mask) == 0:
            current_sol.append(a)
            search(idx + 1, current_sum + weights[idx], used_mask | ma, current_sol)
            current_sol.pop()
            
        # Branch 2: Skip a
        search(idx + 1, current_sum, used_mask, current_sol)

    search(0, 0.0, 0, [])
    
    return best_solution

# EVOLVE-BLOCK-END
