"""Rewritten program using Local Search (Hill Climbing) to improve upon the greedy seed."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import random
    
    MAX_VAL = 511
    N = 512

    # Precompute prime factors for all numbers 1..511
    factors = [set() for _ in range(MAX_VAL + 1)]
    for i in range(2, MAX_VAL + 1):
        if not factors[i]: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                factors[j].add(i)
    
    # Precompute values 1/(N-a)
    values = [0.0] * (MAX_VAL + 1)
    for a in range(1, MAX_VAL + 1):
        values[a] = 1.0 / (N - a)

    def get_score(S):
        return sum(values[a] for a in S)

    # 1. Generate initial greedy solution
    S = set()
    used_primes = set()
    
    # Greedy: pick largest a first
    for a in range(MAX_VAL, 0, -1):
        if not (factors[a] & used_primes):
            S.add(a)
            used_primes.update(factors[a])
            
    current_score = get_score(S)
    
    # 2. Local Search
    max_iterations = 5000
    no_improve_count = 0
    
    for _ in range(max_iterations):
        improved = False
        
        # Move Type 1: Add a single number
        # Find the best candidate not in S that is coprime to all in S
        best_add = None
        best_add_val = 0
        for a in range(MAX_VAL, 0, -1):
            if a not in S:
                if not (factors[a] & used_primes):
                    val = values[a]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = a
        
        if best_add is not None:
            S.add(best_add)
            used_primes.update(factors[best_add])
            current_score += best_add_val
            improved = True
            # After adding, we might be able to add more, so continue loop
            continue
            
        # Move Type 2: Swap/Refill
        # Try removing each element in S and refilling greedily
        best_swap_set = None
        best_swap_used = None
        best_swap_score = current_score
        
        # Iterate over a copy of the list of elements in S
        for a in list(S):
            # Remove a
            S_temp = S.copy()
            S_temp.remove(a)
            # used_primes_temp = used_primes - factors[a] is valid because S is pairwise coprime
            # But we don't need to construct the set explicitly if we use the subset check
            
            score_temp = current_score - values[a]
            
            # Greedily fill S_temp with new numbers
            # We iterate candidates from largest to smallest value
            for b in range(MAX_VAL, 0, -1):
                if b in S_temp:
                    continue
                
                # Check if b is coprime to S_temp
                # S_temp corresponds to used_primes - factors[a]
                # So b is valid if (factors[b] & used_primes) is a subset of factors[a]
                inter = factors[b] & used_primes
                if inter <= factors[a]:
                    S_temp.add(b)
                    # We don't update used_primes_temp explicitly, but we add to S_temp
                    # The next check will use the original used_primes and factors[a]
                    # Wait, if we add b, does it affect subsequent checks for other b's?
                    # Yes. If we add b, its factors become "used".
                    # So we MUST maintain a temporary used_primes set for the refill process.
                    
                    # Let's revert to maintaining used_primes_temp for correctness
                    # The optimization `inter <= factors[a]` only works for the FIRST addition
                    # or if we don't track newly added factors.
                    # Since we are filling greedily, adding one b might block another b.
                    # So we need to track `used_primes_temp`.
                    
                    # Let's re-do this part with correct tracking but optimized copy
                    pass
            
            # The previous logic was flawed for multiple additions.
            # Let's use a proper temporary set but optimize the copy.
            # Since N=512, we can just use the set copy.
            pass
            
        # Re-implementing Swap correctly with optimization
        best_swap_set = None
        best_swap_used = None
        best_swap_score = current_score
        
        for a in list(S):
            S_temp = S.copy()
            S_temp.remove(a)
            used_primes_temp = used_primes.copy()
            used_primes_temp.difference_update(factors[a])
            
            score_temp = current_score - values[a]
            
            for b in range(MAX_VAL, 0, -1):
                if b in S_temp:
                    continue
                if not (factors[b] & used_primes_temp):
                    S_temp.add(b)
                    used_primes_temp.update(factors[b])
                    score_temp += values[b]
            
            if score_temp > best_swap_score:
                best_swap_score = score_temp
                best_swap_set = S_temp
                best_swap_used = used_primes_temp

        if best_swap_set is not None:
            S = best_swap_set
            used_primes = best_swap_used
            current_score = best_swap_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                break
                
    return list(S)
# EVOLVE-BLOCK-END
