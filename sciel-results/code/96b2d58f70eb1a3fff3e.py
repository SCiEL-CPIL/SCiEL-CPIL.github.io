import time
import random
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    # Greedy construction: Pick highest a first if coprime to all picked
    def greedy_construct():
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        # Iterate from largest to smallest
        for a in range(N-1, 0, -1):
            if (factor_masks[a] & chosen_mask) == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask, sum(scores[a] for a in chosen)
    
    best_chosen, best_chosen_set, best_chosen_mask, best_score = greedy_construct()
    
    # Local Search: Best-Improvement 1-swap
    def try_1_swap_best(chosen, chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            best_gain = 0
            best_a = None
            best_b = None
            
            for idx in range(n):
                a = chosen[idx]
                mask_excl_a = chosen_mask & ~factor_masks[a]
                
                # Find best b for this a
                for b in range(N-1, 0, -1):
                    if b in chosen_set:
                        continue
                    if (factor_masks[b] & mask_excl_a) != 0:
                        continue
                    
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_a = a
                        best_b = b
                        # Since scores[b] decreases as b decreases, 
                        # for a fixed a, the first valid b is the best b.
                        # So we can break the inner loop for this a.
                        break
            
            if best_a is not None:
                # Perform swap
                chosen.remove(best_a)
                chosen.append(best_b)
                chosen_set.remove(best_a)
                chosen_set.add(best_b)
                chosen_mask = (chosen_mask & ~factor_masks[best_a]) | factor_masks[best_b]
                current_score += best_gain
                improved = True
            else:
                break
                
        return chosen, chosen_set, chosen_mask, current_score

    # Local Search: Best-Improvement 2-swap (Optimized)
    def try_2_swap_best(chosen, chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            if n < 2:
                break
            
            best_gain = 0
            best_a1 = None
            best_a2 = None
            best_b1 = None
            best_b2 = None
            
            for i in range(n):
                a1 = chosen[i]
                for j in range(i+1, n):
                    a2 = chosen[j]
                    
                    cost = scores[a1] + scores[a2]
                    # Pruning: if max possible gain from any b1, b2 is <= best_gain, skip
                    # Max possible gain is scores[N-1] + scores[N-2] - cost
                    if scores[N-1] + scores[N-2] - cost <= best_gain:
                        continue
                        
                    mask_excl = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                    
                    # Find best pair b1, b2
                    # Iterate b1 from high to low
                    for b1 in range(N-1, 0, -1):
                        if b1 in chosen_set:
                            continue
                        if (factor_masks[b1] & mask_excl) != 0:
                            continue
                        
                        # Pruning for b1: if scores[b1] + max_possible_b2 <= best_gain + cost, skip?
                        # max_possible_b2 is scores[N-1] (if not b1)
                        if scores[b1] + scores[N-1] <= best_gain + cost:
                            break # No further b1 can improve
                    
                        # Find best b2 for this b1
                        for b2 in range(N-1, 0, -1):
                            if b2 == b1 or b2 in chosen_set:
                                continue
                            if (factor_masks[b2] & mask_excl) != 0:
                                continue
                            if (factor_masks[b1] & factor_masks[b2]) != 0:
                                continue
                            
                            gain = scores[b1] + scores[b2] - cost
                            if gain > best_gain + 1e-12:
                                best_gain = gain
                                best_a1 = a1
                                best_a2 = a2
                                best_b1 = b1
                                best_b2 = b2
                                break # Best b2 for this b1 found
                    
            if best_a1 is not None:
                # Perform swap
                chosen.remove(best_a1)
                chosen.remove(best_a2)
                chosen.append(best_b1)
                chosen.append(best_b2)
                chosen_set.remove(best_a1)
                chosen_set.remove(best_a2)
                chosen_set.add(best_b1)
                chosen_set.add(best_b2)
                chosen_mask = (chosen_mask & ~factor_masks[best_a1] & ~factor_masks[best_a2]) | factor_masks[best_b1] | factor_masks[best_b2]
                current_score += best_gain
                improved = True
            else:
                break
                
        return chosen, chosen_set, chosen_mask, current_score

    # Run local search
    best_chosen, best_chosen_set, best_chosen_mask, best_score = try_1_swap_best(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    best_chosen, best_chosen_set, best_chosen_mask, best_score = try_2_swap_best(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    best_chosen, best_chosen_set, best_chosen_mask, best_score = try_1_swap_best(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    
    # Iterated Local Search / Random Restarts
    while time.time() < deadline - 5:
        # Create a copy
        curr_chosen = list(best_chosen)
        curr_chosen_set = set(best_chosen)
        curr_chosen_mask = best_chosen_mask
        curr_score = best_score
        
        # Perturb: pick 1 random element, remove it, add 1 random valid element
        if len(curr_chosen) > 1:
            idx1 = random.randint(0, len(curr_chosen)-1)
            a1 = curr_chosen[idx1]
            
            # Try to find a valid replacement for a1
            mask_excl = curr_chosen_mask & ~factor_masks[a1]
            valid_candidates = [b for b in range(N-1, 0, -1) if b not in curr_chosen_set and (factor_masks[b] & mask_excl) == 0]
            
            if valid_candidates:
                # Pick a random candidate
                b1 = random.choice(valid_candidates)
                
                # Apply perturbation
                curr_chosen.remove(a1)
                curr_chosen.append(b1)
                curr_chosen_set.remove(a1)
                curr_chosen_set.add(b1)
                curr_chosen_mask = (curr_chosen_mask & ~factor_masks[a1]) | factor_masks[b1]
                curr_score = sum(scores[x] for x in curr_chosen)
                
                # Run local search on perturbed solution
                curr_chosen, curr_chosen_set, curr_chosen_mask, curr_score = try_1_swap_best(
                    curr_chosen, curr_chosen_set, curr_chosen_mask, curr_score
                )
                curr_chosen, curr_chosen_set, curr_chosen_mask, curr_score = try_2_swap_best(
                    curr_chosen, curr_chosen_set, curr_chosen_mask, curr_score
                )
                curr_chosen, curr_chosen_set, curr_chosen_mask, curr_score = try_1_swap_best(
                    curr_chosen, curr_chosen_set, curr_chosen_mask, curr_score
                )
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_chosen = curr_chosen
                    best_chosen_set = curr_chosen_set
                    best_chosen_mask = curr_chosen_mask
        
    return best_chosen
