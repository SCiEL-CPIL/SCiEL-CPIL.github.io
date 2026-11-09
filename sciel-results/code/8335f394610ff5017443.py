"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a).
    
    Strategy:
    - Use bitmask-based coprimality checks for speed
    - Multiple greedy initializations with different orderings
    - Local search with 1-swap and optimized 2-swap moves
    - Simulated annealing to escape local optima
    - Multiple restarts with different random seeds
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
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
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            mask |= (1 << prime_to_idx[n])
        return mask
    
    factor_masks = [0] * (N)  # index by a (1..511)
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Precompute coprime-compatible candidates for each mask
    # For each possible mask, precompute which candidates are compatible
    # This is expensive, so we'll compute on-the-fly but cache results
    
    # Compute score of a set
    def compute_score(chosen):
        return sum(scores[a] for a in chosen)
    
    # Greedy construction with given ordering
    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask
    
    # Local search: 1-swap moves (optimized)
    def local_search_1swap(chosen, chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in list(chosen):
                # Find best b to replace a
                best_b = None
                best_gain = 0
                temp_mask = chosen_mask & ~factor_masks[a]
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_b = b
                if best_b is not None:
                    # Perform swap
                    chosen.remove(a)
                    chosen.append(best_b)
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[best_b]
                    current_score += best_gain
                    improved = True
        return chosen, chosen_set, chosen_mask, current_score
    
    # Local search: 2-swap moves (optimized - precompute valid pairs)
    def local_search_2swap(chosen, chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            if n < 2:
                break
            
            # Precompute all valid (b1, b2) pairs for each possible temp_mask
            # We'll do this lazily per temp_mask encountered
            
            for i in range(n):
                if time.time() > deadline:
                    return chosen, chosen_set, chosen_mask, current_score
                for j in range(i+1, n):
                    if time.time() > deadline:
                        return chosen, chosen_set, chosen_mask, current_score
                    a1 = chosen[i]
                    a2 = chosen[j]
                    temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                    
                    # Find best pair (b1, b2) to replace (a1, a2)
                    best_b1 = None
                    best_b2 = None
                    best_gain = 0
                    
                    # Get all candidates compatible with temp_mask
                    compat_candidates = []
                    for b in range(511, 0, -1):
                        if b in chosen_set:
                            continue
                        if factor_masks[b] & temp_mask:
                            continue
                        compat_candidates.append(b)
                    
                    # Find best pair from compat_candidates
                    for idx1 in range(len(compat_candidates)):
                        b1 = compat_candidates[idx1]
                        for idx2 in range(idx1 + 1, len(compat_candidates)):
                            b2 = compat_candidates[idx2]
                            if factor_masks[b1] & factor_masks[b2]:
                                continue
                            gain = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                            if gain > best_gain + 1e-12:
                                best_gain = gain
                                best_b1 = b1
                                best_b2 = b2
                    
                    if best_b1 is not None:
                        # Perform swap
                        chosen.remove(a1)
                        chosen.remove(a2)
                        chosen.append(best_b1)
                        chosen.append(best_b2)
                        chosen_set.remove(a1)
                        chosen_set.remove(a2)
                        chosen_set.add(best_b1)
                        chosen_set.add(best_b2)
                        chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[best_b1] | factor_masks[best_b2]
                        current_score += best_gain
                        improved = True
                        break
                if improved:
                    break
        return chosen, chosen_set, chosen_mask, current_score
    
    # Simulated annealing
    def simulated_annealing(chosen, chosen_set, chosen_mask, current_score):
        temperature = 0.005
        cooling_rate = 0.998
        min_temperature = 0.00005
        max_iters = 5000
        
        iters = 0
        while temperature > min_temperature and time.time() < deadline and iters < max_iters:
            iters += 1
            # Generate a random move
            move_type = random.random()
            if move_type < 0.7 and len(chosen) > 0:
                # 1-swap: remove one, add one
                a = random.choice(chosen)
                temp_mask = chosen_mask & ~factor_masks[a]
                candidates_for_add = [b for b in range(1, 512) if b not in chosen_set and 
                                       (factor_masks[b] & temp_mask) == 0]
                if not candidates_for_add:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates_for_add)
                delta = scores[b] - scores[a]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    # Accept
                    chosen.remove(a)
                    chosen.append(b)
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                    current_score += delta
            elif move_type < 0.9 and len(chosen) >= 2:
                # 2-swap: remove two, add two
                a1, a2 = random.sample(chosen, 2)
                temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                candidates_for_add = [b for b in range(1, 512) if b not in chosen_set and 
                                       (factor_masks[b] & temp_mask) == 0]
                if len(candidates_for_add) < 2:
                    temperature *= cooling_rate
                    continue
                # Try multiple random pairs
                found = False
                for _ in range(5):
                    b1, b2 = random.sample(candidates_for_add, 2)
                    if (factor_masks[b1] & factor_masks[b2]) == 0:
                        delta = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                        if delta > 0 or random.random() < math.exp(delta / temperature):
                            # Accept
                            chosen.remove(a1)
                            chosen.remove(a2)
                            chosen.append(b1)
                            chosen.append(b2)
                            chosen_set.remove(a1)
                            chosen_set.remove(a2)
                            chosen_set.add(b1)
                            chosen_set.add(b2)
                            chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                            current_score += delta
                            found = True
                        break
            else:
                # Just cool down
                pass
            
            temperature *= cooling_rate
        
        return chosen, chosen_set, chosen_mask, current_score
    
    # Main search with multiple restarts
    best_chosen = None
    best_score = -1
    best_chosen_set = None
    best_chosen_mask = 0
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # First restart: deterministic greedy (highest score first)
            ordering = list(range(511, 0, -1))
        elif restart == 1:
            # Second restart: score / num_prime_factors
            def num_prime_factors(a):
                return bin(factor_masks[a]).count('1')
            ordering = sorted(range(1, 512), key=lambda x: scores[x] / max(1, num_prime_factors(x)), reverse=True)
        else:
            # Random restarts with bias towards larger values
            ordering = list(range(1, 512))
            random.shuffle(ordering)
            bias = 0.2 + 0.3 * (restart / num_restarts)
            ordering.sort(key=lambda x: random.random() * bias + scores[x], reverse=True)
        
        chosen, cset, cmask = greedy_construct(ordering)
        s = compute_score(chosen)
        
        # Local search: 1-swap
        chosen, cset, cmask, s = local_search_1swap(chosen, cset, cmask, s)
        
        # Local search: 2-swap
        chosen, cset, cmask, s = local_search_2swap(chosen, cset, cmask, s)
        
        # Simulated annealing
        chosen, cset, cmask, s = simulated_annealing(chosen, cset, cmask, s)
        
        # Final local search passes
        chosen, cset, cmask, s = local_search_1swap(chosen, cset, cmask, s)
        chosen, cset, cmask, s = local_search_2swap(chosen, cset, cmask, s)
        
        if s > best_score:
            best_score = s
            best_chosen = list(chosen)
            best_chosen_set = set(chosen)
            best_chosen_mask = cmask
    
    return best_chosen
# EVOLVE-BLOCK-END
