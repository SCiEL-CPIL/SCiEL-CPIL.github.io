import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 110  # Safety margin before 120s limit
    
    N = 511
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Precompute factor bitmask for each number 1..N
    # 1 has no prime factors (empty set)
    factor_mask = [0] * (N + 1)
    factor_mask[1] = 0
    for a in range(2, N + 1):
        mask = 0
        for p in primes:
            if p * p > a:
                break
            if a % p == 0:
                mask |= (1 << prime_index[p])
                while a % p == 0:
                    a //= p
        if a > 1:  # remaining prime factor
            mask |= (1 << prime_index[a])
        factor_mask[a] = mask
    
    # Candidates sorted by decreasing weight (i.e., decreasing a)
    candidates = list(range(N, 0, -1))
    weights = {a: 1.0 / (512 - a) for a in candidates}
    
    # Phase 1: Greedy initial solution
    greedy_chosen = []
    greedy_mask = 0
    greedy_score = 0.0
    for a in candidates:
        if factor_mask[a] & greedy_mask == 0:
            greedy_chosen.append(a)
            greedy_mask |= factor_mask[a]
            greedy_score += weights[a]
    
    # Phase 2: Beam search
    beam_width = 3000
    # State: (score, mask, chosen_tuple)
    # Start with greedy solution as one of the initial states? 
    # Actually, let's just do standard beam search from empty.
    # But we can seed with the greedy solution to help.
    
    states = [(greedy_score, greedy_mask, tuple(greedy_chosen))]
    
    # Also add empty state to allow beam search to find different solutions
    states.append((0.0, 0, ()))
    
    best_score = greedy_score
    best_chosen = tuple(greedy_chosen)
    
    for a in candidates:
        if time.time() - start_time > DEADLINE:
            break
        
        w_a = weights[a]
        mask_a = factor_mask[a]
        
        new_states = []
        
        for score, mask, chosen in states:
            # Option 1: Exclude a
            new_states.append((score, mask, chosen))
            
            # Option 2: Include a if coprime
            if mask_a & mask == 0:
                new_score = score + w_a
                new_mask = mask | mask_a
                new_chosen = chosen + (a,)
                new_states.append((new_score, new_mask, new_chosen))
                
                if new_score > best_score:
                    best_score = new_score
                    best_chosen = new_chosen
        
        # Prune: keep top beam_width states by score
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]
    
    # Phase 3: Local search polish
    # Try to improve the best solution found
    chosen_set = set(best_chosen)
    chosen_list = list(best_chosen)
    
    # Build a list of all candidates not in chosen
    not_chosen = [a for a in range(1, N + 1) if a not in chosen_set]
    
    # Try adding any candidate that is coprime to all chosen
    current_mask = 0
    for a in best_chosen:
        current_mask |= factor_mask[a]
    
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        
        # Try to add a new candidate
        for a in not_chosen:
            if factor_mask[a] & current_mask == 0:
                # Can add a
                chosen_list.append(a)
                chosen_set.add(a)
                current_mask |= factor_mask[a]
                not_chosen.remove(a)
                improved = True
                break
        
        # Try swapping: remove one chosen, add one or more not_chosen
        if not improved:
            for i, rem in enumerate(chosen_list):
                new_mask = 0
                for j, a in enumerate(chosen_list):
                    if j != i:
                        new_mask |= factor_mask[a]
                
                # Try adding any not_chosen that is coprime to new_mask
                added_any = False
                for a in not_chosen:
                    if factor_mask[a] & new_mask == 0:
                        # Check if adding a gives better score
                        old_score = sum(weights[x] for x in chosen_list)
                        new_score = old_score - weights[rem] + weights[a]
                        if new_score > old_score + 1e-12:
                            # Do the swap
                            chosen_list.pop(i)
                            chosen_set.discard(rem)
                            chosen_list.append(a)
                            chosen_set.add(a)
                            current_mask = 0
                            for x in chosen_list:
                                current_mask |= factor_mask[x]
                            # Update not_chosen
                            if rem in not_chosen:
                                not_chosen.remove(rem)
                            if a in not_chosen:
                                not_chosen.remove(a)
                            improved = True
                            break
                if improved:
                    break
    
    # Final verification and return
    result = list(chosen_list)
    return result
