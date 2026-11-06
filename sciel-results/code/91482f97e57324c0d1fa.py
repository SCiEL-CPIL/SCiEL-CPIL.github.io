import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 512
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    primes = [i for i in range(N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    for a in range(1, N):
        m = a
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask_val |= (1 << prime_index[m])
        masks[a] = mask_val
        
    vals = [0.0] * N
    for a in range(1, N):
        vals[a] = 1.0 / (N - a)

    # Greedy Init
    def greedy_init():
        chosen = []
        chosen_masks = []
        for a in range(N - 1, 0, -1):
            if all((masks[a] & cm) == 0 for cm in chosen_masks):
                chosen.append(a)
                chosen_masks.append(masks[a])
        return chosen, chosen_masks

    best_chosen, best_masks = greedy_init()
    best_score = sum(vals[a] for a in best_chosen)
    
    # Randomized Hill Climbing
    # State: chosen (list), chosen_masks (list), chosen_set (set)
    
    def try_improve(chosen, chosen_masks, chosen_set, score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a few random swaps
            for _ in range(100): # Fixed number of attempts per "iteration"
                if time.time() > deadline:
                    return chosen, chosen_masks, chosen_set, score
                
                # Pick random index to remove
                idx = random.randint(0, len(chosen)-1)
                a = chosen[idx]
                
                # Pick random u to add
                # We need u not in chosen_set
                # Simple way: pick random u in 1..511, if in set, retry or skip
                # Better: maintain list of unselected? Or just random sampling.
                # With ~50 selected, probability of collision is low (~10%).
                u = random.randint(1, N-1)
                if u in chosen_set:
                    continue
                
                mu = masks[u]
                
                # Check if u conflicts with a
                if mu & chosen_masks[idx]:
                    continue
                    
                # Check if u conflicts with others
                ok = True
                for j in range(len(chosen)):
                    if j == idx: continue
                    if mu & chosen_masks[j]:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Valid swap
                delta = vals[u] - vals[a]
                if delta > 1e-9:
                    # Perform swap
                    chosen[idx] = u
                    chosen_masks[idx] = mu
                    chosen_set.discard(a)
                    chosen_set.add(u)
                    score += delta
                    improved = True
                    break # Restart loop to find more improvements
                    
        return chosen, chosen_masks, chosen_set, score

    # Run initial optimization
    best_set = set(best_chosen)
    best_chosen, best_masks, best_set, best_score = try_improve(best_chosen, best_masks, best_set, best_score)
    
    # ILS Loop
    while time.time() < deadline:
        # Perturb
        c_chosen = list(best_chosen)
        c_masks = list(best_masks)
        c_set = set(c_chosen)
        c_score = best_score
        
        # Random perturbation: remove 2-5, add 2-5 valid
        k = random.randint(2, 5)
        for _ in range(k):
            if c_chosen:
                idx = random.randint(0, len(c_chosen)-1)
                rem = c_chosen.pop(idx)
                c_masks.pop(idx)
                c_set.discard(rem)
                c_score -= vals[rem]
                
        for _ in range(k):
            # Try to add random valid u
            for _ in range(10):
                u = random.randint(1, N-1)
                if u in c_set: continue
                if all((masks[u] & m) == 0 for m in c_masks):
                    c_chosen.append(u)
                    c_masks.append(masks[u])
                    c_set.add(u)
                    c_score += vals[u]
                    break
        
        # Optimize perturbed
        c_chosen, c_masks, c_set, c_score = try_improve(c_chosen, c_masks, c_set, c_score)
        
        if c_score > best_score + 1e-9:
            best_chosen = c_chosen
            best_masks = c_masks
            best_set = c_set
            best_score = c_score
            
    return best_chosen
