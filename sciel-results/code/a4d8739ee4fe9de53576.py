import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    values = [0.0] * N
    for a in range(1, N):
        values[a] = 1.0 / (N - a)
        
    # Generate primes < N
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute bitmasks for prime factors
    # Python integers are arbitrary precision, so this is safe for ~100 bits
    masks = [0] * N
    for a in range(1, N):
        n = a
        m = 0
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                m |= (1 << prime_index[p])
                while n % p == 0:
                    n //= p
        if n > 1:
            m |= (1 << prime_index[n])
        masks[a] = m
        
    # Sort indices by value descending
    sorted_indices = sorted(range(1, N), key=lambda x: values[x], reverse=True)
    
    def greedy_fill(used_mask, chosen_set, score):
        """
        Greedily add numbers to chosen_set that are coprime to used_mask.
        Returns (new_score, new_chosen_set, new_used_mask).
        """
        for a in sorted_indices:
            if a in chosen_set:
                continue
            if masks[a] & used_mask:
                continue
            chosen_set.add(a)
            used_mask |= masks[a]
            score += values[a]
        return score, chosen_set, used_mask

    def local_search(chosen_set, used_mask, score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Ensure maximal (add any possible)
            new_score, new_set, new_mask = greedy_fill(used_mask, chosen_set, score)
            if new_score > score + 1e-12:
                score = new_score
                chosen_set = new_set
                used_mask = new_mask
                improved = True
                continue # Restart loop as set changed significantly

            # 2. Remove 1 and Refill
            for a in list(chosen_set):
                if time.time() > deadline:
                    return score, chosen_set, used_mask
                
                old_score = score
                old_set = chosen_set.copy()
                old_mask = used_mask
                
                # Remove a
                chosen_set.remove(a)
                used_mask = old_mask & ~masks[a]
                score = old_score - values[a]
                
                # Refill
                new_score, new_set, new_mask = greedy_fill(used_mask, chosen_set, score)
                
                if new_score > old_score + 1e-12:
                    score = new_score
                    chosen_set = new_set
                    used_mask = new_mask
                    improved = True
                else:
                    # Revert
                    chosen_set = old_set
                    used_mask = old_mask
                    score = old_score
            
            # 3. Remove 2 and Refill
            # This is O(N^2 * Greedy). Greedy is O(N). Total O(N^3).
            # N=511. 511^3 ~ 134M. Too slow if done naively in Python for many iterations.
            # We can limit this or optimize.
            # Since N is small, let's try a limited version or just rely on Remove 1.
            # However, Remove 1 might get stuck in local optima where 2 swaps are needed.
            # Let's try Remove 2 only if time permits or if stuck.
            
            if not improved and time.time() < deadline - 5:
                chosen_list = list(chosen_set)
                # Randomly sample pairs to save time? Or just iterate all if fast enough.
                # Let's iterate all pairs but break early if time is tight.
                for i in range(len(chosen_list)):
                    for j in range(i + 1, len(chosen_list)):
                        if time.time() > deadline - 2:
                            return score, chosen_set, used_mask
                        
                        a = chosen_list[i]
                        b = chosen_list[j]
                        
                        old_score = score
                        old_set = chosen_set.copy()
                        old_mask = used_mask
                        
                        # Remove a, b
                        chosen_set.remove(a)
                        chosen_set.remove(b)
                        used_mask = old_mask & ~masks[a] & ~masks[b]
                        score = old_score - values[a] - values[b]
                        
                        # Refill
                        new_score, new_set, new_mask = greedy_fill(used_mask, chosen_set, score)
                        
                        if new_score > old_score + 1e-12:
                            score = new_score
                            chosen_set = new_set
                            used_mask = new_mask
                            improved = True
                            break # Restart loop with new state
                        else:
                            chosen_set = old_set
                            used_mask = old_mask
                            score = old_score
                    if improved:
                        break
                        
        return score, chosen_set, used_mask

    best_score = 0.0
    best_chosen = set()
    
    import random
    random.seed(42)
    
    # Run multiple restarts
    # Each restart: Greedy Init + Local Search
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        chosen_set = set()
        used_mask = 0
        score = 0.0
        
        # Perturbed Greedy Init
        indices = sorted_indices[:]
        if r > 0:
            # Randomly swap some elements to create diversity
            for k in range(len(indices) - 1):
                if random.random() < 0.05:
                    indices[k], indices[k+1] = indices[k+1], indices[k]
                    
        for a in indices:
            if masks[a] & used_mask:
                continue
            chosen_set.add(a)
            used_mask |= masks[a]
            score += values[a]
            
        # Local Search
        score, chosen_set, used_mask = local_search(chosen_set, used_mask, score)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen_set.copy()
            
    return list(best_chosen)
