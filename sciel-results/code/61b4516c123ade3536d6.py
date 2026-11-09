"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

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
    
    # Precompute prime factor masks for each candidate
    masks = [0] * N
    scores = [0.0] * N
    for a in range(1, N):
        n = a
        mask = 0
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
        masks[a] = mask
        scores[a] = 1.0 / (N - a)
        
    # Greedy construction helper
    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        used_mask = 0
        for a in ordering:
            if masks[a] & used_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                used_mask |= masks[a]
        return chosen, chosen_set, used_mask

    # Multiple initializations
    best_chosen = []
    best_chosen_set = set()
    best_used_mask = 0
    best_score = -1.0

    # 1. Descending score (a descending)
    ordering1 = list(range(N-1, 0, -1))
    ch, cs, cm = greedy_construct(ordering1)
    s = sum(scores[a] for a in ch)
    if s > best_score:
        best_score = s
        best_chosen = ch
        best_chosen_set = cs
        best_used_mask = cm

    # 2. Random with bias towards high scores
    for _ in range(10):
        if time.time() > deadline:
            break
        ordering = list(range(1, N))
        random.shuffle(ordering)
        # Bias: sort by random + score
        ordering.sort(key=lambda x: random.random() * 0.5 + scores[x], reverse=True)
        ch, cs, cm = greedy_construct(ordering)
        s = sum(scores[a] for a in ch)
        if s > best_score:
            best_score = s
            best_chosen = ch
            best_chosen_set = cs
            best_used_mask = cm

    # 3. Score / number of prime factors
    def num_pf(a):
        return bin(masks[a]).count('1')
    ordering3 = sorted(range(1, N), key=lambda x: scores[x] / max(1, num_pf(x)), reverse=True)
    ch, cs, cm = greedy_construct(ordering3)
    s = sum(scores[a] for a in ch)
    if s > best_score:
        best_score = s
        best_chosen = ch
        best_chosen_set = cs
        best_used_mask = cm

    chosen = best_chosen
    chosen_set = best_chosen_set
    used_mask = best_used_mask
    current_score = best_score

    # Local Search
    def local_search():
        nonlocal chosen, chosen_set, used_mask, current_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1-Swap
            non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
            # Sort non_chosen by score descending for faster pruning in 2-swap later
            non_chosen.sort(key=lambda x: scores[x], reverse=True)
            
            for i, a in enumerate(chosen):
                if time.time() > deadline:
                    break
                mask_without_a = used_mask ^ masks[a]
                best_b = -1
                best_gain = 0.0
                
                # Iterate non_chosen (already sorted by score desc)
                for b in non_chosen:
                    if masks[b] & mask_without_a == 0:
                        gain = scores[b] - scores[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    # Pruning: if scores[b] <= scores[a], and sorted, we can break?
                    # No, because we need gain > 0, so scores[b] > scores[a].
                    # Since sorted desc, once scores[b] <= scores[a], all subsequent are <=.
                    if scores[b] <= scores[a]:
                        break
                
                if best_b != -1 and best_gain > 1e-12:
                    chosen[i] = best_b
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    used_mask = mask_without_a | masks[best_b]
                    current_score += best_gain
                    improved = True
                    # Update non_chosen list
                    # Remove best_b, add a
                    # Since we restart the while loop, we will rebuild non_chosen
                    break # Restart 1-swap loop
            
            if improved:
                continue # Restart while loop
            
            # 2-Swap
            # Rebuild non_chosen if needed, but we just built it above if 1-swap didn't trigger restart
            # However, if 1-swap didn't improve, non_chosen is still valid from the top of the loop
            # But we should ensure it's sorted and up to date.
            non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
            non_chosen.sort(key=lambda x: scores[x], reverse=True)
            
            n_chosen = len(chosen)
            found_2swap = False
            
            # Limit search to top K candidates to balance speed/quality
            K = 200
            top_non_chosen = non_chosen[:K]
            
            for i in range(n_chosen):
                if time.time() > deadline:
                    break
                a = chosen[i]
                for j in range(i+1, n_chosen):
                    c = chosen[j]
                    mask_without_ac = used_mask ^ masks[a] ^ masks[c]
                    target_score = scores[a] + scores[c]
                    
                    best_pair = None
                    best_gain = 0.0
                    
                    # Search in top_non_chosen
                    for b_idx, b in enumerate(top_non_chosen):
                        if masks[b] & mask_without_ac != 0:
                            continue
                        # Pruning: if scores[b] + max_possible_d <= target_score, break?
                        # max_possible_d is scores[top_non_chosen[0]] if b is not 0, or next.
                        # Simpler: just iterate.
                        for d_idx in range(b_idx + 1, len(top_non_chosen)):
                            d = top_non_chosen[d_idx]
                            if masks[d] & mask_without_ac != 0:
                                continue
                            if masks[b] & masks[d] != 0:
                                continue
                            gain = scores[b] + scores[d] - target_score
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (b, d)
                            # Pruning: if scores[b] + scores[d] <= target_score and sorted, 
                            # we can break inner loop? 
                            # Since d_idx increases, scores[d] decreases.
                            # If scores[b] + scores[d] <= target_score, then for larger d_idx, it will be even smaller.
                            if scores[b] + scores[d] <= target_score:
                                break
                    
                    if best_pair is not None and best_gain > 1e-12:
                        b, d = best_pair
                        chosen[i] = b
                        chosen[j] = d
                        chosen_set.remove(a)
                        chosen_set.remove(c)
                        chosen_set.add(b)
                        chosen_set.add(d)
                        used_mask = mask_without_ac | masks[b] | masks[d]
                        current_score += best_gain
                        improved = True
                        found_2swap = True
                        break # Restart while loop
                if found_2swap:
                    break

    local_search()

    return chosen
# EVOLVE-BLOCK-END
