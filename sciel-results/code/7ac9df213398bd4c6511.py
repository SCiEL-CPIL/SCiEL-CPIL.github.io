"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 512
    # a in 1..511, score = sum 1/(512 - a) = sum 1/k where k = 512 - a, k in 1..511
    # We want to maximize sum 1/k where k = 512 - a, and the a's must be pairwise coprime.
    # Note: a and b are coprime iff gcd(a,b)==1.
    # k = 512 - a, so a = 512 - k.
    # gcd(512-k1, 512-k2) = 1 for all pairs.
    
    # Precompute: for each a in 1..511, its prime factors (for fast coprimality check)
    # a is in 1..511
    # Let's precompute smallest prime factor (SPF) for numbers up to 511
    max_val = 511
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Return set of distinct prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    # Precompute prime factors for all a in 1..511
    pf = [None] * (max_val + 1)
    pf[0] = set()
    pf[1] = set()
    for a in range(2, max_val + 1):
        pf[a] = get_prime_factors(a)
    
    # Score for choosing a is 1/(512 - a)
    # We want to select a subset of {1, ..., 511} that are pairwise coprime, maximizing sum 1/(512-a)
    
    # Greedy initialization: sort by 1/(512-a) descending (i.e., a descending), pick if coprime to all chosen
    def greedy_init(order):
        chosen = []
        chosen_pfs = []
        for a in order:
            if all(not (pf[a] & c) for c in chosen_pfs):
                chosen.append(a)
                chosen_pfs.append(pf[a])
        return chosen
    
    # Multiple greedy initializations with different orderings
    base_order = list(range(511, 0, -1))  # a descending: 511, 510, ..., 1
    
    # Try several different orderings
    initializations = []
    
    # 1. Standard descending
    initializations.append(greedy_init(base_order))
    
    # 2. Randomized orderings (seeded for reproducibility)
    rng = random.Random(42)
    for trial in range(20):
        order = list(range(511, 0, -1))
        rng.shuffle(order)
        # But weight by 1/(512-a) to bias towards high-value items
        # Actually, let's just shuffle and see
        initializations.append(greedy_init(order))
    
    # 3. Weighted random: prioritize high 1/(512-a) but with some randomness
    for trial in range(20):
        order = []
        remaining = list(range(511, 0, -1))
        while remaining:
            # Weight by 1/(512-a)
            weights = [1.0 / (512 - a) for a in remaining]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    idx = i
                    break
            order.append(remaining[idx])
            remaining.pop(idx)
        initializations.append(greedy_init(order))
    
    # 4. Reverse order (a ascending)
    initializations.append(greedy_init(list(range(1, 512))))
    
    # Find best initialization
    def score_fraction(chosen):
        from fractions import Fraction
        total = Fraction(0, 1)
        for a in chosen:
            total += Fraction(1, 512 - a)
        return total
    
    def score_float(chosen):
        return sum(1.0 / (512 - a) for a in chosen)
    
    best_init = max(initializations, key=score_float)
    best_score = score_float(best_init)
    
    # Local search: try to swap one element for another, or add/remove
    # Use bitmask-style coprimality check via prime factor sets
    
    def is_coprime_to_all(a, chosen):
        """Check if a is coprime to all elements in chosen."""
        pa = pf[a]
        if not pa:
            return True
        for b in chosen:
            if pa & pf[b]:
                return False
        return True
    
    def local_search(chosen, time_limit):
        """Local search: try 1-swap moves and 1-add moves."""
        chosen_set = set(chosen)
        chosen_list = list(chosen)
        current_score = score_float(chosen_list)
        improved = True
        last_improvement = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding a new element (if not already present)
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if is_coprime_to_all(a, chosen_list):
                    new_score = current_score + 1.0 / (512 - a)
                    if new_score > current_score + 1e-12:
                        chosen_list.append(a)
                        chosen_set.add(a)
                        current_score = new_score
                        improved = True
                        last_improvement = time.time()
                        break  # restart scan after improvement
            
            if time.time() >= time_limit:
                break
            
            # Try swapping: remove one, add another
            for i, old_a in enumerate(chosen_list):
                if time.time() >= time_limit:
                    break
                # Try removing old_a and adding a new_a
                for new_a in range(511, 0, -1):
                    if new_a in chosen_set or new_a == old_a:
                        continue
                    # Check if new_a is coprime to all except old_a
                    if is_coprime_to_all(new_a, chosen_list):
                        delta = 1.0 / (512 - new_a) - 1.0 / (512 - old_a)
                        if delta > 1e-12:
                            # Do the swap
                            chosen_list[i] = new_a
                            chosen_set.remove(old_a)
                            chosen_set.add(new_a)
                            current_score += delta
                            improved = True
                            last_improvement = time.time()
                            break  # restart after improvement
                    elif new_a == old_a:
                        continue
                    else:
                        # Check if new_a is coprime to all others (excluding old_a)
                        # This is more complex; let's just check full coprimality with the rest
                        rest = [b for j, b in enumerate(chosen_list) if j != i]
                        if is_coprime_to_all(new_a, rest):
                            delta = 1.0 / (512 - new_a) - 1.0 / (512 - old_a)
                            if delta > 1e-12:
                                chosen_list[i] = new_a
                                chosen_set.remove(old_a)
                                chosen_set.add(new_a)
                                current_score += delta
                                improved = True
                                last_improvement = time.time()
                                break
            
            if time.time() >= time_limit:
                break
        
        return chosen_list, current_score
    
    # Run local search on the best initialization
    time_limit_ls = min(deadline - 5, start_time + 130)
    best_result, best_score = local_search(best_init, time_limit_ls)
    
    # Try local search on a few other initializations if time permits
    for init in initializations[:5]:
        if time.time() > deadline - 10:
            break
        if score_float(init) >= best_score:
            continue
        result, sc = local_search(init, min(deadline - 5, start_time + 135))
        if sc > best_score:
            best_score = sc
            best_result = result
    
    # Final polish: try 2-opt style swaps
    if time.time() < deadline - 5:
        chosen_set = set(best_result)
        chosen_list = list(best_result)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try all pairs of swaps
            for i in range(len(chosen_list)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(chosen_list)):
                    if time.time() > deadline - 3:
                        break
                    old_a = chosen_list[i]
                    old_b = chosen_list[j]
                    for new_a in range(511, 0, -1):
                        if time.time() > deadline - 3:
                            break
                        if new_a in chosen_set or new_a == old_a or new_a == old_b:
                            continue
                        for new_b in range(511, 0, -1):
                            if new_b in chosen_set or new_b == old_a or new_b == old_b or new_b == new_a:
                                continue
                            rest = [x for k, x in enumerate(chosen_list) if k != i and k != j]
                            if is_coprime_to_all(new_a, rest) and is_coprime_to_all(new_b, rest) and gcd(new_a, new_b) == 1:
                                delta = (1.0/(512-new_a) + 1.0/(512-new_b)) - (1.0/(512-old_a) + 1.0/(512-old_b))
                                if delta > 1e-12:
                                    chosen_list[i] = new_a
                                    chosen_list[j] = new_b
                                    chosen_set.remove(old_a)
                                    chosen_set.remove(old_b)
                                    chosen_set.add(new_a)
                                    chosen_set.add(new_b)
                                    current_score += delta
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if current_score > best_score:
            best_score = current_score
            best_result = chosen_list
    
    return best_result
# EVOLVE-BLOCK-END
