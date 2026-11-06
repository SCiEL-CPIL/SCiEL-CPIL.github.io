import time
import random
import math
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    
    # Candidate prime sets where sum(1/p) <= 1
    # We generate multiple diverse sets to explore different structural possibilities.
    
    candidate_sets = []
    
    # Set 1: Greedy smallest primes (excluding 2 to save budget for density? No, 2 is very dense. 
    # But 2 covers half the numbers. 3 covers 1/3. 
    # Let's try a set that includes 2, 3, 5, 7 and then fills with larger primes.
    # Sum(1/2+1/3+1/5+1/7) = 0.5+0.33+0.2+0.14 = 1.17 > 1. 
    # So we can't have {2,3,5,7}.
    # Max sum for {2,3,5} is 1.03 > 1. 
    # So {2,3} is 0.833. Remaining budget 0.166. 
    # {2,3,7} = 0.5+0.333+0.142 = 0.975. Remaining 0.025. Next prime 11 (0.09) too big. 
    # So {2,3,7} + primes >= 40? 1/40 ~ 0.025. 
    # Let's systematically generate sets.
    
    def generate_sets(prime_list):
        sets = []
        
        # Strategy A: Include 2, exclude 3. Fill with smallest available.
        # Budget 1 - 0.5 = 0.5.
        # Add 5 (0.2) -> 0.7. Add 7 (0.14) -> 0.84. Add 11 (0.09) -> 0.93. Add 13 (0.07) -> 1.0 > 1.
        # So {2, 5, 7, 11} is valid. Sum = 0.93.
        # Can we add more? Next is 17 (0.058). 0.93+0.058 = 0.988. Valid.
        # Next 19 (0.052). 0.988+0.052 = 1.04 > 1.
        # So {2, 5, 7, 11, 17} is a good set.
        
        # Strategy B: Exclude 2. Include 3, 5, 7.
        # Sum 1/3+1/5+1/7 = 0.33+0.2+0.14 = 0.67.
        # Add 11 (0.09) -> 0.76. Add 13 (0.07) -> 0.83. Add 17 (0.05) -> 0.88. 
        # Add 19 (0.05) -> 0.93. Add 23 (0.043) -> 0.97. Add 29 (0.034) -> 1.01 > 1.
        # So {3, 5, 7, 11, 13, 17, 19, 23} is valid.
        
        # Strategy C: Randomized Greedy.
        # Sort primes by 1/p descending (i.e., p ascending).
        # Pick primes randomly with probability proportional to 1/p, ensuring sum <= 1.
        
        sets.append([2, 5, 7, 11, 17])
        sets.append([3, 5, 7, 11, 13, 17, 19, 23])
        
        # Generate 5 random sets
        for _ in range(5):
            s = 0
            current_set = []
            available = primes[:]
            random.shuffle(available)
            # Bias towards smaller primes? 
            # Let's just pick smallest first but skip some randomly.
            for p in sorted(available):
                if s + 1/p <= 1:
                    # 50% chance to take it, 50% to skip (to create diversity)
                    if random.random() < 0.5 or len(current_set) == 0:
                        current_set.append(p)
                        s += 1/p
                else:
                    # If adding p exceeds budget, we can't add it or any larger prime.
                    # But maybe we skipped a smaller one? 
                    # Since we iterate sorted, if p doesn't fit, no larger prime will fit 
                    # IF we were taking all. But we skipped some. 
                    # Actually, if s + 1/p > 1, then for any q > p, 1/q < 1/p, so s + 1/q might be <= 1.
                    # So we can't break immediately.
                    pass
            
            if current_set:
                # Verify exact sum
                exact_sum = sum(Fraction(1, p) for p in current_set)
                if exact_sum <= 1:
                    sets.append(current_set)
                    
        return sets

    valid_sets = generate_sets(primes)
    
    if not valid_sets:
        # Fallback
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []
    
    L_MAX = 20000 # Cap for coverage calculation.
    
    # Time allocation
    time_per_set = 25.0 # 6 sets * 25s = 150s. Tight. Let's use 20s.
    time_per_set = 20.0
    
    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        P_sorted = sorted(P)
        primes_in_set = P_sorted
        
        # State
        residues = {p: 0 for p in primes_in_set}
        cover_count = np.zeros(L_MAX + 1, dtype=np.int8)
        
        def update_cover(p, r, delta):
            first = r if r > 0 else p
            if first > L_MAX:
                return
            indices = np.arange(first, L_MAX + 1, p)
            cover_count[indices] += delta

        def get_L():
            zeros = np.where(cover_count[1:L_MAX+1] == 0)[0]
            if len(zeros) == 0:
                return L_MAX
            return zeros[0]
        
        # Initialization: Greedy
        assigned_primes = set()
        for n in range(1, L_MAX + 1):
            if cover_count[n] == 0:
                chosen_p = None
                for p in P_sorted:
                    if p not in assigned_primes:
                        chosen_p = p
                        break
                if chosen_p is None:
                    break
                r = n % chosen_p
                residues[chosen_p] = r
                assigned_primes.add(chosen_p)
                update_cover(chosen_p, r, 1)
            
            if time.time() > set_deadline:
                break
        
        current_L = get_L()
        best_set_L = current_L
        best_set_residues = residues.copy()
        
        # Optimization Loop: Hybrid Local Search
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        no_improve_steps = 0
        max_no_improve = 5000 # Restart if stuck
        
        while time.time() < set_deadline:
            if no_improve_steps > max_no_improve:
                # Perturb: Change 5 random residues
                for _ in range(5):
                    p = random.choice(primes_in_set)
                    r_old = residues[p]
                    r_new = random.randint(0, p - 1)
                    if r_new != r_old:
                        update_cover(p, r_old, -1)
                        update_cover(p, r_new, 1)
                        residues[p] = r_new
                current_L = get_L()
                no_improve_steps = 0
                T = 5.0 # Reheat
                
            # Choose move type
            # 70% Repair, 30% Random
            if random.random() < 0.7:
                # Repair: Target first gap
                if current_L >= L_MAX:
                    break
                target = current_L + 1
                # Find prime that minimizes damage
                best_p = None
                best_cost = float('inf')
                best_r_new = None
                
                # Sample a few primes to save time? Or all?
                # Number of primes is small (~10-20). So check all.
                for p in primes_in_set:
                    r_old = residues[p]
                    r_new = target % p
                    if r_new == r_old:
                        continue
                    
                    # Calculate cost: how many k in [1, current_L] are uncovered?
                    # This is expensive to calculate exactly for every p.
                    # Approximation: 
                    # The numbers covered by p are r_old, r_old+p, ...
                    # The numbers covered by p (new) are r_new, r_new+p, ...
                    # Damage = | {k < current_L : k % p == r_old AND k % p != r_new} |
                    # Gain   = | {k < current_L : k % p == r_new AND k % p != r_old} |
                    # Net change in coverage for k < current_L.
                    
                    # We can compute this quickly using precomputed counts?
                    # Let's just estimate: 
                    # If we change r, we lose coverage for ~current_L/p numbers and gain ~current_L/p numbers.
                    # The net effect is small. 
                    # However, we specifically WANT to cover `target`.
                    # So the "Gain" includes `target` (which is not in 1..current_L, so it doesn't reduce L directly, it extends it).
                    # The "Damage" is the number of k < current_L that were covered by r_old and NOT by r_new.
                    
                    # To be safe and fast, let's just pick the prime with the smallest p? 
                    # Smallest p has highest density, so changing it might have larger impact.
                    # Or largest p? Smallest density, so changing it has least impact on existing coverage.
                    # Strategy: Pick the LARGEST prime to minimize damage to existing prefix.
                    
                    if best_p is None or p > best_p:
                        best_p = p
                        best_cost = 0 # We assume large p is good
                        best_r_new = r_new
                
                if best_p:
                    p = best_p
                    r_old = residues[p]
                    r_new = best_r_new
                    if r_new != r_old:
                        update_cover(p, r_old, -1)
                        update_cover(p, r_new, 1)
                        residues[p] = r_new
                        new_L = get_L()
                        
                        if new_L > current_L:
                            current_L = new_L
                            no_improve_steps = 0
                            if current_L > best_set_L:
                                best_set_L = current_L
                                best_set_residues = residues.copy()
                        else:
                            # Accept with prob exp(-delta/T)
                            delta = new_L - current_L
                            if delta < 0:
                                prob = math.exp(delta / T)
                                if random.random() > prob:
                                    # Reject
                                    update_cover(p, r_new, -1)
                                    update_cover(p, r_old, 1)
                                    residues[p] = r_old
                            no_improve_steps += 1
            else:
                # Random Move
                p = random.choice(primes_in_set)
                r_old = residues[p]
                r_new = random.randint(0, p - 1)
                if r_new != r_old:
                    update_cover(p, r_old, -1)
                    update_cover(p, r_new, 1)
                    residues[p] = r_new
                    new_L = get_L()
                    
                    if new_L > current_L:
                        current_L = new_L
                        no_improve_steps = 0
                        if current_L > best_set_L:
                            best_set_L = current_L
                            best_set_residues = residues.copy()
                    else:
                        delta = new_L - current_L
                        if delta < 0:
                            prob = math.exp(delta / T)
                            if random.random() > prob:
                                update_cover(p, r_new, -1)
                                update_cover(p, r_old, 1)
                                residues[p] = r_old
                        no_improve_steps += 1
                        
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[p, best_set_residues[p]] for p in best_set_residues]

    if not best_result_global:
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result_global
