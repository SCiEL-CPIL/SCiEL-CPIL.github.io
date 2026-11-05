import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    random.seed(42)

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Generate candidate prime sets
    # 1. Greedy smallest primes: {2, 3, 7}
    # 2. Skip 3, take next smallest: {2, 5, 7, 11, 13} (Sum approx 1.0)
    # 3. Skip 3, 5, take next: {2, 7, 11, 13, 17, 19, 23, 29} ...
    
    def build_candidate_set(skip_primes):
        s = Fraction(0, 1)
        final_set = []
        for p in primes:
            if p in skip_primes:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return final_set

    candidates = []
    candidates.append(build_candidate_set([])) # {2, 3, 7}
    candidates.append(build_candidate_set({3})) # {2, 5, 7, 11, 13, ...}
    candidates.append(build_candidate_set({3, 5})) # {2, 7, 11, 13, 17, ...}
    candidates.append(build_candidate_set({2})) # {3, 5, 7, 11, ...}
    candidates.append(build_candidate_set({2, 3})) # {5, 7, 11, ...}
    
    best_L = 0
    best_classes = []
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
        if not P:
            continue
            
        # Local Search on residues
        # Initial residues: random or greedy?
        # Let's try random initialization and hill climbing.
        
        residues = {p: random.randint(0, p-1) for p in P}
        max_L = 100000
        # We only care about prefix up to some limit, say 2000, to speed up?
        # The score is capped at 10^5, but likely we won't reach that.
        # Let's evaluate up to 2000 first, then if we improve, expand?
        # Or just evaluate up to 1000 since the target is 999.
        # Let's use 2000 to be safe.
        eval_limit = 2000
        
        def compute_L(res_dict):
            covered = [False] * (eval_limit + 1)
            for p, r in res_dict.items():
                for k in range(r if r > 0 else p, eval_limit + 1, p):
                    covered[k] = True
            L = 0
            for i in range(1, eval_limit + 1):
                if covered[i]:
                    L = i
                else:
                    break
            return L

        current_L = compute_L(residues)
        
        # Hill Climbing
        no_improve_count = 0
        max_steps = 5000
        step = 0
        while step < max_steps and no_improve_count < 200 and time.time() < deadline - 5:
            step += 1
            # Pick a random prime to change
            p = random.choice(P)
            old_r = residues[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            residues[p] = new_r
            new_L = compute_L(residues)
            
            if new_L > current_L:
                current_L = new_L
                no_improve_count = 0
            else:
                # Revert
                residues[p] = old_r
                no_improve_count += 1
                
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, residues[p]] for p in P]
            
    return best_classes
