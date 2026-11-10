import time
from math import gcd
from fractions import Fraction
import random

def construct():
    N = 512
    start = time.time()
    deadline = start + 140.0  # Safety margin
    
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        for m in range(p, 512, p):
            prime_factors[m].add(p)
    
    # Precompute conflict masks using prime factor overlap
    mask = [0] * 512
    for a in range(1, 512):
        if not prime_factors[a]:
            continue
        m = 0
        for pf in prime_factors[a]:
            for b in range(pf, 512, pf):
                if b != a:
                    m |= (1 << b)
        mask[a] = m
    
    def score_float(S):
        return sum(1.0 / (N - a) for a in S)
    
    def score_frac(S):
        return sum(Fraction(1, N - a) for a in S)
    
    def greedy(order, chosen_mask=0):
        S = []
        cm = chosen_mask
        for a in order:
            if (cm & mask[a]) == 0:
                S.append(a)
                cm |= (1 << a)
        return S, cm
    
    # Initial greedy in descending order
    best, best_mask = greedy(range(511, 0, -1))
    best_score = score_float(best)
    best_frac = score_frac(best)
    
    def local_search_merge_split(S, S_mask):
        """Apply merge and split moves to improve the set."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            
            # Try merging pairs: replace a, b with a*b if a*b < 512 and coprime to rest
            for i in range(len(S_list)):
                if time.time() > deadline:
                    break
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    prod = a * b
                    if prod >= N:
                        continue
                    # Check if prod is coprime to all other elements
                    rest_mask = S_mask & ~(1 << a) & ~(1 << b)
                    if (rest_mask & mask[prod]) != 0:
                        continue
                    # Compare scores
                    old_val = Fraction(1, N - a) + Fraction(1, N - b)
                    new_val = Fraction(1, N - prod)
                    if new_val > old_val:
                        S.remove(a)
                        S.remove(b)
                        S.add(prod)
                        S_mask = 0
                        for x in S:
                            S_mask |= (1 << x)
                        improved = True
                        S_list = list(S)
                        break
                if improved:
                    break
            
            # Try splitting: replace composite a with its prime factors if better
            for a in list(S):
                if time.time() > deadline:
                    break
                if len(prime_factors[a]) <= 1:
                    continue  # prime or 1
                distinct_factors = list(prime_factors[a])
                # Check if all factors are coprime to rest
                rest_mask = S_mask & ~(1 << a)
                ok = True
                for p in distinct_factors:
                    if (rest_mask & mask[p]) != 0:
                        ok = False
                        break
                if not ok:
                    continue
                old_val = Fraction(1, N - a)
                new_val = sum(Fraction(1, N - p) for p in distinct_factors)
                if new_val > old_val:
                    S.remove(a)
                    S.update(distinct_factors)
                    S_mask = 0
                    for x in S:
                        S_mask |= (1 << x)
                    improved = True
                    break
        
        return S, S_mask
    
    # Apply local search
    best, best_mask = local_search_merge_split(set(best), best_mask)
    best_score = score_float(best)
    best_frac = score_frac(best)
    
    # Multi-start: try different greedy orderings
    rng = random.Random(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed ordering
        order = list(range(511, 0, -1))
        # Randomly swap a few elements to create diversity
        for _ in range(3):
            i, j = rng.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
        
        S, S_mask = greedy(order)
        S, S_mask = local_search_merge_split(set(S), S_mask)
        s_score = score_float(S)
        if s_score > best_score + 1e-12:
            best = list(S)
            best_mask = S_mask
            best_score = s_score
            best_frac = score_frac(best)
    
    # Final polish: try single additions
    if time.time() < deadline - 1:
        best_set = set(best)
        for z in range(511, 0, -1):
            if z in best_set:
                continue
            if (best_mask & mask[z]) != 0:
                continue
            new_score = score_float(best + [z])
            if new_score > best_score + 1e-12:
                best.append(z)
                best_mask |= (1 << z)
                best_score = new_score
                best_set.add(z)
    
    return best
