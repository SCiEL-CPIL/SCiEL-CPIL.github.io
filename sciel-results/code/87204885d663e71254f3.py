import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s margin
    
    # Precompute prime factors for 1..511
    def get_pf(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf_cache = {i: get_pf(i) for i in range(1, 512)}
    
    # Candidates sorted by weight descending (a descending)
    candidates = list(range(511, 0, -1))
    
    def calc_score(S):
        total = Fraction(0, 1)
        for a in S:
            total += Fraction(1, 512 - a)
        return total

    def greedy_fill(blocked_primes, current_set):
        """
        Given a set of blocked primes and a set of already chosen numbers,
        greedily add more numbers from candidates to maximize score.
        Returns the new set.
        """
        new_set = set(current_set)
        # Re-calculate blocked primes from current_set to be safe, 
        # though we pass it in, it's easier to just use the passed one 
        # if we ensure it's correct.
        
        # Actually, let's just iterate and add if compatible.
        # We need to know which primes are blocked by current_set.
        # The input blocked_primes should be the union of pf(a) for a in current_set.
        
        for a in candidates:
            if a in new_set:
                continue
            pa = pf_cache[a]
            if not pa & blocked_primes:
                new_set.add(a)
                blocked_primes = blocked_primes | pa
        return new_set

    # 1. Initial Greedy
    S = set()
    blocked = set()
    for a in candidates:
        pa = pf_cache[a]
        if not pa & blocked:
            S.add(a)
            blocked = blocked | pa
            
    best_S = S
    best_score = calc_score(S)
    
    # 2. Local Search
    while time.time() < deadline:
        improved = False
        # Try removing each element
        # Sort by weight ascending (lowest weight first) to try removing "cheap" items first
        # This might be more efficient as cheap items are more likely to be blockers
        for x in sorted(S, key=lambda a: 512 - a): 
            # x has the lowest weight in this iteration of the loop? 
            # No, sorted(S, key=lambda a: 512-a) sorts by denominator ascending, 
            # which means weight descending. 
            # We want to try removing low weight items first.
            # Weight = 1/(512-a). Low weight -> large 512-a -> small a.
            # So sort by a ascending.
            pass
            
        # Let's just iterate through all, or sort by a ascending (low weight)
        for x in sorted(S, key=lambda a: a):
            if time.time() > deadline:
                break
                
            S_temp = S - {x}
            # Recalculate blocked primes for S_temp
            blocked_temp = set()
            for y in S_temp:
                blocked_temp |= pf_cache[y]
                
            S_new = greedy_fill(blocked_temp, S_temp)
            score_new = calc_score(S_new)
            
            if score_new > best_score:
                best_score = score_new
                best_S = S_new
                improved = True
                # Break and restart with new best_S
                break
        
        if not improved:
            break
            
    return list(best_S)

# EVOLVE-BLOCK-END
