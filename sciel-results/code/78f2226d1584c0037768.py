# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction

def _sieve_primes(limit):
    """Generate all primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _get_L(cov, cap=100000):
    """Find largest L such that cov[1..L] > 0."""
    for i in range(1, cap + 1):
        if cov[i] == 0:
            return i - 1
    return cap

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    primes = _sieve_primes(1000)
    # We want to maximize L. Smaller primes cover more frequently but cost more budget.
    # Larger primes are cheaper but cover less frequently.
    # Strategy: Start with a set of small primes to cover early numbers, then optimize.
    
    # Initialize with a greedy approach:
    # We want to cover 1, 2, 3, ...
    # Let's start with a set of primes that are known to be good for covering.
    # The "Erdos-Ruzsa" style construction often uses primes p where p-1 has many factors,
    # or simply small primes.
    
    # Let's use a heuristic initialization:
    # Use primes such that sum(1/p) is close to 1 but we have enough small primes to cover 1..~100.
    
    # A good starting set: Primes up to some limit.
    # Let's try to construct a set that covers 1..100 well.
    # If we pick p, we can pick r.
    
    # Initial set: Primes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Sum of 1/p for primes <= 97 is approx 1.0?
    # Sum_{p<=100} 1/p ~ 1.0.
    
    # Let's start with all primes <= 100.
    initial_primes = [p for p in primes if p <= 100]
    
    # Check budget
    budget = Fraction(0, 1)
    for p in initial_primes:
        budget += Fraction(1, p)
        
    # If budget > 1, we must remove some.
    # We want to keep the "most useful" primes. Small primes are generally more useful for covering early integers.
    # However, if we have too many small primes, we might waste budget.
    # Let's just take the smallest primes until budget is nearly full, or use a specific heuristic.
    
    # Actually, let's use a randomized initial state that is valid, then optimize.
    # Valid initial state: Empty set (L=0).
    # Better: Greedy add primes that cover the most new numbers in 1..100.
    
    classes = []
    used_primes = set()
    current_budget = Fraction(0, 1)
    
    # Greedy construction for initial coverage
    # We want to cover 1..L.
    # Let's try to add primes in order of size, or randomly?
    # Let's try to add primes such that they cover the first gap.
    
    # Simpler robust initialization:
    # Use primes up to 100. If budget > 1, remove largest primes (cheapest) first? 
    # No, removing largest primes saves the least budget. We should remove smallest primes?
    # No, smallest primes are most expensive. We want to KEEP small primes if possible?
    # Actually, to maximize L, we need good coverage of small numbers.
    # Small primes p cover 1/p fraction of numbers.
    # Let's just start with a valid set.
    
    # Let's try: Primes p where p <= 100.
    # If sum(1/p) > 1, we have a problem.
    # Sum(1/p) for p<=100 is ~ 1.0.
    # Let's calculate exactly.
    
    # To be safe, let's start with a subset.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Let's just take the first N primes such that budget <= 1.
    # This biases towards small primes.
    
    for p in primes:
        if current_budget + Fraction(1, p) <= Fraction(1, 1):
            classes.append([p, 0]) # Initial remainder 0
            used_primes.add(p)
            current_budget += Fraction(1, p)
        else:
            # If we can't add this prime, we might still add larger ones later if we remove others?
            # For now, just stop adding in this greedy pass.
            pass
    
    # This greedy approach might result in very few primes if 2, 3, 5... are too expensive.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 + 1/37 ...
    # 0.5 + 0.33 = 0.83
    # 0.83 + 0.2 = 1.03 > 1. So we can only take 2, 3? No.
    # 1/2 + 1/3 = 5/6 = 0.833.
    # Next is 5. 5/6 + 1/5 = 25/30 + 6/30 = 31/30 > 1.
    # So with this greedy "smallest first", we only get {2, 3}.
    # Coverage of 2, 3:
    # If r=0: covers 2,4,6... and 3,6,9...
    # Union: 2,3,4,6,8,9,10,12...
    # 1 is not covered. 5 is not covered.
    # L=0 or L=1? 1 is not covered by 2 or 3 (if r=0).
    # If we set r=1 for p=2: covers 1, 3, 5, 7...
    # If we set r=0 for p=3: covers 3, 6, 9...
    # 1 covered. 2 not covered. L=1.
    
    # This greedy initialization is weak.
    # Better initialization: Random valid set.
    # Or: Use a known good set for covering systems.
    # A classic covering system uses primes like 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97.
    # But sum(1/p) > 1.
    
    # Let's try a different initialization:
    # Pick random primes and remainders, ensuring budget <= 1.
    
    random.seed(42)
    
    # Try to build a set with ~20-30 primes.
    # Available budget is 1.
    # Average 1/p for p in [10, 1000] is small.
    
    # Let's try to add primes in random order, but prefer smaller ones?
    # No, let's just shuffle primes and add if budget allows.
    
    shuffled_primes = primes[:]
    random.shuffle(shuffled_primes)
    
    classes = []
    used_primes = set()
    current_budget = Fraction(0, 1)
    
    for p in shuffled_primes:
        if current_budget + Fraction(1, p) <= Fraction(1, 1):
            r = random.randint(0, p - 1)
            classes.append([p, r])
            used_primes.add(p)
            current_budget += Fraction(1, p)
            
    # This might result in a poor covering.
    # Let's run a local search to improve it.
    
    def _update_coverage(classes, cap=100000):
        cov = bytearray(cap + 1)
        for p, r in classes:
            r_mod = r % p
            # Start from first n >= 1 such that n % p == r_mod
            # If r_mod == 0, first n is p.
            # If r_mod > 0, first n is r_mod.
            start = r_mod if r_mod > 0 else p
            for n in range(start, cap + 1, p):
                cov[n] += 1
        return cov

    def _get_L(cov, cap=100000):
        for i in range(1, cap + 1):
            if cov[i] == 0:
                return i - 1
        return cap

    cov = _update_coverage(classes)
    best_L = _get_L(cov)
    
    # Local Search
    # 1. Try to change remainders of existing primes.
    # 2. Try to swap primes.
    
    # To make it fast, we only evaluate L up to a certain limit, say 2000, during search, and full 100000 at the end.
    # But the score is the exact L.
    
    # Optimization: 
    # If we change a remainder, we can update cov incrementally?
    # Or just recompute cov. Recomputing cov for K primes and N=10000 is fast in Python if using bytearray?
    # 100000 * K operations. If K=20, 2M ops. In Python, 2M ops is ~0.1s.
    # We have 110s. So we can do ~1000-2000 evaluations.
    
    # Let's use a faster evaluation: check only up to current_best_L + 100.
    # If we can't extend beyond best_L, we don't care about the rest.
    
    search_limit = 5000
    
    def _eval_L(classes, limit):
        cov = bytearray(limit + 1)
        for p, r in classes:
            r_mod = r % p
            start = r_mod if r_mod > 0 else p
            for n in range(start, limit + 1, p):
                cov[n] += 1
        # Find first gap
        for i in range(1, limit + 1):
            if cov[i] == 0:
                return i - 1
        return limit

    # Heuristic: The first gap G is the bottleneck.
    # We want to cover G.
    # If G is not covered, we need a prime p and remainder r such that G % p == r.
    # We can:
    # a) Change an existing prime's remainder to cover G.
    # b) Add a new prime p such that G % p == r (for some r) and we have budget.
    # c) Remove a prime that doesn't help cover G..G+something and add one that does.
    
    # Let's implement a simple hill climbing on L.
    
    current_classes = [list(c) for c in classes]
    current_budget = current_budget
    current_L = best_L
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find current first gap
        # We need to know where the gap is.
        # Let's compute cov up to current_L + 100 to find the gap.
        check_limit = min(current_L + 100, search_limit)
        cov_check = bytearray(check_limit + 1)
        for p, r in current_classes:
            r_mod = r % p
            start = r_mod if r_mod > 0 else p
            for n in range(start, check_limit + 1, p):
                cov_check[n] += 1
        
        gap = None
        for i in range(1, check_limit + 1):
            if cov_check[i] == 0:
                gap = i
                break
        
        if gap is None:
            # No gap in this range, maybe we can extend search?
            # For now, assume we are stuck or L is high.
            # Try random mutations to see if we can jump further.
            pass
        else:
            # We have a gap at 'gap'.
            # Try to cover it.
            
            # Strategy 1: Change remainder of an existing prime to cover 'gap'.
            # For each prime p in current_classes:
            #   We need r_new = gap % p.
            #   If we change r to r_new, does it break coverage of 1..gap-1?
            #   It might. But if it covers gap, it might extend L.
            #   We should check if the new L is > current_L.
            
            # Strategy 2: Add a new prime.
            # We need a prime p not in used_primes.
            # And r = gap % p.
            # And current_budget + 1/p <= 1.
            
            # Let's try Strategy 1 first (cheaper).
            found_improvement = False
            for i in range(len(current_classes)):
                p, old_r = current_classes[i]
                new_r = gap % p
                if new_r == old_r:
                    continue
                
                # Test this change
                test_classes = [list(c) for c in current_classes]
                test_classes[i] = [p, new_r]
                
                # Evaluate L
                # To speed up, only check up to current_L + 100
                test_limit = min(current_L + 100, search_limit)
                test_L = _eval_L(test_classes, test_limit)
                
                if test_L > current_L:
                    current_classes = test_classes
                    current_L = test_L
                    improved = True
                    found_improvement = True
                    break
            
            if found_improvement:
                continue
            
            # Strategy 2: Add a prime
            # Find available primes
            avail_primes = [p for p in primes if p not in used_primes]
            # Sort by p? Or random?
            # Smaller p means larger 1/p cost.
            # We need budget.
            # Let's try random available primes.
            if avail_primes:
                # Shuffle to avoid bias
                random.shuffle(avail_primes)
                for p in avail_primes:
                    if current_budget + Fraction(1, p) > Fraction(1, 1):
                        continue
                    
                    new_r = gap % p
                    test_classes = current_classes + [[p, new_r]]
                    test_budget = current_budget + Fraction(1, p)
                    
                    test_limit = min(current_L + 100, search_limit)
                    test_L = _eval_L(test_classes, test_limit)
                    
                    if test_L > current_L:
                        current_classes = test_classes
                        current_budget = test_budget
                        used_primes.add(p)
                        current_L = test_L
                        improved = True
                        break
            
            if improved:
                continue
            
            # Strategy 3: Swap
            # Remove one prime, add one.
            # This is expensive. Only do if we have time.
            if time.time() < deadline - 20:
                # Pick a random prime to remove
                if current_classes:
                    idx_to_remove = random.randint(0, len(current_classes) - 1)
                    p_remove, r_remove = current_classes[idx_to_remove]
                    
                    # Calculate budget saved
                    saved_budget = Fraction(1, p_remove)
                    
                    # Find a prime to add
                    # We need p_add such that 1/p_add <= saved_budget + (1 - current_budget)
                    # i.e. 1/p_add <= 1 - (current_budget - saved_budget)
                    # Let's just try available primes.
                    
                    avail_primes = [p for p in primes if p not in used_primes or p == p_remove]
                    # Exclude p_remove from "add" list if we are swapping?
                    # If we remove p_remove, it becomes available.
                    # So avail_primes should include p_remove? No, we are swapping.
                    # We remove p_remove, so it's not in used_primes anymore.
                    # We add p_add.
                    
                    # Let's just iterate over all primes not in (used_primes - {p_remove})
                    candidate_primes = [p for p in primes if p not in used_primes or p == p_remove]
                    # Wait, if p == p_remove, we are just changing remainder?
                    # That was Strategy 1.
                    # So for swap, p_add != p_remove.
                    candidate_primes = [p for p in candidate_primes if p != p_remove]
                    
                    random.shuffle(candidate_primes)
                    
                    for p_add in candidate_primes:
                        cost = Fraction(1, p_add)
                        if current_budget - saved_budget + cost > Fraction(1, 1):
                            continue
                        
                        # We want to cover 'gap' with p_add?
                        # Or maybe p_add helps cover something else?
                        # Let's try r_add = gap % p_add.
                        new_r = gap % p_add
                        
                        test_classes = [c for i, c in enumerate(current_classes) if i != idx_to_remove]
                        test_classes.append([p_add, new_r])
                        
                        test_limit = min(current_L + 100, search_limit)
                        test_L = _eval_L(test_classes, test_limit)
                        
                        if test_L > current_L:
                            # Apply swap
                            current_classes = test_classes
                            current_budget = current_budget - saved_budget + cost
                            used_primes.discard(p_remove)
                            used_primes.add(p_add)
                            current_L = test_L
                            improved = True
                            break
            
            if improved:
                continue

    # Final polish: Local search on remainders for all primes
    # Now that we have a good L, try to improve it further by tweaking remainders.
    # We can afford to be more thorough here.
    
    final_L = current_L
    final_classes = current_classes
    
    # Run a few rounds of "try all remainders for small primes"
    # This is expensive, so limit it.
    
    # Let's just do a final full evaluation to get the exact L.
    # And ensure validity.
    
    # Check budget
    final_budget = Fraction(0, 1)
    for p, r in final_classes:
        final_budget += Fraction(1, p)
        
    if final_budget > Fraction(1, 1):
        # This shouldn't happen if we maintained budget, but just in case.
        # Remove primes until valid.
        while final_budget > Fraction(1, 1) and final_classes:
            # Remove the one with largest p (smallest cost)? Or random?
            # Let's remove random to avoid bias.
            idx = random.randint(0, len(final_classes) - 1)
            p, r = final_classes.pop(idx)
            final_budget -= Fraction(1, p)
            
    # Calculate exact L
    # Use the fast _eval_L with cap=100000
    # But _eval_L uses bytearray and loops.
    # Let's use the same logic.
    
    # To be safe, let's re-evaluate with cap=100000.
    # If time allows.
    if time.time() < deadline - 5:
        final_L = _eval_L(final_classes, 100000)
        
    return [[p, r] for p, r in final_classes]
# EVOLVE-BLOCK-END
