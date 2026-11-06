"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s/150s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit=1000):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _compute_coverage(primes, residues, cap=100000):
    """
    Returns the largest L such that every integer 1..L is covered.
    An integer n is covered if there exists (p, r) in the list such that n % p == r.
    """
    if not primes:
        return 0
    
    # For speed, we can check sequentially.
    # Since cap is 10^5, and we have ~168 primes, checking 10^5 * 168 is ~1.68e7 ops, which is fine in Python if optimized or done in C via numpy.
    # However, doing it in pure Python loop might be slow if called many times.
    # Let's use a bitset or boolean array approach for the current check.
    
    covered = [False] * (cap + 1)
    for p, r in zip(primes, residues):
        # Mark all n in 1..cap such that n % p == r
        # The first number >= 1 with n % p == r is:
        if r == 0:
            start = p
        else:
            start = r
        # Ensure start >= 1
        if start < 1:
            start += p * ((1 - start + p - 1) // p) # Not quite, let's just find first valid
            # Simpler:
            # n = k*p + r. We want n >= 1.
            # If r >= 1, start = r.
            # If r == 0, start = p.
            pass
        
        # Correct logic for start:
        if r == 0:
            start = p
        else:
            start = r
            
        for n in range(start, cap + 1, p):
            covered[n] = True
            
    # Find longest prefix of True starting from 1
    L = 0
    while L < cap and covered[L + 1]:
        L += 1
    return L

def construct():
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to a known valid solution (the seed)
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]

    all_primes = _get_primes(1000)
    
    # Strategy: 
    # 1. Start with a greedy set of primes that fits the budget.
    #    The seed uses primes from 13 upwards. Sum(1/p) for p=13..997 is approx 0.92.
    #    We can potentially use smaller primes (2, 3, 5, 7) if we drop some larger ones, 
    #    or keep the same set but optimize residues.
    #    Using small primes is generally better for coverage density.
    #    Let's try to include 2, 3, 5, 7.
    
    # Calculate budget usage for seed
    # Seed primes: 13, 17, 19, ...
    # Let's try a set that includes small primes.
    # Sum(1/2 + 1/3 + 1/5 + 1/7) = 0.5 + 0.333 + 0.2 + 0.142 = 1.175 > 1.
    # So we can't take all of them.
    # Best combination?
    # 1/2 + 1/3 + 1/5 = 0.5 + 0.333 + 0.2 = 1.033 > 1.
    # 1/2 + 1/3 + 1/7 = 0.5 + 0.333 + 0.142 = 0.975 < 1.
    # 1/2 + 1/5 + 1/7 = 0.5 + 0.2 + 0.142 = 0.842.
    # 1/3 + 1/5 + 1/7 = 0.333 + 0.2 + 0.142 = 0.675.
    
    # The seed uses primes >= 13. Sum is ~0.92.
    # Can we do better by swapping some large primes for smaller ones?
    # e.g., replace 997 (1/997 ~ 0.001) with 7 (1/7 ~ 0.142)? No, budget would explode.
    # We need to maintain sum(1/p) <= 1.
    
    # Let's initialize with the seed's primes, but maybe try to include 2?
    # If we include 2, we need to drop enough reciprocals to make room.
    # 1/2 = 0.5. Remaining budget 0.5.
    # Primes > 2 with sum of reciprocals <= 0.5.
    # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 ...
    # 1/3+1/5+1/7 = 0.675 > 0.5.
    # 1/3+1/5 = 0.533 > 0.5.
    # 1/3+1/7 = 0.475.
    # So {2, 3, 7} is a valid set with sum 0.975.
    # Coverage of {2, 3, 7}:
    # 1: 1%2=1, 1%3=1, 1%7=1. If residues are (1,1,1), 1 is covered.
    # 2: 2%2=0, 2%3=2, 2%7=2.
    # 3: 3%2=1, 3%3=0, 3%7=3.
    # 4: 4%2=0, 4%3=1, 4%7=4.
    # 5: 5%2=1, 5%3=2, 5%7=5.
    # 6: 6%2=0, 6%3=0, 6%7=6.
    # 7: 7%2=1, 7%3=1, 7%7=0.
    # 8: 8%2=0, 8%3=2, 8%7=1.
    # 9: 9%2=1, 9%3=0, 9%7=2.
    # 10: 10%2=0, 10%3=1, 10%7=3.
    # 11: 11%2=1, 11%3=2, 11%7=4.
    # 12: 12%2=0, 12%3=0, 12%7=5.
    # 13: 13%2=1, 13%3=1, 13%7=6.
    # 14: 14%2=0, 14%3=2, 14%7=0.
    # 15: 15%2=1, 15%3=0, 15%7=1.
    # 16: 16%2=0, 16%3=1, 16%7=2.
    # 17: 17%2=1, 17%3=2, 17%7=3.
    # 18: 18%2=0, 18%3=0, 18%7=4.
    # 19: 19%2=1, 19%3=1, 19%7=5.
    # 20: 20%2=0, 20%3=2, 20%7=6.
    # 21: 21%2=1, 21%3=0, 21%7=0.
    # 22: 22%2=0, 22%3=1, 22%7=1.
    # 23: 23%2=1, 23%3=2, 23%7=2.
    # 24: 24%2=0, 24%3=0, 24%7=3.
    # 25: 25%2=1, 25%3=1, 25%7=4.
    # 26: 26%2=0, 26%3=2, 26%7=5.
    # 27: 27%2=1, 27%3=0, 27%7=6.
    # 28: 28%2=0, 28%3=1, 28%7=0.
    # 29: 29%2=1, 29%3=2, 29%7=1.
    # 30: 30%2=0, 30%3=0, 30%7=2.
    # 31: 31%2=1, 31%3=1, 31%7=3.
    # 32: 32%2=0, 32%3=2, 32%7=4.
    # 33: 33%2=1, 33%3=0, 33%7=5.
    # 34: 34%2=0, 34%3=1, 34%7=6.
    # 35: 35%2=1, 35%3=2, 35%7=0.
    # 36: 36%2=0, 36%3=0, 36%7=1.
    # 37: 37%2=1, 37%3=1, 37%7=2.
    # 38: 38%2=0, 38%3=2, 38%7=3.
    # 39: 39%2=1, 39%3=0, 39%7=4.
    # 40: 40%2=0, 40%3=1, 40%7=5.
    # 41: 41%2=1, 41%3=2, 41%7=6.
    # 42: 42%2=0, 42%3=0, 42%7=0.
    # 43: 43%2=1, 43%3=1, 43%7=1.
    # 44: 44%2=0, 44%3=2, 44%7=2.
    # 45: 45%2=1, 45%3=0, 45%7=3.
    # 46: 46%2=0, 46%3=1, 46%7=4.
    # 47: 47%2=1, 47%3=2, 47%7=5.
    # 48: 48%2=0, 48%3=0, 48%7=6.
    # 49: 49%2=1, 49%3=1, 49%7=0.
    # 50: 50%2=0, 50%3=2, 50%7=1.
    # 51: 51%2=1, 51%3=0, 51%7=2.
    # 52: 52%2=0, 52%3=1, 52%7=3.
    # 53: 53%2=1, 53%3=2, 53%7=4.
    # 54: 54%2=0, 54%3=0, 54%7=5.
    # 55: 55%2=1, 55%3=1, 55%7=6.
    # 56: 56%2=0, 56%3=2, 56%7=0.
    # 57: 57%2=1, 57%3=0, 57%7=1.
    # 58: 58%2=0, 58%3=1, 58%7=2.
    # 59: 59%2=1, 59%3=2, 59%7=3.
    # 60: 60%2=0, 60%3=0, 60%7=4.
    # 61: 61%2=1, 61%3=1, 61%7=5.
    # 62: 62%2=0, 62%3=2, 62%7=6.
    # 63: 63%2=1, 63%3=0, 63%7=0.
    # 64: 64%2=0, 64%3=1, 64%7=1.
    # 65: 65%2=1, 65%3=2, 65%7=2.
    # 66: 66%2=0, 66%3=0, 66%7=3.
    # 67: 67%2=1, 67%3=1, 67%7=4.
    # 68: 68%2=0, 68%3=2, 68%7=5.
    # 69: 69%2=1, 69%3=0, 69%7=6.
    # 70: 70%2=0, 70%3=1, 70%7=0.
    # 71: 71%2=1, 71%3=2, 71%7=1.
    # 72: 72%2=0, 72%3=0, 72%7=2.
    # 73: 73%2=1, 73%3=1, 73%7=3.
    # 74: 74%2=0, 74%3=2, 74%7=4.
    # 75: 75%2=1, 75%3=0, 75%7=5.
    # 76: 76%2=0, 76%3=1, 76%7=6.
    # 77: 77%2=1, 77%3=2, 77%7=0.
    # 78: 78%2=0, 78%3=0, 78%7=1.
    # 79: 79%2=1, 79%3=1, 79%7=2.
    # 80: 80%2=0, 80%3=2, 80%7=3.
    # 81: 81%2=1, 81%3=0, 81%7=4.
    # 82: 82%2=0, 82%3=1, 82%7=5.
    # 83: 83%2=1, 83%3=2, 83%7=6.
    # 84: 84%2=0, 84%3=0, 84%7=0.
    # 85: 85%2=1, 85%3=1, 85%7=1.
    # 86: 86%2=0, 86%3=2, 86%7=2.
    # 87: 87%2=1, 87%3=0, 87%7=3.
    # 88: 88%2=0, 88%3=1, 88%7=4.
    # 89: 89%2=1, 89%3=2, 89%7=5.
    # 90: 90%2=0, 90%3=0, 90%7=6.
    # 91: 91%2=1, 91%3=1, 91%7=0.
    # 92: 92%2=0, 92%3=2, 92%7=1.
    # 93: 93%2=1, 93%3=0, 93%7=2.
    # 94: 94%2=0, 94%3=1, 94%7=3.
    # 95: 95%2=1, 95%3=2, 95%7=4.
    # 96: 96%2=0, 96%3=0, 96%7=5.
    # 97: 97%2=1, 97%3=1, 97%7=6.
    # 98: 98%2=0, 98%3=2, 98%7=0.
    # 99: 99%2=1, 99%3=0, 99%7=1.
    # 100: 100%2=0, 100%3=1, 100%7=2.
    
    # It seems {2, 3, 7} with residues (1, 1, 1) covers 1..100?
    # Let's check 1..100 for {2,3,7} with residues (1,1,1).
    # n is covered if n%2==1 OR n%3==1 OR n%7==1.
    # 1: 1,1,1 -> Yes
    # 2: 0,2,2 -> No. 2 is NOT covered.
    # So residues (1,1,1) fail at 2.
    # We need to cover 2.
    # 2%2=0, 2%3=2, 2%7=2.
    # So we need r2=0 or r3=2 or r7=2.
    
    # This is a constraint satisfaction problem.
    # Local search is the way to go.
    
    # Initialize with seed
    primes = [p for p in range(13, 1001) if _is_prime(p)]
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    best_L = _compute_coverage(primes, residues)
    best_primes = primes[:]
    best_residues = residues[:]
    
    # Randomize initial state slightly? Or just start from seed.
    # Let's try to improve.
    
    # We can also try different prime sets.
    # But first, let's optimize residues for the current set.
    
    rng = random.Random(42)
    
    # Local search loop
    while time.time() - _START_TIME < _TIME_LIMIT:
        # Pick a random prime index
        idx = rng.randrange(len(primes))
        p = primes[idx]
        old_r = residues[idx]
        
        # Try a new residue
        # To be efficient, we might only try residues that cover the first uncovered number.
        # Find first uncovered number
        L = best_L
        # Check if L+1 is covered
        covered_next = False
        for j, pj in enumerate(primes):
            if (L + 1) % pj == residues[j]:
                covered_next = True
                break
        
        if not covered_next:
            # We need to change a residue to cover L+1
            # The prime at idx can cover L+1 if we set residues[idx] = (L+1) % p
            new_r = (L + 1) % p
            if new_r != old_r:
                residues[idx] = new_r
                new_L = _compute_coverage(primes, residues)
                if new_L > best_L:
                    best_L = new_L
                    best_primes = primes[:]
                    best_residues = residues[:]
                else:
                    # Revert if not improved (or accept with some probability for diversity?)
                    # Simple hill climbing: revert
                    residues[idx] = old_r
        else:
            # Already covered, try random perturbation
            new_r = rng.randrange(p)
            if new_r != old_r:
                residues[idx] = new_r
                new_L = _compute_coverage(primes, residues)
                if new_L > best_L:
                    best_L = new_L
                    best_primes = primes[:]
                    best_residues = residues[:]
                else:
                    residues[idx] = old_r
                    
    return [[p, r] for p, r in zip(best_primes, best_residues)]
# EVOLVE-BLOCK-END
