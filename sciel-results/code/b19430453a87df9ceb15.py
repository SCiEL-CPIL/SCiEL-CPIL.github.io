"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _all_subset_sums_distinct(S):
    """Check if all subset sums of reciprocals of S are distinct using exact arithmetic."""
    k = len(S)
    if k == 0:
        return True
    # Compute all 2^k subset sums as Fractions
    sums = set()
    sums.add(Fraction(0))
    for i in range(k):
        inv = Fraction(1, S[i])
        current = list(sums)
        for s in current:
            new_s = s + inv
            if new_s in sums:
                return False
            sums.add(new_s)
    return True

def _try_add(S, candidate):
    """Try to add candidate to S and check dissociation."""
    new_S = S + [candidate]
    return _all_subset_sums_distinct(new_S)

def _greedy_construct(order, universe):
    """Greedy: iterate through 'order', add element if it keeps set dissociated."""
    S = []
    for x in order:
        if x not in universe:
            continue
        if _try_add(S, x):
            S.append(x)
    return S

def _local_search(S, universe, max_iters=200):
    """Try to improve S by swapping elements or adding new ones."""
    S_set = set(S)
    best = list(S)
    for _ in range(max_iters):
        improved = False
        # Try adding elements not in S
        for x in universe:
            if x not in S_set:
                if _try_add(best, x):
                    best = best + [x]
                    S_set.add(x)
                    improved = True
                    break
        if improved:
            continue
        # Try swapping: remove one, add another
        for i in range(len(best)):
            for x in universe:
                if x not in S_set:
                    new_S = best[:i] + best[i+1:] + [x]
                    if _all_subset_sums_distinct(new_S):
                        best = new_S
                        S_set = set(new_S)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    universe = set(range(1, 23))
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # Seed fallback
    
    # Strategy 1: Greedy ascending
    S = _greedy_construct(list(range(1, 23)), universe)
    if len(S) > len(best):
        best = S
    
    # Strategy 2: Greedy descending
    S = _greedy_construct(list(range(22, 0, -1)), universe)
    if len(S) > len(best):
        best = S
    
    # Strategy 3: Greedy by "rarity" - prefer numbers with few divisors (primes, powers of 2)
    def rarity_score(x):
        # Prefer numbers that are "independent" - large primes, powers of 2
        score = 0
        if x == 1:
            return -100  # 1 is very prone to collisions
        # Count number of divisors
        divs = sum(1 for d in range(1, x+1) if x % d == 0)
        score = -divs  # Fewer divisors = better
        # Bonus for being prime
        if x > 1:
            is_prime = all(x % i != 0 for i in range(2, int(x**0.5)+1))
            if is_prime:
                score += 10
        # Bonus for powers of 2
        if x & (x-1) == 0:
            score += 5
        return score
    
    order3 = sorted(range(1, 23), key=lambda x: -rarity_score(x))
    S = _greedy_construct(order3, universe)
    if len(S) > len(best):
        best = S
    
    # Strategy 4: Randomized greedy with local search
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Random permutation
        order = list(range(1, 23))
        random.shuffle(order)
        S = _greedy_construct(order, universe)
        # Local search to improve
        S = _local_search(S, universe, max_iters=50)
        if len(S) > len(best):
            best = S
    
    # Strategy 5: More targeted random attempts
    for attempt in range(100):
        if time.time() > deadline:
            break
        # Try biased random: include some known-good elements
        base_elements = [2, 3, 5, 7, 11, 13, 17, 19]
        remaining = [x for x in range(1, 23) if x not in base_elements]
        random.shuffle(remaining)
        order = base_elements + remaining
        random.shuffle(order)
        S = _greedy_construct(order, universe)
        S = _local_search(S, universe, max_iters=30)
        if len(S) > len(best):
            best = S
    
    # Strategy 6: Try to build up from small dissociated cores
    cores = [
        [2, 3, 5, 7],
        [3, 5, 7, 11],
        [2, 3, 5, 11, 13],
        [2, 3, 7, 11, 13, 17],
        [3, 5, 7, 11, 13, 17, 19],
    ]
    for core in cores:
        if time.time() > deadline:
            break
        S = list(core)
        # Try adding remaining elements
        for x in range(1, 23):
            if x not in S and _try_add(S, x):
                S.append(x)
        S = _local_search(S, universe, max_iters=30)
        if len(S) > len(best):
            best = S
    
    # Final: try exhaustive local search on best
    if time.time() < deadline - 10:
        best = _local_search(best, universe, max_iters=100)
    
    return sorted(best)
# EVOLVE-BLOCK-END

