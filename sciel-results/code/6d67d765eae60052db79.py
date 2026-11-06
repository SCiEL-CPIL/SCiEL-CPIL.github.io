# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    N = 192
    MAX_K = 48

    # Precompute all triples of distinct elements from [1..N]
    # For efficiency, we'll check constraints on-the-fly but with memoization-like approach

    # Greedy approach with multiple strategies:
    # 1. Start with high numbers (large numbers are less likely to divide sums)
    # 2. Use a priority-based selection

    def is_valid_addition(selected, candidate):
        """Check if adding candidate to selected keeps the set valid."""
        new_set = selected + [candidate]
        
        # For each element a in new_set, check that a does not divide
        # the sum of any 3 DISTINCT OTHER elements in new_set
        for a in new_set:
            others = [x for x in new_set if x != a]
            # Check all combinations of 3 from others
            for triple in combinations(others, 3):
                if (triple[0] + triple[1] + triple[2]) % a == 0:
                    return False
        return True

    # Strategy 1: Greedy descending (start with largest, add smallest compatible)
    def greedy_descending():
        selected = []
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
        return selected

    # Strategy 2: Greedy ascending with bias toward larger numbers
    def greedy_mixed():
        selected = []
        # Try to add numbers, prioritizing those that are "harder to conflict with"
        # Larger numbers are generally safer
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
        return selected

    # Strategy 3: Randomized greedy with restarts
    def randomized_greedy(seed=None):
        import random
        rng = random.Random(seed)
        selected = []
        candidates = list(range(N, 0, -1))
        rng.shuffle(candidates)
        # But we want to prefer large numbers, so sort by a mix
        # Actually, let's try: pick from high end first with some randomization
        remaining = list(range(N, 0, -1))
        rng.shuffle(remaining)
        # Re-sort by value descending but with some noise
        scored = [(x - rng.random() * 5, x) for x in remaining]
        scored.sort(key=lambda t: -t[0])
        
        for _, x in scored:
            if time.time() > deadline:
                break
            if is_valid_addition(selected, x):
                selected.append(x)
        return selected

    # Run multiple strategies and keep the best
    best = []
    
    # Strategy 1: Pure descending greedy
    result1 = greedy_descending()
    if len(result1) > len(best):
        best = result1
    
    if time.time() > deadline:
        return best

    # Strategy 2: Try a few randomized variants
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 10:
            break
        result = randomized_greedy(seed)
        if len(result) > len(best):
            best = result

    # Strategy 3: Local improvement - try to swap out elements for better ones
    if len(best) > 0 and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current = list(best)
            for i, a in enumerate(current):
                if time.time() > deadline - 5:
                    break
                # Try removing a and adding a larger unused number
                for b in range(N, 0, -1):
                    if b in current:
                        continue
                    new_current = [x for x in current if x != a] + [b]
                    if is_valid_addition(new_current, b):
                        # Actually we need to check the whole set
                        # Let's just verify directly
                        if _verify_set(new_current):
                            best = new_current
                            improved = True
                            break

    return best

def _verify_set(S):
    """Verify that set S satisfies the constraint."""
    for a in S:
        others = [x for x in S if x != a]
        for triple in combinations(others, 3):
            if (triple[0] + triple[1] + triple[2]) % a == 0:
                return False
    return True
# EVOLVE-BLOCK-END
