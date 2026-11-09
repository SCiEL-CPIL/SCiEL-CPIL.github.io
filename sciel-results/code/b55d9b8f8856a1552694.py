"""Find a 12-element set in [0,128] minimizing the largest Sidon subset size."""
import time
import random
from itertools import combinations

DEADLINE = time.time() + 80.0  # safety margin before 90s kill


def _is_sidon(subset):
    """Check if subset has all unordered pair sums (including diagonals) unique."""
    s = sorted(subset)
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True


def _max_sidon_size(elements):
    """Find the largest subset of elements that is Sidon, via brute force."""
    n = len(elements)
    # Try subsets of size n, n-1, ..., 1
    for size in range(n, 0, -1):
        for combo in combinations(elements, size):
            if _is_sidon(list(combo)):
                return size
    return 0


def _score(elements):
    """Lower is better: the max Sidon subset size."""
    return _max_sidon_size(elements)


def construct():
    """Return exactly 12 distinct integers in [0,128] minimizing the max Sidon subset."""
    random.seed(42)
    
    best_set = None
    best_score = 12  # worst case: all 12 form a Sidon set
    
    # Initial candidates: try some structured sets
    # 1. Consecutive (likely large Sidon subset)
    # 2. Even numbers in a small range
    # 3. Numbers clustered in [0, 20]
    
    # Let's do a local search
    # Start with a random set in a small range to encourage collisions
    
    def _random_set(lo=0, hi=30):
        while True:
            s = set()
            while len(s) < 12:
                s.add(random.randint(lo, hi))
            return sorted(s)
    
    # Try several initializations
    initializations = []
    
    # Dense in small range
    for _ in range(20):
        initializations.append(_random_set(0, 25))
    
    # Dense in medium range
    for _ in range(20):
        initializations.append(_random_set(0, 40))
    
    # Arithmetic progression based
    for start in range(0, 10):
        for step in range(1, 5):
            s = [start + i * step for i in range(12)]
            if max(s) <= 128:
                initializations.append(s)
    
    # Multiplicative patterns
    for base in [2, 3, 4]:
        s = [base ** (i % 6) + i for i in range(12)]
        if len(set(s)) == 12 and max(s) <= 128:
            initializations.append(sorted(s))
    
    # Power-of-2 based with perturbations
    for _ in range(10):
        s = [random.choice([0,1,2,4,8,16,32,64]) + random.randint(0, 5) for _ in range(12)]
        # Make them distinct
        while len(set(s)) < 12:
            s[random.randint(0, 11)] = random.randint(0, 128)
        if max(s) <= 128:
            initializations.append(sorted(s))
    
    # Evaluate initializations
    for init in initializations:
        if len(set(init)) != 12 or any(x < 0 or x > 128 for x in init):
            continue
        s = _score(init)
        if s < best_score:
            best_score = s
            best_set = list(init)
            if best_score <= 4:
                break
    
    # Local search: swap one element with another in [0,128]
    if best_set is None:
        best_set = _random_set(0, 30)
        best_score = _score(best_set)
    
    best_set = list(best_set)
    cur_score = best_score
    
    # Local search with random restarts
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < DEADLINE:
        # Try swaps
        improved = False
        for i in range(12):
            if time.time() > DEADLINE:
                break
            old_val = best_set[i]
            # Try replacing with random value
            for _ in range(50):
                new_val = random.randint(0, 128)
                if new_val == old_val:
                    continue
                trial = list(best_set)
                trial[i] = new_val
                if len(set(trial)) != 12:
                    continue
                s = _score(trial)
                if s < cur_score:
                    best_set = trial
                    cur_score = s
                    if cur_score < best_score:
                        best_score = cur_score
                    improved = True
                    no_improve = 0
                    break
                elif s == cur_score and random.random() < 0.1:
                    # Accept equal with small prob (simulated annealing-like)
                    best_set = trial
                    no_improve += 1
                    break
            if no_improve > max_no_improve:
                break
        
        if not improved:
            no_improve += 1
        
        # If stuck, restart
        if no_improve > max_no_improve:
            no_improve = 0
            # Random restart, biased toward dense sets
            if random.random() < 0.5:
                best_set = _random_set(0, 25)
            else:
                best_set = _random_set(0, 50)
            if len(set(best_set)) != 12:
                best_set = _random_set(0, 30)
            cur_score = _score(best_set)
            if cur_score < best_score:
                best_score = cur_score
    
    return best_set


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Distinct: {len(set(result)) == 12}")
    print(f"In range: {all(0 <= x <= 128 for x in result)}")
    print(f"Max Sidon size: {_max_sidon_size(result)}")
    print(f"Time: {t1-t0:.2f}s")
