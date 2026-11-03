"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the score for a set U of nonnegative integers."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    
    # Compute sumset and difference set sizes
    # For efficiency, use sets
    sumset = set()
    diffset = set()
    
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    # The score is 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    # Note: |U-U| >= |U+U| always for sets containing 0, so ratio >= 1, log >= 0
    max_u = max(U)
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / denom
    return score


def _construct_twin_prime_ladder(start_prime, num_pairs):
    """
    Construct U as cumulative sum of twin prime pairs.
    U = {0, p1, p1+(p1+2), p1+(p1+2)+p2, ...}
    where (p1, p1+2), (p2, p2+2), ... are consecutive twin prime pairs.
    """
    def is_prime(n):
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
    
    U = [0]
    current = 0
    p = start_prime
    
    # Find twin prime pairs
    pairs_found = 0
    while pairs_found < num_pairs:
        if is_prime(p) and is_prime(p + 2):
            # Add the first element of the twin prime pair
            current += p
            U.append(current)
            pairs_found += 1
            # Skip ahead
            p += 2
        else:
            p += 1
    
    return U


def _construct_exhaustive_small(n_max):
    """
    Exhaustive search over small sets containing 0.
    Try all subsets of {0, 1, ..., n_max-1} that contain 0.
    """
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # For small n_max, we can be more exhaustive
    # Try sets of size 3 to 6
    for size in range(3, min(n_max, 7)):
        # Generate combinations
        for combo in combinations(range(1, n_max), size - 1):
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score


def _construct_random_search(n_max, num_trials, time_limit):
    """
    Random search over sets of varying sizes.
    """
    best_score = 0.0
    best_set = [0, 1, 3]
    
    deadline = time.time() + time_limit
    random.seed(42)
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Random size
        size = random.randint(3, min(n_max, 15))
        
        # Generate random set
        if size <= n_max:
            candidates = list(range(1, n_max))
            random.shuffle(candidates)
            U = [0] + candidates[:size - 1]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score


def _construct_greedy_extension(base, max_size, time_limit):
    """
    Greedy extension: start from base, add elements one by one to maximize score.
    """
    U = base[:]
    deadline = time.time() + time_limit
    
    for size in range(len(U) + 1, max_size + 1):
        if time.time() > deadline:
            break
        
        best_add = None
        best_score = _score_set(U)
        
        # Try adding each candidate
        max_u = max(U)
        for candidate in range(max_u + 1, max_u + 20):
            U_new = U + [candidate]
            s = _score_set(U_new)
            if s > best_score:
                best_score = s
                best_add = candidate
        
        if best_add is not None:
            U.append(best_add)
        else:
            break
    
    return U, _score_set(U)


def construct():
    """
    Main construct function.
    Returns a list of nonnegative integers containing 0 that maximizes the score.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Strategy 1: Exhaustive search over small sets
    # This is fast for small n_max
    if time.time() < deadline - 30:
        best_small, score_small = _construct_exhaustive_small(10)
    else:
        best_small = [0, 1, 3]
        score_small = _score_set(best_small)
    
    # Strategy 2: Twin prime ladder with grid search
    best_ladder = [0, 1, 3]
    score_ladder = score_small
    
    if time.time() < deadline - 30:
        # Grid search over starting prime and number of pairs
        for start_p in [3, 5, 11, 17, 29, 41, 59, 71]:
            for num_pairs in range(2, 12):
                if time.time() > deadline - 30:
                    break
                U = _construct_twin_prime_ladder(start_p, num_pairs)
                s = _score_set(U)
                if s > score_ladder:
                    score_ladder = s
                    best_ladder = U
    
    # Strategy 3: Random search
    best_random = best_small
    score_random = max(score_small, score_ladder)
    
    if time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        best_random, score_random = _construct_random_search(20, 5000, remaining)
    
    # Strategy 4: Greedy extension from best found
    best_final = best_small
    score_final = max(score_small, score_ladder, score_random)
    
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        # Try greedy extension from each of the best candidates
        for base in [best_small, best_ladder, best_random]:
            if time.time() > deadline - 5:
                break
            U_ext, s_ext = _construct_greedy_extension(base, 20, remaining / 3)
            if s_ext > score_final:
                score_final = s_ext
                best_final = U_ext
    
    # Return the best result
    return best_final

# EVOLVE-BLOCK-END
