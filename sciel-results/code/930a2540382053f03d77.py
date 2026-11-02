"""Optimized construction for sumset/difference-set exponent lower bound."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a given set U (list of nonnegative ints containing 0)."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(num_diffs / num_sums) / denom

def _try_local_search(U, time_limit, rng):
    """Local search: try adding, removing, or shifting elements."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = list(U)
    start = time.time()
    
    # Candidate pool: try values up to some max
    max_val = max(U) + 50 if U else 50
    
    while time.time() - start < time_limit:
        improved = False
        # Try adding a new element
        for x in range(1, max_val + 1):
            if x in U:
                continue
            new_U = sorted(U + [x])
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                U = new_U
                improved = True
                max_val = max(U) + 50
                break
        if improved:
            continue
        
        # Try removing an element (if size > 2)
        if len(U) > 2:
            for i, x in enumerate(U):
                if x == 0:
                    continue
                new_U = U[:i] + U[i+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
        if improved:
            continue
        
        # Try shifting a non-zero element
        for i, x in enumerate(U):
            if x == 0:
                continue
            for delta in range(-10, 11):
                new_x = x + delta
                if new_x < 0 or new_x == 0 or new_x in U:
                    continue
                new_U = list(U)
                new_U[i] = new_x
                new_U = sorted(set(new_U))
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # No improvement found; try random restart with different initial set
        # Generate a random initial set
        size = rng.randint(3, 15)
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        new_U = [0] + candidates[:size - 1]
        new_U = sorted(set(new_U))
        s = _score_set(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
            U = new_U
        else:
            # Try a few more random sets
            for _ in range(5):
                size = rng.randint(3, 20)
                candidates = list(range(1, max_val + 20))
                rng.shuffle(candidates)
                new_U = [0] + candidates[:size - 1]
                new_U = sorted(set(new_U))
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    break
    
    return best_U, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with known good seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 2, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 3, 4, 10],
    ]
    
    best_U = seeds[0]
    best_score = _score_set(seeds[0])
    
    rng = random.Random(42)
    
    # Try local search from each seed
    for seed in seeds:
        if time.time() > deadline:
            break
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        U, score = _try_local_search(seed, time_remaining * 0.8, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Final polish: try more aggressive local search on best
    if time.time() < deadline - 2:
        time_remaining = deadline - time.time()
        U, score = _try_local_search(best_U, time_remaining, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
