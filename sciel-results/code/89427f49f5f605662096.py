import time
import random
import numpy as np

def _count_progression_diffs(S):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    S_list = list(S)
    for a in S_list:
        for b in S_list:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def _count_progression_diffs_fast(S):
    """Faster count using numpy for the inner check."""
    S_arr = np.array(sorted(S), dtype=np.int32)
    S_set = set(S)
    diffs = set()
    n = len(S_arr)
    for i in range(n):
        a = S_arr[i]
        for j in range(i + 1, n):
            d = S_arr[j] - a
            if d > 0 and (a + 2 * d) in S_set:
                diffs.add(d)
    return len(diffs)

def _local_search(current, deadline, rng):
    """Perform local search / simulated annealing from current set."""
    N = 32
    MAXV = 512
    current_score = _count_progression_diffs(current)
    best = set(current)
    best_score = current_score
    
    T = 20.0
    T_min = 0.5
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate a move: pick a random element to remove and a random new element to add
        if rng.random() < 0.7:
            # Swap
            to_remove = rng.choice(list(current))
            candidates = [x for x in range(1, MAXV + 1) if x not in current]
            to_add = rng.choice(candidates)
            if to_remove == to_add:
                continue
            new_set = set(current)
            new_set.remove(to_remove)
            new_set.add(to_add)
        else:
            # Try to find a better swap by checking a few random candidates
            improved = False
            for _ in range(5):
                to_remove = rng.choice(list(current))
                candidates = [x for x in range(1, MAXV + 1) if x not in current]
                to_add = rng.choice(candidates)
                if to_remove == to_add:
                    continue
                new_set = set(current)
                new_set.remove(to_remove)
                new_set.add(to_add)
                new_score = _count_progression_diffs(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best = set(current)
                continue
        
        new_score = _count_progression_diffs(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current = new_set
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s limit
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    # Multiple restarts
    best_overall = set(range(1, N + 1))
    best_overall_score = _count_progression_diffs(best_overall)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Random initial set
        current = set(rng.sample(range(1, MAXV + 1), N))
        current_score = _count_progression_diffs(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search
        best, best_score = _local_search(current, deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
    
    return sorted(list(best_overall))
