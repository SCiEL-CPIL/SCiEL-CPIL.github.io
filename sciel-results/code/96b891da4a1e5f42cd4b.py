import time
import random
import sys

def _count_progression_diffs(S):
    """Count distinct d > 0 such that exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    S_list = list(S)
    diffs = set()
    for a in S_list:
        for b in S_list:
            d = b - a
            if d > 0:
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def _count_progression_diffs_fast(S):
    """Faster count using bitset-like approach with sorted list."""
    S_set = set(S)
    S_list = sorted(S)
    diffs = set()
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            d = b - a
            if a + 2 * d in S_set:
                diffs.add(d)
    return len(diffs)

def _greedy_init(N, MAXV):
    """Greedy construction: start with random, add elements that maximize score."""
    S = set()
    # Start with a few random elements
    initial = random.sample(range(1, MAXV + 1), 8)
    S = set(initial)
    
    while len(S) < N:
        best_elem = None
        best_score = _count_progression_diffs(S)
        # Try adding random candidates
        candidates = random.sample(range(1, MAXV + 1), min(200, MAXV - len(S)))
        for c in candidates:
            if c in S:
                continue
            test = S | {c}
            sc = _count_progression_diffs(test)
            if sc > best_score:
                best_score = sc
                best_elem = c
        if best_elem is None:
            # Pick a random one
            while True:
                c = random.randint(1, MAXV)
                if c not in S:
                    best_elem = c
                    break
        S.add(best_elem)
    return S

def _local_search(S, deadline):
    """Full single-swap hill climbing."""
    current = set(S)
    current_score = _count_progression_diffs(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        blist = list(current)
        for i, elem in enumerate(blist):
            # Try all possible replacements
            for cand in range(1, 513):
                if cand in current:
                    continue
                test = set(current)
                test.remove(elem)
                test.add(cand)
                sc = _count_progression_diffs(test)
                if sc > current_score:
                    current = test
                    current_score = sc
                    improved = True
                    break
            if time.time() > deadline:
                break
    return current, current_score

def _local_search_limited(S, deadline, max_trials=500):
    """Limited local search: try random swaps."""
    current = set(S)
    current_score = _count_progression_diffs(current)
    improved = True
    trials = 0
    while improved and time.time() < deadline and trials < max_trials:
        improved = False
        for _ in range(10):
            if time.time() > deadline:
                break
            elem = random.choice(list(current))
            cand = random.randint(1, 512)
            if cand in current:
                continue
            test = set(current)
            test.remove(elem)
            test.add(cand)
            sc = _count_progression_diffs(test)
            if sc > current_score:
                current = test
                current_score = sc
                improved = True
                trials += 1
                break
            trials += 1
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    random.seed(12345)
    N = 32
    MAXV = 512
    
    # Greedy initialization
    current = _greedy_init(N, MAXV)
    current_score = _count_progression_diffs(current)
    best = set(current)
    best_score = current_score
    
    # Iterated Local Search
    while time.time() < deadline - 10.0:
        # Local search to local optimum
        local_best, local_score = _local_search_limited(current, deadline - 5.0, max_trials=200)
        
        if local_score > best_score:
            best_score = local_score
            best = set(local_best)
        
        # Perturb: Randomly swap 2-4 elements
        perturb_strength = random.randint(2, 4)
        current = set(local_best)
        for _ in range(perturb_strength):
            elem_to_remove = random.choice(list(current))
            new_elem = random.randint(1, MAXV)
            while new_elem in current:
                new_elem = random.randint(1, MAXV)
            current.remove(elem_to_remove)
            current.add(new_elem)
    
    # Final local search on best
    final_best, final_score = _local_search_limited(best, deadline - 2.0, max_trials=500)
    if final_score > best_score:
        best = final_best
        best_score = final_score
    
    return list(best)

