"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
    s_set = set(s)
    diffs = set()
    for a in s:
        for b in s:
            if b == a:
                continue
            d = b - a
            if d <= 0:
                continue
            if (a + 2 * d) in s_set:
                diffs.add(d)
    return len(diffs)

def _greedy_init(n=32, max_val=512, rng=None):
    """Greedy construction: add elements that maximize new distinct differences."""
    if rng is None:
        rng = random.Random(42)
    s = set()
    candidates = list(range(1, max_val + 1))
    rng.shuffle(candidates)
    # Start with a small random seed
    for c in candidates[:5]:
        s.add(c)
    
    remaining = candidates[5:]
    while len(s) < n:
        best_c = None
        best_score = -1
        current_score = _score_set(s)
        for c in remaining:
            s.add(c)
            sc = _score_set(s)
            s.remove(c)
            if sc > best_score:
                best_score = sc
                best_c = c
        s.add(best_c)
        remaining.remove(best_c)
    return sorted(s)

def _simulated_annealing(initial_set, max_time=140.0, rng=None):
    """Simulated annealing to improve the set."""
    if rng is None:
        rng = random.Random(42)
    
    s = set(initial_set)
    current_score = _score_set(s)
    best_set = set(s)
    best_score = current_score
    
    start_time = time.time()
    T = 10.0
    cooling = 0.99999
    min_T = 0.01
    
    while T > min_T:
        if time.time() - start_time > max_time:
            break
        
        # Try a random swap: remove one element, add another
        elems = list(s)
        i = rng.randrange(len(elems))
        old_elem = elems[i]
        
        # Pick a new element from 1..512 not in s
        new_elem = rng.randint(1, 512)
        attempts = 0
        while new_elem in s and attempts < 50:
            new_elem = rng.randint(1, 512)
            attempts += 1
        if new_elem in s:
            continue
        
        s.remove(old_elem)
        s.add(new_elem)
        new_score = _score_set(s)
        
        delta = new_score - current_score
        if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(s)
        else:
            # Revert
            s.remove(new_elem)
            s.add(old_elem)
        
        T *= cooling
    
    return sorted(best_set), best_score

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Try multiple random seeds for greedy init + annealing
    best_result = None
    best_score = -1
    
    # First, do a greedy construction
    greedy_set = _greedy_init(n=32, max_val=512)
    greedy_score = _score_set(greedy_set)
    best_result = greedy_set
    best_score = greedy_score
    
    # Now do simulated annealing starting from greedy, with time budget
    time_left = deadline - time.time()
    if time_left > 5:
        anneal_time = max(1, time_left - 5)
        annealed_set, annealed_score = _simulated_annealing(greedy_set, max_time=anneal_time)
        if annealed_score > best_score:
            best_result = annealed_set
            best_score = annealed_score
    
    # Try a few more random initializations if time permits
    time_left = deadline - time.time()
    for seed in [123, 456, 789]:
        if time_left < 10:
            break
        rng = random.Random(seed)
        init_set = _greedy_init(n=32, max_val=512, rng=rng)
        init_score = _score_set(init_set)
        if init_score > best_score:
            best_result = init_set
            best_score = init_score
        
        time_left = deadline - time.time()
        if time_left > 15:
            anneal_time = max(1, time_left - 10)
            annealed_set, annealed_score = _simulated_annealing(init_set, max_time=anneal_time, rng=rng)
            if annealed_score > best_score:
                best_result = annealed_set
                best_score = annealed_score
    
    return best_result
# EVOLVE-BLOCK-END
