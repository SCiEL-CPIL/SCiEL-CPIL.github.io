"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_set(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S for some a."""
    S = set(S)
    diffs = set()
    for a in S:
        for d in range(1, 513):
            if (a + d) in S and (a + 2*d) in S:
                diffs.add(d)
    return len(diffs)

def _construct():
    deadline = time.time() + 130.0
    n = 32
    max_val = 512
    
    # Pre-compute a "dense" seed: numbers that participate in many short APs
    # Start with a structured set that has good 3-AP coverage
    # Use a greedy approach to build initial set
    
    # Initial seed: combine blocks with different spacings to maximize difference diversity
    # Strategy: use numbers that are well-spread but also have local clusters
    random.seed(42)
    
    # Start with a set that has good structure
    # Use a base set of well-distributed numbers
    candidates = list(range(1, max_val + 1))
    
    # Greedy initialization: pick numbers that maximize new differences
    current_set = set()
    # Start with some anchor points
    anchors = [1, 64, 128, 192, 256, 320, 384, 448, 512]
    for a in anchors:
        current_set.add(a)
    
    # Add more greedily
    while len(current_set) < n:
        best_candidate = None
        best_score = -1
        # Sample candidates for speed
        sample = random.sample([x for x in candidates if x not in current_set], min(200, len([x for x in candidates if x not in current_set])))
        for cand in sample:
            trial = current_set | {cand}
            s = _score_set(trial)
            if s > best_score:
                best_score = s
                best_candidate = cand
        if best_candidate is not None:
            current_set.add(best_candidate)
        else:
            # Fallback: add random
            remaining = [x for x in candidates if x not in current_set]
            if remaining:
                current_set.add(random.choice(remaining))
            else:
                break
    
    best_set = list(current_set)
    best_score = _score_set(best_set)
    
    # Simulated annealing
    T = 10.0
    cooling = 0.999
    T_min = 0.01
    step_count = 0
    
    while time.time() < deadline and T > T_min:
        step_count += 1
        # Generate a move: swap one element
        if random.random() < 0.7:
            # Swap
            idx = random.randint(0, len(best_set) - 1)
            old_elem = best_set[idx]
            new_elem = random.randint(1, max_val)
            if new_elem in best_set:
                continue
            trial = best_set.copy()
            trial[idx] = new_elem
            trial_set = set(trial)
            new_score = _score_set(trial_set)
            delta = new_score - best_score
            if delta >= 0 or random.random() < min(1.0, exp(-delta / T)):
                best_set = trial
                best_score = new_score
        else:
            # Single move: replace one element
            idx = random.randint(0, len(best_set) - 1)
            old_elem = best_set[idx]
            new_elem = random.randint(1, max_val)
            if new_elem in best_set:
                continue
            trial = best_set.copy()
            trial[idx] = new_elem
            trial_set = set(trial)
            new_score = _score_set(trial_set)
            delta = new_score - best_score
            if delta >= 0 or random.random() < min(1.0, exp(-delta / T)):
                best_set = trial
                best_score = new_score
        
        T *= cooling
    
    return sorted(best_set)

def exp(x):
    import math
    return math.exp(x)

def construct():
    return _construct()
# EVOLVE-BLOCK-END
