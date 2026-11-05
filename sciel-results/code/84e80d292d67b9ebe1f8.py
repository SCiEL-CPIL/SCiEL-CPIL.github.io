import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def _score_with_diffs(s):
    """Return (score, set_of_diffs) for a set."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs), diffs

def _hill_climb(current_set, current_score, max_steps=5000, time_limit=None):
    """
    Perform hill climbing from current_set.
    Returns (best_set, best_score).
    """
    best_set = set(current_set)
    best_score = current_score
    steps = 0
    no_improve_steps = 0
    
    while steps < max_steps:
        if time_limit and time.time() > time_limit:
            break
        
        # Pick a random element to remove and a random element to add
        elem_out = random.choice(list(current_set))
        elem_in = random.randint(1, 512)
        
        if elem_in in current_set:
            steps += 1
            continue
            
        new_set = set(current_set)
        new_set.remove(elem_out)
        new_set.add(elem_in)
        
        new_score, _ = _score_with_diffs(new_set)
        delta = new_score - current_score
        
        if delta > 0:
            current_set = new_set
            current_score = new_score
            steps = 0
            no_improve_steps = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            no_improve_steps += 1
            steps += 1
            
        if no_improve_steps > 1000:
            break
            
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_set = None
    best_score = 0
    
    # Phase 1: Diverse Initializations and Hill Climbing
    # Run for ~30 seconds or a fixed number of restarts
    restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 60.0 and restarts < max_restarts:
        restarts += 1
        
        # Generate initial set
        # Strategy: Mix of random, interval, and structured
        r = random.random()
        if r < 0.3:
            # Random
            current_set = set(random.sample(range(1, 513), 32))
        elif r < 0.6:
            # Interval + Perturbation
            base = random.randint(1, 480)
            current_set = set(base + i for i in range(32))
            # Perturb
            for _ in range(5):
                e_out = random.choice(list(current_set))
                e_in = random.randint(1, 512)
                if e_in not in current_set:
                    current_set.remove(e_out)
                    current_set.add(e_in)
        else:
            # Two clusters
            c1 = random.randint(1, 200)
            c2 = random.randint(250, 450)
            current_set = set(c1 + i for i in range(16)) | set(c2 + i for i in range(16))
            
        if len(current_set) != 32:
            continue
            
        current_score, _ = _score_with_diffs(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Hill Climb
        hc_set, hc_score = _hill_climb(current_set, current_score, max_steps=2000, time_limit=deadline - 60.0)
        
        if hc_score > best_score:
            best_score = hc_score
            best_set = set(hc_set)
            
    # Phase 2: Simulated Annealing on the best found so far
    if best_set is not None:
        current_set = set(best_set)
        current_score, _ = _score_with_diffs(current_set)
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        stuck_count = 0
        max_stuck = 5000
        
        while time.time() < deadline - 1.0:
            elem_out = random.choice(list(current_set))
            elem_in = random.randint(1, 512)
            
            if elem_in in current_set:
                continue
                
            new_set = set(current_set)
            new_set.remove(elem_out)
            new_set.add(elem_in)
            
            new_score, _ = _score_with_diffs(new_set)
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                stuck_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > max_stuck:
                # Restart SA with a kick
                for _ in range(10):
                    e_out = random.choice(list(current_set))
                    e_in = random.randint(1, 512)
                    if e_in not in current_set:
                        current_set = set(current_set)
                        current_set.remove(e_out)
                        current_set.add(e_in)
                current_score, _ = _score_with_diffs(current_set)
                stuck_count = 0
                T = 20.0

    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END
