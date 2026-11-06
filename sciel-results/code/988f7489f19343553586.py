import time
import random
import sys

def _score_set(S):
    """
    Calculate the number of distinct d > 0 such that 
    a, a+d, a+2d are in S for some a.
    S is a set of integers.
    """
    diffs = set()
    # Convert to list for iteration
    elements = list(S)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(n):
            if i == j:
                continue
            b = elements[j]
            if a > b:
                continue
            # d = b - a
            c = 2 * b - a
            if c in S:
                d = b - a
                if d > 0:
                    diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_score = -1
    best_set = None
    
    # Initial random restarts to find a good starting point
    # The space is large, random starts might be bad, but SA can fix it.
    # Let's try a few initial configurations.
    
    # Option 1: Consecutive integers 1..32 (Score 15)
    # Option 2: Random
    
    # We will run SA for the remaining time.
    # To ensure we beat 1.0, we should start with a set that has score >= 15.
    # S_init = set(range(1, 33))
    # score = 15.
    
    # Let's use a mix. Start with range(1,33) and improve.
    # Also try random starts.
    
    def run_sa(initial_set, time_limit):
        current_set = set(initial_set)
        current_score = _score_set(current_set)
        best_local_score = current_score
        best_local_set = set(current_set)
        
        # Simulated Annealing parameters
        T = 10.0
        cooling = 0.9995
        min_T = 0.1
        
        while time.time() < time_limit and T > min_T:
            # Generate a move: swap one element
            # Pick x from current_set
            x = random.choice(list(current_set))
            # Pick y from 1..512 not in current_set
            # To avoid infinite loop if set is full (it's not, 32/512), 
            # we can just pick random y and check.
            
            # Efficient way to pick y not in S:
            # Since 512 is small, we can pick random and retry, or use a precomputed list of all candidates.
            # Precomputing list of all candidates is better for speed.
            
            # Let's restructure to have a pool of candidates.
            # But for now, simple retry.
            while True:
                y = random.randint(1, 512)
                if y not in current_set:
                    break
            
            new_set = set(current_set)
            new_set.remove(x)
            new_set.add(y)
            
            new_score = _score_set(new_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_set = new_set
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = set(current_set)
            
            T *= cooling
        
        return best_local_set, best_local_score

    # We have ~110s.
    # Let's do multiple restarts if time permits, or one long SA.
    # One long SA from a good start is usually robust.
    # Start with range(1, 33).
    
    start_set = set(range(1, 33))
    s_set, s_score = run_sa(start_set, deadline - 5) # Leave 5s for final checks
    
    if s_score > best_score:
        best_score = s_score
        best_set = s_set
    
    # Try a random start if time remains
    if time.time() < deadline - 10:
        rand_set = set(random.sample(range(1, 513), 32))
        r_set, r_score = run_sa(rand_set, deadline - 5)
        if r_score > best_score:
            best_score = r_score
            best_set = r_set
            
    # If best is still low, maybe try a structured set?
    # But SA from 1..32 should find something better than 15 quickly.
    
    if best_set is None:
        return list(range(1, 33))
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
