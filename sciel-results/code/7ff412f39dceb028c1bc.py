import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    
    # Precomputation
    N = 511
    factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not factors[i]: # i is prime
            for j in range(i, N + 1, i):
                factors[j].add(i)
    
    # We want to maximize sum 1/(512-a).
    # Sort candidates a from 511 down to 1.
    candidates = list(range(N, 0, -1))
    weights = {a: 1.0 / (512 - a) for a in candidates}
    
    # Beam Search
    # State: (score, factors_set, chosen_tuple)
    # We use a list of states.
    beam_width = 2000
    
    # Initial state: score 0, empty factors, empty chosen
    states = [(0.0, frozenset(), ())]
    
    best_score = 0.0
    best_chosen = ()
    
    for a in candidates:
        new_states = []
        w_a = weights[a]
        fac_a = factors[a]
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
            
        for score, fac_set, chosen in states:
            # Option 1: Exclude a
            new_states.append((score, fac_set, chosen))
            
            # Option 2: Include a
            # Check if disjoint
            if not (fac_a & fac_set):
                new_score = score + w_a
                new_fac = frozenset(fac_set | fac_a)
                new_chosen = chosen + (a,)
                new_states.append((new_score, new_fac, new_chosen))
                
                if new_score > best_score:
                    best_score = new_score
                    best_chosen = new_chosen
        
        # Prune: keep top beam_width states by score
        # Sorting 4000 items is fast.
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]
        
        # Optional: If the best score in the beam is less than best_score, 
        # and we are deep enough, we might prune more, but simple top-K is fine.
        
    # If we broke out of the loop early due to deadline, we still have best_chosen
    # If we finished, best_chosen is the max found in the beam search
    
    return list(best_chosen)
