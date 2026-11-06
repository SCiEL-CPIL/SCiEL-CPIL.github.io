"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

def _count_progression_diffs(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S."""
    S_set = set(S)
    elems = sorted(S)
    n = len(elems)
    valid_ds = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = elems[j] - elems[i]
            if elems[j] + d in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def construct():
    """Return exactly 32 distinct integers in 1..512 maximizing progression diffs."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate a strong initial candidate
    # Dense sets tend to have many small differences.
    # A set like {1..32} has many progressions.
    # Let's start with a mix of dense and structured.
    
    best_set = list(range(1, 33))
    best_score = _count_progression_diffs(best_set)
    
    # Try a few other initial structures
    alt_candidates = [
        list(range(1, 513, 16))[:32], # Sparse AP
        list(range(1, 33)) + list(range(33, 65))[:32-32], # Just 1..32
        random.sample(range(1, 513), 32),
    ]
    
    for c in alt_candidates:
        # Ensure valid set
        c = list(set(c))
        while len(c) < 32:
            c.append(random.randint(1, 512))
        c = c[:32]
        score = _count_progression_diffs(c)
        if score > best_score:
            best_score = score
            best_set = c
    
    # Local Search / Hill Climbing
    S = list(best_set)
    S_set = set(S)
    current_score = best_score
    
    # Use a simple hill climbing with random restarts if stuck
    # Or just a long hill climb.
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        i = random.randint(0, 31)
        old_val = S[i]
        new_val = random.randint(1, 512)
        
        if new_val in S_set:
            continue
            
        S[i] = new_val
        S_set.discard(old_val)
        S_set.add(new_val)
        
        new_score = _count_progression_diffs(S)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_steps = 0
        else:
            # Revert
            S[i] = old_val
            S_set.discard(new_val)
            S_set.add(old_val)
            no_improve_steps += 1
        
        # Occasionally, if stuck, try a more aggressive move or restart
        if no_improve_steps > 1000:
             # Try swapping two elements or a random jump
             # For simplicity, just keep going, the random nature will eventually find a way out
             # Or force a reset to a random state if very stuck
             if no_improve_steps > 5000:
                 S = random.sample(range(1, 513), 32)
                 S_set = set(S)
                 current_score = _count_progression_diffs(S)
                 no_improve_steps = 0
                 if current_score > best_score:
                     best_score = current_score
                     best_set = list(S)
    
    # Update best if we improved
    if current_score > best_score:
        best_score = current_score
        best_set = list(S)
    
    return best_set

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
