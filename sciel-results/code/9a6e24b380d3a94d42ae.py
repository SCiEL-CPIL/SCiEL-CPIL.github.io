import random
import time

def _score_set(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in S.
    S is a set of integers.
    """
    # Max possible d: a=1, a+2d=512 => 2d=511 => d=255
    max_d = 255
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    
    count = 0
    for d in range(1, max_d + 1):
        # Check if there exists a such that a, a+d, a+2d are in S
        # This is equivalent to: (mask & (mask >> d) & (mask >> (2*d))) != 0
        # Note: shifting right by d moves bit at pos i to i-d.
        # If bit i, i+d, i+2d are set in mask, then:
        # mask has bit i.
        # mask >> d has bit i-d (which was i in mask? No.)
        # Let's trace:
        # Bit k in (mask >> d) corresponds to bit k+d in mask.
        # Bit k in (mask >> 2d) corresponds to bit k+2d in mask.
        # We want indices a, a+d, a+2d.
        # Let a be the index in the shifted version.
        # If we shift mask right by d, bit a is 1 if mask had bit a+d.
        # If we shift mask right by 2d, bit a is 1 if mask had bit a+2d.
        # So (mask & (mask >> d) & (mask >> 2d)) has bit a set if mask has a, a+d, a+2d.
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    n = 32
    max_val = 512
    
    best_score = -1
    best_set = None
    
    # Helper to optimize a set via hill climbing
    def optimize(S):
        current_score = _score_set(S)
        S = set(S)
        while time.time() < deadline:
            # Pick random element to remove
            x = random.choice(list(S))
            # Pick random element to add
            # To ensure distinctness and range
            y = random.randint(1, max_val)
            if y in S:
                continue
            
            new_S = S.copy()
            new_S.remove(x)
            new_S.add(y)
            
            new_score = _score_set(new_S)
            
            if new_score >= current_score:
                S = new_S
                current_score = new_score
            # Optionally: allow some worse moves for simulated annealing?
            # Given the time constraint and discrete nature, strict hill climbing
            # with multiple restarts is usually robust.
            
            # If we get stuck, we might break out or just continue.
            # To avoid infinite loops on plateaus, we could count consecutive
            # non-improving moves and break if too many.
            # But random restarts handle this globally.
            
            # Add a small probability to accept equal score to move around plateau?
            # if new_score == current_score and random.random() < 0.1:
            #     S = new_S
            #     current_score = new_score
            
        return S, current_score

    # Multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Generate random initial set
        S = set(random.sample(range(1, max_val + 1), n))
        
        # Optimize
        opt_S, score = optimize(S)
        
        if score > best_score:
            best_score = score
            best_set = opt_S
            # print(f"New best: {score}")
        
        restarts += 1
        # If we've done many restarts, we might want to increase complexity
        # or just continue.
        
        # Check if we've hit a theoretical max?
        # Max distinct diffs is 255. Unlikely to reach.
        # Current best known in similar problems is much lower.
        
        # Stop if we have very little time left
        if time.time() > deadline - 5.0:
            break

    if best_set is None:
        # Fallback to seed
        return list(range(1, 33))
        
    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    n = 32
    max_val = 512
    
    best_score = -1
    best_set = None
    
    def _score_set(S):
        max_d = 255
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        count = 0
        for d in range(1, max_d + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def optimize(S):
        current_score = _score_set(S)
        S = set(S)
        no_improve_count = 0
        max_no_improve = 5000  # Break if stuck
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
                
            # Pick random element to remove
            x = random.choice(list(S))
            # Pick random element to add
            y = random.randint(1, max_val)
            if y in S:
                continue
            
            new_S = S.copy()
            new_S.remove(x)
            new_S.add(y)
            
            new_score = _score_set(new_S)
            
            if new_score > current_score:
                S = new_S
                current_score = new_score
                no_improve_count = 0
            elif new_score == current_score:
                # Accept equal score with some probability to explore plateau
                if random.random() < 0.2:
                    S = new_S
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        return S, current_score

    # Multiple restarts
    while time.time() < deadline:
        # Generate random initial set
        S = set(random.sample(range(1, max_val + 1), n))
        
        # Optimize
        opt_S, score = optimize(S)
        
        if score > best_score:
            best_score = score
            best_set = opt_S
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    if best_set is None:
        return list(range(1, 33))
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
