import random
import time

def _check_and_add(x, chosen, counts):
    """
    Check if x can be added to chosen without violating B2[2] condition.
    Returns True if added, False otherwise.
    Modifies chosen and counts if added.
    """
    # We need to check sums x + y for all y in chosen, and 2*x.
    # The condition is that for any sum s, the number of pairs (a,b) with a<=b, a+b=s is <= 2.
    # Currently, counts[s] holds the number of such pairs.
    # If we add x, we add pairs (x, y) for y in chosen where y <= x, and (y, x) for y in chosen where y < x?
    # Wait, the pair is unordered {a,b} with a<=b.
    # If we add x, new pairs are {x, y} for all y in chosen.
    # If y == x, pair is {x,x}, sum 2x.
    # If y < x, pair is {y,x}, sum x+y.
    # If y > x, pair is {x,y}, sum x+y.
    # So for each y in chosen, the sum s = x + y gets one new representation.
    # We must ensure that for all such s, the new count does not exceed 2.
    # i.e., current count of s must be < 2.
    
    # Optimization: If any y in chosen makes x+y have count >= 2, we cannot add x.
    # We can early exit.
    
    for y in chosen:
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
        
    # Check 2*x
    s2 = 2 * x
    if counts.get(s2, 0) >= 2:
        return False
        
    # If we reach here, x can be added.
    # Update counts.
    for y in chosen:
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    counts[s2] = counts.get(s2, 0) + 1
    
    chosen.append(x)
    return True

def _greedy(candidates):
    chosen = []
    counts = {}
    for x in candidates:
        _check_and_add(x, chosen, counts)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Baseline: sorted ascending
    baseline = _greedy(list(range(1, 1001)))
    best_set = baseline
    best_size = len(best_set)
    
    # Randomized search
    candidates_base = list(range(1, 1001))
    random.seed(42) # For reproducibility in evaluation if needed, though random is fine
    
    while time.time() < deadline:
        # Try different strategies
        # 1. Random shuffle
        candidates = candidates_base[:]
        random.shuffle(candidates)
        
        current_set = _greedy(candidates)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # 2. Reverse sorted (largest first)
        # This might be worth trying explicitly or as part of the shuffle distribution
        # But shuffle covers it probabilistically.
        
        # 3. Maybe try to bias towards larger numbers?
        # We can create a weighted shuffle or just stick to uniform shuffle.
        # Uniform shuffle is a good general heuristic.
        
        # To speed up, we can break early if we find a very large set?
        # Theoretical max is ~90. If we hit 90, we might stop or keep looking for 91.
        # But we don't know the exact max.
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_set
