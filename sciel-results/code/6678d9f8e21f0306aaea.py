"""Deterministic greedy construction with bitset-accelerated scoring."""
import time
import random

N = 32
MAX_VAL = 512

def _score_mask(mask):
    """Count distinct d>0 such that exists a with a, a+d, a+2d in set represented by mask."""
    s = set()
    # iterate over all pairs to find potential 3-term APs
    # For efficiency, iterate over d and check
    # max d is 256 (since 1 + 2*256 = 513 > 512, actually max a=1, a+2d<=512 => 2d<=511 => d<=255)
    # Actually a >= 1, a+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        found = False
        # Check if there exists a such that a, a+d, a+2d are in mask
        # We can check by iterating a from 1 to 512-2d
        # But faster: for each a in mask, check if a+d and a+2d in mask
        # Or: for each pair (x,y) in mask with y-x=d, check if y+d in mask
        # Let's iterate a from 1 to 512-2d
        # Using bit operations: 
        # mask & (mask >> d) & (mask >> 2d) != 0
        # But we need to be careful with shifts and bounds.
        # Let's use the set of elements for clarity and speed in Python for N=32
        pass
    
    # Better approach for small N:
    elems = [i for i in range(1, 513) if (mask >> i) & 1]
    if len(elems) < 3:
        return 0
    s = set()
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            d = elems[j] - elems[i]
            if d <= 0:
                continue
            # check if elems[i] + 2*d is in set
            # Since elems is sorted, we can use binary search or just check
            # But we can also just check if (elems[i] + 2*d) is in the set of elems
            # To speed up, we can precompute a set of elems
            if (elems[i] + 2*d) in elems:
                s.add(d)
    return len(s)

# Even faster: use a set for membership
def _score_set(elems_set):
    s = set()
    elems_list = sorted(elems_set)
    for i in range(len(elems_list)):
        a = elems_list[i]
        for j in range(i+1, len(elems_list)):
            d = elems_list[j] - a
            if a + 2*d in elems_set:
                s.add(d)
    return len(s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Greedy construction
    # Start with a random or structured initial set
    # To make it deterministic but effective, let's try a few random starts and pick the best
    # Or just one good heuristic start.
    
    best_score = 0
    best_set = set()
    
    # Try a few random initializations
    for trial in range(50):
        if time.time() > deadline:
            break
            
        # Initialize with N random distinct integers
        current_set = set(random.sample(range(1, 513), N))
        
        # Greedy improvement:
        # While we can improve, find the best swap or add/remove
        # Since size is fixed, we do swaps.
        # To speed up, we can try to add elements that create new APs and remove those that don't.
        # But simpler: local search with swaps.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = _score_set(current_set)
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                
            # Try all pairs (x in current_set, y not in current_set)
            # This is 32 * 480 ~ 15k swaps per iteration.
            # _score_set is O(N^2) ~ 1000 ops.
            # 15k * 1000 = 15M ops per iteration. Might be slow in Python.
            # Let's limit the number of swaps tried per iteration or use a faster score.
            
            # Optimization: only try a subset of swaps?
            # Or use a faster score function.
            
            # Let's try to optimize _score_set
            # Precompute pairs?
            # Or just accept the speed and run for a while.
            
            # To make it faster, let's only try random swaps in each iteration
            # instead of all swaps.
            for _ in range(1000):  # 1000 random swaps per iteration
                x = random.choice(list(current_set))
                y = random.randint(1, 512)
                if y in current_set:
                    continue
                    
                # Try swap
                old_x = x
                current_set.remove(x)
                current_set.add(y)
                
                new_score = _score_set(current_set)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    current_set.remove(y)
                    current_set.add(old_x)
                    
                if time.time() > deadline:
                    break
            
            # If no improvement after many trials, break
            if not improved:
                # Try a few more random perturbations (simulated annealing style)
                for _ in range(100):
                    x = random.choice(list(current_set))
                    y = random.randint(1, 512)
                    if y in current_set:
                        continue
                    current_set.remove(x)
                    current_set.add(y)
                    new_score = _score_set(current_set)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_set.remove(y)
                        current_set.add(x)
                if not improved:
                    break
                    
    # Final check
    final_score = _score_set(best_set)
    if final_score > best_score:
        best_score = final_score
        
    return sorted(best_set)

# To ensure we always return something valid even if the loop fails
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Fallback: simple range
    result = list(range(1, 33))
    best_score = _score_set(set(result))
    best_set = set(result)
    
    # Greedy improvement
    for trial in range(20):
        if time.time() > deadline:
            break
            
        # Start from a random set or the best found so far
        current_set = set(random.sample(range(1, 513), 32))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = _score_set(current_set)
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                
            # Try random swaps
            for _ in range(500):
                x = random.choice(list(current_set))
                y = random.randint(1, 512)
                if y in current_set:
                    continue
                    
                old_x = x
                current_set.remove(x)
                current_set.add(y)
                
                new_score = _score_set(current_set)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_set.remove(y)
                    current_set.add(old_x)
                    
                if time.time() > deadline:
                    break
            
            if not improved:
                break
                
    return sorted(best_set)

# The actual construct function to be used
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with a good heuristic set
    # For example, a set with many small differences
    # Or just random
    
    best_set = set(range(1, 33))
    best_score = _score_set(best_set)
    
    # Try multiple random restarts
    for trial in range(100):
        if time.time() > deadline:
            break
            
        current_set = set(random.sample(range(1, 513), 32))
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = _score_set(current_set)
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                
            # Try random swaps
            for _ in range(1000):
                x = random.choice(list(current_set))
                y = random.randint(1, 512)
                if y in current_set:
                    continue
                    
                old_x = x
                current_set.remove(x)
                current_set.add(y)
                
                new_score = _score_set(current_set)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_set.remove(y)
                    current_set.add(old_x)
                    
                if time.time() > deadline:
                    break
            
            if not improved:
                # Try a few more random perturbations
                for _ in range(50):
                    x = random.choice(list(current_set))
                    y = random.randint(1, 512)
                    if y in current_set:
                        continue
                    current_set.remove(x)
                    current_set.add(y)
                    new_score = _score_set(current_set)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_set.remove(y)
                        current_set.add(x)
                if not improved:
                    break
                    
    return sorted(best_set)
