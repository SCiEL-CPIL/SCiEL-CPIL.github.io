import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    n_max = 256

    # Precompute conflict masks: bit j set in mask[i] means i and j conflict
    conflict_mask = [0] * (n_max + 1)
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    def popcount(x):
        return x.bit_count()

    def can_add(mask, x):
        return (mask & conflict_mask[x]) == 0

    # Greedy completion: add all possible elements
    def greedy_complete(mask):
        # Iterate through all elements, if not in mask and compatible, add it
        # This is O(N^2) in worst case per call, but N=256 is small.
        # We can optimize by checking conflicts only for elements not yet in mask.
        for i in range(1, n_max + 1):
            bit = 1 << i
            if not (mask & bit):
                if (mask & conflict_mask[i]) == 0:
                    mask |= bit
        return mask

    # Multi-start greedy to find good initial solution
    all_elements = list(range(1, n_max + 1))
    evens = list(range(2, n_max + 1, 2))
    odds = list(range(1, n_max + 1, 2))

    random.seed(42)
    best_mask = 0
    best_size = 0

    # Start from all odds (known good base)
    base_mask = 0
    for i in odds:
        base_mask |= (1 << i)
    best_mask = greedy_complete(base_mask)
    best_size = popcount(best_mask)

    # Try multiple random greedy orderings
    for _ in range(50): # Reduced from 120 to save time
        if time.time() > deadline - 10.0: break
        cur = 0
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        for e in shuffled:
            if can_add(cur, e):
                cur |= (1 << e)
        cur = greedy_complete(cur)
        sz = popcount(cur)
        if sz > best_size:
            best_size = sz
            best_mask = cur

    # Also try starting from random subsets of various sizes
    for trial_size in [64, 128]: # Reduced trials
        for _ in range(10):
            if time.time() > deadline - 10.0: break
            cur = 0
            sample = random.sample(all_elements, min(trial_size, n_max))
            for e in sample:
                if can_add(cur, e):
                    cur |= (1 << e)
            cur = greedy_complete(cur)
            sz = popcount(cur)
            if sz > best_size:
                best_size = sz
                best_mask = cur

    # Simulated Annealing with optimized move generation
    current_mask = best_mask
    current_size = best_size

    T = 5.0
    T_min = 0.001
    cooling_rate = 0.999 # Slower cooling to explore more
    steps = 0

    # Precompute lists for faster random selection if needed, but randint is fine
    while time.time() < deadline - 5.0 and T > T_min:
        steps += 1
        r = random.random()
        new_mask = current_mask
        delta = 0

        if r < 0.30:
            # Add a random element
            x = random.randint(1, n_max)
            bit_x = 1 << x
            if not (current_mask & bit_x):
                if (current_mask & conflict_mask[x]) == 0:
                    new_mask = current_mask | bit_x
                    delta = 1
        elif r < 0.50:
            # Remove a random element
            x = random.randint(1, n_max)
            bit_x = 1 << x
            if current_mask & bit_x:
                new_mask = current_mask & ~bit_x
                delta = -1
        else:
            # Swap: remove x, add y
            x = random.randint(1, n_max)
            bit_x = 1 << x
            if current_mask & bit_x:
                y = random.randint(1, n_max)
                bit_y = 1 << y
                if not (current_mask & bit_y):
                    mask_without = current_mask & ~bit_x
                    if (mask_without & conflict_mask[y]) == 0:
                        new_mask = mask_without | bit_y
                        delta = 0

        new_size = new_mask.bit_count()

        if new_size > current_size:
            current_mask = new_mask
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        elif new_size < current_size:
            # Accept worse moves with probability exp(delta/T)
            # delta is -1 here
            if random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_size = new_size
        else:
            # Same size: accept to explore
            current_mask = new_mask

        T *= cooling_rate

        # Periodic greedy completion
        if steps % 50 == 0:
            completed = greedy_complete(current_mask)
            csz = completed.bit_count()
            if csz > current_size:
                current_mask = completed
                current_size = csz
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

    # Final local search: simple greedy polish
    # The SA should have found a good local optimum. 
    # A heavy local search here might exceed time. 
    # Just do a final greedy completion which is fast.
    final_mask = greedy_complete(best_mask)
    
    # Convert to sorted list
    result = []
    for i in range(1, n_max + 1):
        if final_mask & (1 << i):
            result.append(i)

    return result
