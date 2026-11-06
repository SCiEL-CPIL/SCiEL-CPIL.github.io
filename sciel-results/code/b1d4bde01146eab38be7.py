import random
import time

def construct():
    N = 512
    elements = list(range(1, N + 1))
    
    # Pre-compute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    aps = []
    # Map from element to list of AP indices containing it
    elem_to_aps = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for x in ap:
                elem_to_aps[x].append(idx)

    # Precompute for each element, the other 3 elements in each AP it belongs to
    # elem_other_3[x] = list of tuples (y1, y2, y3) for each AP containing x
    elem_other_3 = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(aps):
        a, b, c, d = ap
        for x in ap:
            others = tuple(y for y in ap if y != x)
            elem_other_3[x].append(others)

    # Function to check if a set (as a boolean list) is valid for adding x
    def is_valid_addition(x, is_in):
        """Check if adding x to the current set creates a 4-AP."""
        for others in elem_other_3[x]:
            if is_in[others[0]] and is_in[others[1]] and is_in[others[2]]:
                return False
        return True

    def greedy_fill(is_in, order):
        """Add elements from 'order' to is_in if valid."""
        for x in order:
            if not is_in[x]:
                if is_valid_addition(x, is_in):
                    is_in[x] = True

    best_size = 0
    best_set = [False] * (N + 1)
    
    start_time = time.time()
    # Use 55s limit to be safe (evaluator kills at 90s, need margin for overhead)
    time_limit = 55.0
    
    def time_exceeded():
        return (time.time() - start_time) > time_limit
    
    # Initial solution: Random Greedy
    # We will run multiple iterations of ILS
    
    # First, find a good starting point
    for i in range(50):
        if time_exceeded():
            break
        is_in = [False] * (N + 1)
        order = elements[:]
        random.shuffle(order)
        greedy_fill(is_in, order)
        current_size = sum(is_in)
        if current_size > best_size:
            best_size = current_size
            best_set = is_in[:]

    # ILS Loop
    # Perturbation: remove k elements
    # Repair: greedy fill
    
    iteration = 0
    while not time_exceeded():
        # Check time more frequently
        if time.time() - start_time > time_limit - 2:
            break
        
        # Start from current best
        is_in = best_set[:]
        current_size = best_size
        
        # Collect indices in set
        in_set = [i for i in range(1, N + 1) if is_in[i]]
        
        # Perturb: remove k random elements that are in the set
        k = random.randint(2, 8)
        
        if len(in_set) < k:
            k = max(1, len(in_set) // 2)
            if k == 0:
                continue
            
        to_remove = random.sample(in_set, k)
        for x in to_remove:
            is_in[x] = False
            current_size -= 1
            
        # Repair: Greedy fill with random order
        order = elements[:]
        random.shuffle(order)
        
        greedy_fill(is_in, order)
        
        new_size = sum(is_in)
        
        if new_size > best_size:
            best_size = new_size
            best_set = is_in[:]
        
        iteration += 1
        # Safety check every 100 iterations
        if iteration % 100 == 0 and time.time() - start_time > time_limit - 5:
            break
            
    # Convert best_set to sorted list
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
