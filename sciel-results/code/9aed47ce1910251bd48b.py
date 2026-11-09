import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    MIN_SUM = 128
    MAX_SUM = 256
    N = 128
    
    A = set(range(N + 1))
    
    counts = [0] * (MAX_SUM + 1)
    
    def compute_counts():
        for s in range(MIN_SUM, MAX_SUM + 1):
            counts[s] = 0
        for a in A:
            for b in A:
                if a <= b:
                    s = a + b
                    if MIN_SUM <= s <= MAX_SUM:
                        counts[s] += 1

    def get_peak():
        return max(counts[MIN_SUM:MAX_SUM+1])
        
    def is_valid():
        return min(counts[MIN_SUM:MAX_SUM+1]) >= 1

    compute_counts()
    current_peak = get_peak()
    best_peak = current_peak
    best_A = set(A)
    
    # Helper to update counts incrementally
    def remove_element(x):
        if 2 * x >= MIN_SUM and 2 * x <= MAX_SUM:
            counts[2 * x] -= 1
        for y in A:
            if y != x:
                s = x + y
                if MIN_SUM <= s <= MAX_SUM:
                    counts[s] -= 1
        A.remove(x)
        
    def add_element(x):
        A.add(x)
        if 2 * x >= MIN_SUM and 2 * x <= MAX_SUM:
            counts[2 * x] += 1
        for y in A:
            if y != x:
                s = x + y
                if MIN_SUM <= s <= MAX_SUM:
                    counts[s] += 1

    # Phase 1: Greedy Peel
    # We want to remove elements as long as validity is maintained.
    # This reduces the set size and potentially the peak.
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        items = list(A)
        random.shuffle(items)
        for x in items:
            if time.time() > deadline - 30:
                break
            old_peak = get_peak()
            
            # Simulate removal
            remove_element(x)
            
            if is_valid():
                new_peak = get_peak()
                if new_peak <= old_peak:
                    # Accept
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                    current_peak = new_peak
                    improved = True
                    break # Restart loop with new A
                else:
                    # Rollback
                    add_element(x)
            else:
                # Rollback
                add_element(x)

    # Phase 2: Simulated Annealing
    # Try to further reduce peak by swapping or adding/removing.
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline - 5:
        if T < 0.01:
            break
            
        # Pick a random move
        move_type = random.randint(0, 2)
        old_peak = get_peak()
        
        if move_type == 0 and A:
            # Remove
            x = random.choice(list(A))
            remove_element(x)
            if is_valid():
                new_peak = get_peak()
                delta = new_peak - old_peak
                if delta < 0 or random.random() < math.exp(-delta / T):
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                    current_peak = new_peak
                else:
                    add_element(x)
            else:
                add_element(x)
                
        elif move_type == 1:
            # Add
            # Pick y not in A
            y = random.randint(0, N)
            if y not in A:
                add_element(y)
                new_peak = get_peak()
                delta = new_peak - old_peak
                # Adding usually increases peak or keeps it same. 
                # We only accept if it helps cover something? No, we are already valid.
                # Adding an element might allow us to remove another later.
                # But for now, if it increases peak, we probably shouldn't accept unless T is high.
                if delta < 0 or random.random() < math.exp(-delta / T):
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                    current_peak = new_peak
                else:
                    remove_element(y)
                    
        elif move_type == 2 and A:
            # Swap
            x = random.choice(list(A))
            y = random.randint(0, N)
            if y not in A:
                remove_element(x)
                add_element(y)
                if is_valid():
                    new_peak = get_peak()
                    delta = new_peak - old_peak
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        if new_peak < best_peak:
                            best_peak = new_peak
                            best_A = set(A)
                        current_peak = new_peak
                    else:
                        # Rollback
                        remove_element(y)
                        add_element(x)
                else:
                    # Rollback
                    remove_element(y)
                    add_element(x)
                    
        T *= cooling
        
    return list(best_A)
