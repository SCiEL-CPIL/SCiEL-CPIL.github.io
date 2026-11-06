import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s
    
    n_max = 256
    
    # Precompute conflict masks
    # conflict_mask[i] is an integer bitmask where bit j is 1 if i and j conflict
    conflict_mask = [0] * (n_max + 1)
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            if (a * b) % (a + b) == 0:
                # Set bit b in mask a, and bit a in mask b
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # Helper to check if adding x to mask is valid
    def can_add(mask, x):
        return (mask & conflict_mask[x]) == 0
    
    # Helper to get size of mask
    def popcount(x):
        return bin(x).count("1")
        
    # Initial solution: all odds
    base_mask = 0
    for i in range(1, n_max + 1, 2):
        base_mask |= (1 << i)
        
    best_mask = base_mask
    best_size = popcount(best_mask)
    
    # Greedy improvement of initial solution
    current_mask = base_mask
    # Try adding evens
    evens = [i for i in range(2, n_max + 1, 2)]
    random.seed(42)
    
    # Multiple greedy passes with different orderings to find a strong starting point
    for _ in range(50):
        cur = base_mask
        # Shuffle evens
        shuffled_evens = evens[:]
        random.shuffle(shuffled_evens)
        for e in shuffled_evens:
            if can_add(cur, e):
                cur |= (1 << e)
        if popcount(cur) > best_size:
            best_size = popcount(cur)
            best_mask = cur
            
    # Simulated Annealing
    # State: current_mask
    # Energy: -size (we want to maximize size, so minimize -size)
    
    current_mask = best_mask
    current_size = best_size
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, n_max + 1))
    
    steps = 0
    while time.time() < deadline - 2.0 and T > T_min:
        steps += 1
        
        # Choose a random move type
        # 1. Add a random element (if not present)
        # 2. Remove a random element (if present)
        # 3. Swap a random present element with a random absent element
        
        r = random.random()
        new_mask = current_mask
        delta_size = 0
        
        if r < 0.4:
            # Try to add
            # Pick random candidate not in mask
            # To speed up, we can pick from a small subset or just random 1..256
            x = random.randint(1, n_max)
            if not (current_mask & (1 << x)):
                if can_add(current_mask, x):
                    new_mask = current_mask | (1 << x)
                    delta_size = 1
                else:
                    # If it conflicts, maybe we can't just add it.
                    # In SA, if we propose an invalid state, we usually reject it immediately 
                    # or treat it as infinite energy. 
                    # However, a better move is "Add if valid, else do nothing (or swap)".
                    # Let's stick to valid moves only for simplicity, or allow "Add" to fail and result in no change.
                    # If no change, delta_size = 0, accept prob 1.
                    pass
        elif r < 0.7:
            # Try to remove
            # Pick random element in mask
            # Efficient way: pick random index, if not in mask, try again or just skip
            # For 256 elements, picking random 1..256 and checking is fine
            x = random.randint(1, n_max)
            if (current_mask & (1 << x)):
                new_mask = current_mask & ~(1 << x)
                delta_size = -1
        else:
            # Swap
            # Pick x in mask, y not in mask
            x = random.randint(1, n_max)
            if (current_mask & (1 << x)):
                y = random.randint(1, n_max)
                if not (current_mask & (1 << y)):
                    # Check if y can be added after removing x
                    # mask_without_x = current_mask & ~(1 << x)
                    # if can_add(mask_without_x, y):
                    if ( (current_mask & ~(1 << x)) & conflict_mask[y] ) == 0:
                        new_mask = (current_mask & ~(1 << x)) | (1 << y)
                        delta_size = 0 # Size unchanged
                    else:
                        # Invalid swap, no change
                        pass

        new_size = popcount(new_mask)
        
        # Calculate acceptance probability
        if new_size > current_size:
            # Always accept improvement
            current_mask = new_mask
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        elif new_size < current_size:
            # Accept with probability exp(delta_size / T)
            # delta_size is negative
            prob = math.exp(delta_size / T)
            if random.random() < prob:
                current_mask = new_mask
                current_size = new_size
        else:
            # Same size, always accept to explore
            current_mask = new_mask
            # current_size unchanged
            
        T *= cooling_rate
        
        # Periodic greedy cleanup: try to add any valid element to current_mask
        if steps % 100 == 0:
            for i in range(1, n_max + 1):
                if not (current_mask & (1 << i)):
                    if can_add(current_mask, i):
                        current_mask |= (1 << i)
            current_size = popcount(current_mask)
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask

    # Final greedy pass on best_mask
    final_mask = best_mask
    for i in range(1, n_max + 1):
        if not (final_mask & (1 << i)):
            if can_add(final_mask, i):
                final_mask |= (1 << i)
                
    # Convert mask to sorted list
    result = []
    for i in range(1, n_max + 1):
        if final_mask & (1 << i):
            result.append(i)
            
    return result
