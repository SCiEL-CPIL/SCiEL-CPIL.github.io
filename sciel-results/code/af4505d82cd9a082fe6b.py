import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 32
    MAX_VAL = 512
    
    def get_score(B):
        count = 0
        for d in range(1, 256):
            if (B & (B >> d) & (B >> (2 * d))) != 0:
                count += 1
        return count

    # Initialize
    current_set = set(random.sample(range(1, MAX_VAL + 1), N))
    current_list = list(current_set)
    current_B = 0
    for x in current_set:
        current_B |= (1 << x)
    
    current_score = get_score(current_B)
    
    best_set = set(current_set)
    best_score = current_score
    
    T = 1.0
    cooling_rate = 0.99999
    
    while time.time() < deadline:
        # Pick random element to change
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]
        
        # Pick new value
        new_val = random.randint(1, MAX_VAL)
        while new_val in current_set:
            new_val = random.randint(1, MAX_VAL)
            
        # Update Bitmask
        new_B = current_B ^ (1 << old_val) ^ (1 << new_val)
        
        new_score = get_score(new_B)
        
        # Acceptance
        if new_score > current_score or (random.random() < (new_score - current_score) / T):
            # Accept
            current_set.remove(old_val)
            current_set.add(new_val)
            current_list[idx] = new_val
            current_B = new_B
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_set = set(current_set)
        
        T *= cooling_rate
        if T < 0.001:
            T = 0.001

    return sorted(list(best_set))
