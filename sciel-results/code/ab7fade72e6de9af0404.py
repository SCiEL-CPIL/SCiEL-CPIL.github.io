import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Bitset representation
    # x in 1..512 maps to bit x-1
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def get_bitset(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b

    def score_bitset(b):
        count = 0
        # d can be 1..255
        # If d is too large, S >> 2d will be 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> 2 * d)):
                count += 1
        return count

    def random_set():
        return random.sample(range(1, 513), 32)
    
    # Initialize with multiple random restarts
    best_set = None
    best_bit = None
    best_score = -1
    
    num_restarts = 5
    restart_time = (deadline - start_time) / (num_restarts + 1)  # Leave time for final polish
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 10:  # Stop if close to deadline
            break
            
        current_set = random_set()
        current_bit = get_bitset(current_set)
        current_score = score_bitset(current_bit)
        
        # Simulated Annealing
        T = 15.0
        T_min = 0.05
        cooling_rate = 0.99998
        stall_count = 0
        stall_threshold = 2000
        
        while time.time() < deadline - 10 and T > T_min:
            # Generate neighbor: swap one element
            idx = random.randint(0, 31)
            old_val = current_set[idx]
            new_val = random.randint(1, 512)
            
            if new_val == old_val:
                continue
            if new_val in current_set:
                continue
            
            # Update bitset
            new_bit = current_bit
            new_bit &= ~(1 << (old_val - 1))
            new_bit |= (1 << (new_val - 1))
            
            new_score = score_bitset(new_bit)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                # Accept
                current_set[idx] = new_val
                current_bit = new_bit
                current_score = new_score
                stall_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    best_bit = current_bit
                    
            else:
                stall_count += 1
                
            # Increase cooling when stalled
            if stall_count > stall_threshold:
                T *= 0.95
                stall_count = 0
                
            T *= cooling_rate
        
        # Final polish: deterministic local search
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(32):
                    if time.time() >= deadline - 5:
                        break
                    old_val = current_set[i]
                    best_new_val = None
                    best_new_score = current_score
                    for new_val in range(1, 513):
                        if new_val == old_val or new_val in current_set:
                            continue
                        new_bit = current_bit
                        new_bit &= ~(1 << (old_val - 1))
                        new_bit |= (1 << (new_val - 1))
                        new_score = score_bitset(new_bit)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_val = new_val
                    if best_new_val is not None:
                        current_set[i] = best_new_val
                        current_bit = get_bitset(current_set)
                        current_score = best_new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                            best_bit = current_bit
    
    if best_set is None:
        best_set = list(range(1, 33))
        
    return best_set
