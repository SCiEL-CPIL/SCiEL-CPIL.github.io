import time
import sys

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin
    
    n = 12
    # Initial best: powers of 2
    best_seq = [1 << i for i in range(n)]
    best_max = best_seq[-1]
    
    def dfs(idx, current_seq, bits, current_max):
        nonlocal best_seq, best_max
        
        # Time check
        if time.time() - start_time > time_limit:
            return False
        
        if idx == n:
            if current_max < best_max:
                best_max = current_max
                best_seq = current_seq[:]
            return True
        
        # Pruning
        if current_max + (n - idx) >= best_max:
            return False
        
        start_x = current_seq[-1] + 1 if current_seq else 1
        end_x = best_max - 1
        
        max_sum = bits.bit_length() - 1
        if start_x > max_sum:
            x = start_x
            new_bits = bits | (bits << x)
            if dfs(idx + 1, current_seq + [x], new_bits, x):
                return True
            return False
        
        for x in range(start_x, end_x + 1):
            if time.time() - start_time > time_limit:
                return False
            
            if bits & (bits << x):
                continue
            
            new_bits = bits | (bits << x)
            
            if dfs(idx + 1, current_seq + [x], new_bits, x):
                return True
        
        return False

    initial_bits = 1
    initial_seq = []
    
    dfs(0, initial_seq, initial_bits, 0)
    
    return best_seq

