import time
import random
import math

def construct():
    n = 256
    deadline = time.time() + 55  # 5s margin
    
    # Thue-Morse initialization
    def thue_morse(n):
        seq = []
        for i in range(n):
            # Count bits in binary representation of i
            # Parity of count
            c = bin(i).count('1')
            if c % 2 == 0:
                seq.append(-1)
            else:
                seq.append(1)
        return seq

    seq = thue_morse(n)
    
    def get_cost(s):
        max_disc = 0
        nn = n
        # Optimization: local variable lookup
        for d in range(1, nn + 1):
            for r in range(d):
                curr = 0
                min_v = 0
                max_v = 0
                idx = r
                while idx < nn:
                    curr += s[idx]
                    if curr < min_v:
                        min_v = curr
                    if curr > max_v:
                        max_v = curr
                    idx += d
                disc = max_v - min_v
                if disc > max_disc:
                    max_disc = disc
        return max_disc

    best_seq = seq[:]
    best_cost = get_cost(seq)
    
    rng = random.Random(42)
    
    # Simulated Annealing Parameters
    # Start with high temp, decay
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    # We will run multiple restarts if time permits
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        # Check if we should restart
        # Simple heuristic: if T is low and we haven't improved in a while, restart
        # Or just run SA until T is low, then restart from best_seq with new T
        
        T = 10.0
        current_seq = best_seq[:]
        current_cost = best_cost
        
        no_improve = 0
        patience = 5000 # iterations without improvement
        
        while time.time() < deadline:
            idx = rng.randrange(n)
            current_seq[idx] *= -1
            
            new_cost = get_cost(current_seq)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = current_seq[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    # Revert
                    current_seq[idx] *= -1
                    no_improve += 1
            
            T *= cooling
            if T < min_T:
                break
            
            if no_improve > patience:
                break
        
        # If we broke due to time, exit outer loop
        if time.time() >= deadline:
            break
            
        # If we broke due to patience or min_T, restart
        restarts += 1
        if restarts > max_restarts:
            break
            
    return best_seq
