import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    # Leave 5 seconds for overhead/exit
    deadline = start_time + 55.0
    
    # Generate Thue-Morse sequence
    # t[0] = 1
    # t[2k] = t[k]
    # t[2k+1] = -t[k]
    s = [1] * n
    for i in range(1, n):
        # Find highest power of 2 dividing i?
        # Or just use the recurrence.
        # Recurrence:
        # if i is even, i = 2k -> s[i] = s[k]
        # if i is odd, i = 2k+1 -> s[i] = -s[k]
        # This requires s[k] to be computed.
        # Since k < i, this works iteratively.
        pass # The simple loop above doesn't work directly because s[k] might not be the "parent" in a simple 1D fill if not careful.
        # Actually, s[i] depends on s[i//2].
        # Let's just compute it properly.
    
    # Correct Thue-Morse generation
    s = [1] * n
    for i in range(1, n):
        if i % 2 == 0:
            s[i] = s[i // 2]
        else:
            s[i] = -s[i // 2]
            
    def evaluate(seq):
        max_disc = 0
        for d in range(1, n // 2 + 1):
            for a in range(n):
                curr = 0
                idx = a
                while idx < n:
                    curr += seq[idx]
                    if curr > max_disc:
                        max_disc = curr
                    elif -curr > max_disc:
                        max_disc = -curr
                    idx += d
        return max_disc

    current_s = s[:]
    current_disc = evaluate(current_s)
    best_s = current_s[:]
    best_disc = current_disc
    
    rng = random.Random(123)
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        new_s = current_s[:]
        new_s[i] *= -1
        
        new_disc = evaluate(new_s)
        
        delta = new_disc - current_disc
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_s = new_s
            current_disc = new_disc
            
            if current_disc < best_disc:
                best_s = current_s[:]
                best_disc = current_disc
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart logic: if stuck for a while, maybe reset to best?
            # For now, just keep cooling.
            
    return best_s
