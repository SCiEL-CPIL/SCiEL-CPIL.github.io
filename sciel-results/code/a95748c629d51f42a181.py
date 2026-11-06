import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precompute 4-APs
    # Indices 0 to N-1 correspond to 1 to N
    aps = []
    # d >= 1
    # a + 3d <= N
    # a <= N - 3d
    # a >= 1
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            # a, a+d, a+2d, a+3d are 1-indexed
            # convert to 0-indexed
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    num_aps = len(aps)
    
    # Map bit index to list of AP indices it participates in
    bit_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            bit_to_aps[pos].append(idx)
    
    # Function to count monochromatic APs
    def count_mono(x):
        count = 0
        for ap in aps:
            if x[ap[0]] == x[ap[1]] == x[ap[2]] == x[ap[3]]:
                count += 1
        return count

    # Function to calculate delta for flipping bit i
    def delta_flip(x, i):
        delta = 0
        old_val = x[i]
        new_val = 1 - old_val
        for ap_idx in bit_to_aps[i]:
            ap = aps[ap_idx]
            # Get the other 3 positions
            # We need to check if the other 3 are equal to old_val and new_val
            # Let's extract the values of the other 3 bits
            # To avoid creating lists, we can do manual checks or a small loop
            # Since max APs per bit is small, a simple loop is fine
            others = []
            for pos in ap:
                if pos != i:
                    others.append(x[pos])
            
            # Check if all others == old_val
            was_mono = True
            for b in others:
                if b != old_val:
                    was_mono = False
                    break
            
            # Check if all others == new_val
            is_mono = True
            for b in others:
                if b != new_val:
                    is_mono = False
                    break
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
            # else delta unchanged (0)
        return delta

    # Simulated Annealing
    def simulated_annealing(initial_x, time_limit):
        x = initial_x.copy()
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        # Cooling factor per step? Or per time?
        # Let's use a fixed number of steps or time-based.
        # We don't know how many steps we can do.
        # Let's iterate until time_limit.
        
        while time.time() < time_limit:
            # Pick random bit
            i = random.randint(0, N - 1)
            d = delta_flip(x, i)
            
            # Accept if improvement or with probability exp(-d/T)
            if d < 0:
                x[i] = 1 - x[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            else:
                # Accept with probability
                if random.random() < np.exp(-d / T):
                    x[i] = 1 - x[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
            
            # Cool down
            # Exponential cooling
            T *= 0.99999 # Adjust based on speed
            if T < T_min:
                T = T_min
        
        return best_x, best_cost

    # Initialize
    # Try Thue-Morse
    def thue_morse(n):
        # Generate first n bits of Thue-Morse
        # t(i) = popcount(i) % 2
        return [bin(i).count('1') % 2 for i in range(n)]
    
    tm_x = np.array(thue_morse(N))
    
    # Try Random
    rand_x = np.array([random.randint(0, 1) for _ in range(N)])
    
    # Run SA on both, split time
    # Or run SA on one, if time left, run on another?
    # Better: Run multiple short SA runs from different starts?
    # Or one long run.
    # Let's run SA on Thue-Morse and Random in parallel? No, single thread.
    # Let's run SA on Random first, as it's more likely to be in a good basin if landscape is random-ish.
    # Actually, Thue-Morse is structured.
    # Let's do 2 restarts.
    
    # Time split: 50% for first, 50% for second?
    # Or just one long run with random start.
    # Let's do:
    # 1. Run SA on Random start for 60s.
    # 2. Run SA on Thue-Morse start for remaining time.
    # Keep the best.
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run 1
    t1_limit = start_time + 55.0
    if time.time() < t1_limit:
        x1, c1 = simulated_annealing(rand_x, t1_limit)
        if c1 < best_overall_cost:
            best_overall_cost = c1
            best_overall_x = x1
    
    # Run 2
    t2_limit = deadline
    if time.time() < t2_limit:
        x2, c2 = simulated_annealing(tm_x, t2_limit)
        if c2 < best_overall_cost:
            best_overall_cost = c2
            best_overall_x = x2
    
    # Fallback
    if best_overall_x is None:
        best_overall_x = rand_x
    
    # Return as list of ints
    return [int(v) for v in best_overall_x]
