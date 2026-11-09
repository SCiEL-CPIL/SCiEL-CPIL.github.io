import random
import time

def calc_disc(seq):
    n = 256
    max_disc = 0
    for d in range(1, n):
        for r in range(1, d + 1):
            # Efficient extraction
            # Using list comprehension is okay for N=256
            sub = [seq[i-1] for i in range(r, n+1, d)]
            if not sub:
                continue
            
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            
            for val in sub:
                max_ending_here = max(0, max_ending_here + val)
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                
                min_ending_here = min(0, min_ending_here + val)
                if min_ending_here < min_so_far:
                    min_so_far = min_ending_here
            
            disc = max(max_so_far, -min_so_far)
            if disc > max_disc:
                max_disc = disc
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_seq = None
    best_cost = float('inf')
    
    n = 256
    
    # Multiple restarts
    while time.time() < deadline:
        # Random start
        seq = [random.choice((-1, 1)) for _ in range(n)]
        current_cost = calc_disc(seq)
        
        # Hill Climbing
        no_improve_count = 0
        while time.time() < deadline:
            # Try to improve
            improved = False
            # Try a random subset of flips to save time? 
            # Or just one random flip?
            # One random flip is O(1) setup + O(N^2) eval.
            # If we do 1000 steps, that's 1000 evals.
            
            idx = random.randint(0, n - 1)
            new_seq = seq[:]
            new_seq[idx] *= -1
            new_cost = calc_disc(new_seq)
            
            if new_cost < current_cost:
                seq = new_seq
                current_cost = new_cost
                no_improve_count = 0
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = seq[:]
            else:
                no_improve_count += 1
                # If we stagnate, break and restart
                if no_improve_count > 50:
                    break
        
        # If we hit time limit in inner loop, break outer
        if time.time() >= deadline:
            break

    return best_seq
