import numpy as np
import time
import random

def construct():
    n = 512
    start_time = time.time()
    max_time = 55.0  # Safety margin for 90s limit (leaves 35s)

    def is_valid_addition(x, S, n):
        # S is a boolean numpy array of length n+1
        # Check 4 cases for any d
        # Case 1: x is 4th term: x-3d, x-2d, x-d in S
        # d >= 1, x-3d >= 1 => d <= (x-1)//3
        for d in range(1, (x-1)//3 + 1):
            if S[x-d] and S[x-2*d] and S[x-3*d]:
                return False
        
        # Case 2: x is 1st term: x+d, x+2d, x+3d in S
        # x+3d <= n => d <= (n-x)//3
        for d in range(1, (n-x)//3 + 1):
            if S[x+d] and S[x+2*d] and S[x+3*d]:
                return False
        
        # Case 3: x is 2nd term: x-d, x+d, x+2d in S
        # x-d >= 1 => d <= x-1
        # x+2d <= n => d <= (n-x)//2
        max_d = min(x-1, (n-x)//2)
        for d in range(1, max_d + 1):
            if S[x-d] and S[x+d] and S[x+2*d]:
                return False
        
        # Case 4: x is 3rd term: x-2d, x-d, x+d in S
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= n => d <= n-x
        max_d = min((x-1)//2, n-x)
        for d in range(1, max_d + 1):
            if S[x-2*d] and S[x-d] and S[x+d]:
                return False
        
        return True

    # Initial deterministic greedy to get a good starting point
    S_best = np.zeros(n+1, dtype=bool)
    for x in range(1, n+1):
        if is_valid_addition(x, S_best, n):
            S_best[x] = True
    
    best_count = int(np.sum(S_best))
    best_set_list = [i for i in range(1, n+1) if S_best[i]]
    
    # Randomized Greedy with Local Search (Swaps)
    # We will run multiple restarts
    # To speed up, we can use a fixed seed for reproducibility if needed, but random is fine.
    
    # Optimization: Pre-compute valid d ranges? No, x changes.
    
    # We can parallelize? No, standard library only.
    # We can optimize the inner loop by using NumPy operations?
    # The loops are small (max ~170). Python loops are okay.
    
    # Let's try to run as many restarts as possible.
    # One restart takes ~512 * (avg checks).
    # Avg checks per x: ~50. 512 * 50 = 25k ops. Very fast.
    # We can do 1000 restarts easily in < 1s.
    # Let's do 5000 restarts to be sure.
    
    # Actually, let's add a "local improvement" step?
    # If we have a set, can we swap elements to increase size?
    # Randomized greedy is usually sufficient for these sizes.
    
    # Let's vary the random seed for each restart.
    
    for restart in range(5000):
        if time.time() - start_time > max_time:
            break
        
        S_curr = np.zeros(n+1, dtype=bool)
        # Random permutation
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, S_curr, n):
                S_curr[x] = True
        
        curr_count = int(np.sum(S_curr))
        
        # Local Search: Try to swap elements to increase size
        # If we remove y and add x, and size increases? No, size stays same.
        # We want to find a pair (x in S, y not in S) such that swapping x->y keeps validity?
        # No, that doesn't increase size.
        # We want to find x not in S that can be added, and y in S that can be removed, such that 
        # we can then add another element? This is complex.
        
        # Simpler Local Search: Try to add elements by removing conflicting ones?
        # Or just try random swaps to escape local minima?
        # Since greedy is already quite good, maybe just more restarts is better.
        
        # However, inspiration 1 improved score with "greedy_greedy_order".
        # Let's try a different ordering strategy: 
        # Start with random, then try to "fill gaps" by checking if removing one element allows adding two?
        # This is too slow.
        
        # Let's stick to more restarts but optimize the check.
        # The current check is O(n^2) effectively.
        
        # Let's try to improve the initial set with a simple hill climbing:
        # Try to add any missing element. If it conflicts, try removing one conflicting element and adding the new one?
        # If we remove x and add y, size is same.
        # We need to find a sequence of removals/additions that increases size.
        
        # Given the time limit, let's just run more restarts.
        # The previous program ran 5000 restarts. 
        # Let's see if we can do more by optimizing the inner loop.
        
        # Actually, the previous program's score was 1.0. 
        # Inspiration 1 had score 1.008. 
        # Let's look at the diff in inspiration 1.
        # It removed the local search idea and just did greedy? 
        # No, the diff shows it simplified the code.
        
        # Let's try to be smarter about the random shuffle.
        # Instead of pure random, maybe sort by some heuristic?
        # E.g., numbers with fewer divisors? Or numbers that are harder to place?
        
        # Let's try "Least Constraining Value" heuristic?
        # Or just run more restarts.
        
        if curr_count > best_count:
            best_count = curr_count
            best_set_list = [i for i in range(1, n+1) if S_curr[i]]
            
    return best_set_list
