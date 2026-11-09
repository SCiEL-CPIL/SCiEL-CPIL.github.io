"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def count_mono_4ap(coloring):
    """
    Count the number of monochromatic 4-term arithmetic progressions.
    coloring: numpy array of 0s and 1s, length 256.
    """
    n = len(coloring)
    count = 0
    # Iterate over all possible 4-APs: a, a+d, a+2d, a+3d
    # d ranges from 1 to floor((n-1)/3)
    # a ranges from 1 to n - 3*d
    # Using 0-indexed: indices are i, i+d, i+2d, i+3d
    # i + 3d < n  => i < n - 3d
    for d in range(1, n // 3 + 1):
        # max i such that i + 3*d < n
        max_i = n - 3 * d
        for i in range(max_i):
            c1 = coloring[i]
            c2 = coloring[i + d]
            c3 = coloring[i + 2 * d]
            c4 = coloring[i + 3 * d]
            if c1 == c2 == c3 == c4:
                count += 1
    return count

def thue_morse(n):
    """Generate the first n bits of the Thue-Morse sequence."""
    # Thue-Morse t(n) is the parity of the number of 1s in the binary representation of n.
    # We can generate it efficiently using vectorized operations or a simple loop.
    # Since n=256 is small, a simple loop or np.vectorize is fine, but let's be explicit.
    tm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits in i
        # i is 0-indexed in tm, but Thue-Morse is usually defined for integers.
        # t(0)=0, t(1)=1, t(2)=1, t(3)=0, etc.
        # Python's bin(i).count('1') works.
        tm[i] = bin(i).count('1') % 2
    return tm

def local_search(initial_coloring, max_time=100.0):
    """
    Perform local search to minimize monochromatic 4-APs.
    Uses a greedy hill-climbing approach with random restarts if stuck or time allows.
    """
    n = len(initial_coloring)
    current_coloring = initial_coloring.copy()
    current_score = count_mono_4ap(current_coloring)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Pre-compute the list of 4-APs to speed up delta calculation?
    # Or just recount? Recounting is O(N * D) ~ 256 * 85 ~ 20k operations. 
    # Very fast. Let's just recount for simplicity and robustness.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize the order of positions to break symmetry and avoid cycles
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            # Try flipping bit i
            old_bit = current_coloring[i]
            new_bit = 1 - old_bit
            current_coloring[i] = new_bit
            
            new_score = count_mono_4ap(current_coloring)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                current_coloring[i] = old_bit
                
    return current_coloring, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 256
    
    # Strategy:
    # 1. Generate Thue-Morse sequence.
    # 2. Run local search on it.
    # 3. If time remains, try random restarts or perturbations.
    
    # Initial Thue-Morse
    tm = thue_morse(n)
    
    best_coloring = None
    best_score = float('inf')
    
    # Run local search from Thue-Morse
    # We allocate most of the time to the first run, which starts from a good heuristic.
    # If we have time left, we can do more restarts.
    
    # Let's run local search with a generous time budget for the first pass
    # But we need to return the result before the hard deadline.
    # Let's reserve 5 seconds for the return logic and safety.
    search_time_budget = 105.0
    
    current_coloring, current_score = local_search(tm, max_time=search_time_budget)
    
    if current_score < best_score:
        best_score = current_score
        best_coloring = current_coloring.copy()
        
    # If we still have time, we can try a few random restarts or perturbations of the best solution found
    while time.time() < deadline - 5.0:
        # Perturb the best solution: flip a few random bits
        perturbed = best_coloring.copy()
        num_flips = random.randint(5, 20)
        flip_indices = random.sample(range(n), num_flips)
        for idx in flip_indices:
            perturbed[idx] = 1 - perturbed[idx]
        
        # Run a shorter local search from the perturbed state
        # Time left check
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            break
            
        # Use a fraction of the remaining time for this restart
        # Or just a fixed short duration if it's fast enough
        # Local search on 256 bits is very fast. One full pass is < 0.1s.
        # So we can run many restarts.
        # Let's run local search with a very short time limit to ensure it finishes quickly
        # or just run it to convergence if it's fast.
        
        # Actually, local_search above has a time limit. Let's give it a small slice.
        # If the previous run converged, the next one might also converge quickly.
        # Let's just run it with a small max_time, e.g., 2.0s, and see if it improves.
        c, s = local_search(perturbed, max_time=min(2.0, time_left))
        
        if s < best_score:
            best_score = s
            best_coloring = c.copy()
            
    # Return the best coloring found
    # The interface expects 256 colours in {0,1} for integers 1..256.
    # Our array is 0-indexed, so index i corresponds to integer i+1.
    # The return value should be a list of 256 ints.
    return list(best_coloring)

# EVOLVE-BLOCK-END
