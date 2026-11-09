import numpy as np
import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    MAX_TIME = 55.0  # Leave buffer for evaluation overhead
    start_time = time.time()
    
    # We want to maximize the number of words.
    # Start with the zero word.
    # Try multiple random orderings of the remaining 2^N - 1 words.
    
    best_code = [0]
    best_size = 1
    
    # Pre-compute all possible words? 2^17 = 131072. This is small enough to store in a list.
    all_words = list(range(2**N))
    # Remove 0 from candidates for the first step, or just handle it.
    candidates_pool = all_words[1:] # 1 to 2^17-1
    
    # To speed up distance checks, we can't easily vectorize the "check against all existing" part in a pure Python loop without numpy overhead per word.
    # However, we can use a simple loop.
    
    # Strategy: 
    # 1. Try a few random shuffles.
    # 2. For each shuffle, greedily add valid words.
    # 3. Keep the largest code found.
    
    # Optimization: If we find a code of size S, we can stop adding if we are not likely to beat it? No, just run.
    
    # We can also try a "Weight-based" ordering first, as low weight words might be easier to pack? 
    # Or high weight? 
    # Actually, random is usually best for greedy packing.
    
    # Let's do 5-10 restarts.
    # To ensure we don't time out, we monitor time.
    
    # A better heuristic: Start with a random valid code of size ~200?
    # No, let's stick to greedy from 0, it's deterministic in structure but random in order.
    
    # One improvement: "Greedy with Backtracking" is too slow.
    # "Greedy with Random Restarts" is the standard.
    
    # Let's try to optimize the inner loop.
    # Instead of checking distance to all words in `current_code`, we can maintain a set of "blocked" words?
    # No, the set of words within dist < 6 from ANY word in the code is the union of Hamming balls.
    # The size of a Hamming ball of radius 5 in N=17 is sum(C(17,i) for i=0..5) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402.
    # If code size is 300, the union of balls is at most 300 * 9402 ~ 2.8M. But they overlap.
    # Total space is 131k. So eventually, the "blocked" set covers most of the space.
    # Checking if a candidate is in the "blocked" set is O(1) if we maintain a boolean array or a set of blocked words.
    # Updating the blocked set when adding a new word w:
    #   For all v in [0, 2^N): if dist(w, v) < 6: block[v] = True.
    # This is O(2^N) per addition. 300 * 131k = 39M operations. This is feasible.
    # And checking a candidate is O(1).
    # Total complexity: O(K * 2^N) for updates + O(2^N) for scan.
    # 300 * 131k = 39M. In Python, 39M simple operations might take ~2-4 seconds per restart.
    # 10 restarts = 20-40 seconds. This is risky but possible.
    
    # Let's implement this "Blocked Set" greedy approach.
    
    def greedy_packing(order):
        """
        order: a list of candidates to consider (shuffled)
        Returns the code found.
        """
        current_code = [0]
        # blocked[i] is True if i is within distance < MIN_DIST of any word in current_code
        blocked = [False] * (2**N)
        
        # Initialize blocked with neighbors of 0
        # Neighbors of 0 are words with weight < 6
        # We can precompute neighbors for 0? Or just compute.
        # Actually, let's just compute neighbors for the first word 0.
        for w in range(2**N):
            if w.bit_count() < MIN_DIST:
                blocked[w] = True
        
        # Iterate through the shuffled candidates
        for cand in order:
            if blocked[cand]:
                continue
            
            # If not blocked, add it
            current_code.append(cand)
            
            # Update blocked set: mark all neighbors of cand as blocked
            # Neighbors are words v such that dist(cand, v) < MIN_DIST
            # i.e., dist(0, cand ^ v) < MIN_DIST
            # i.e., (cand ^ v).bit_count() < MIN_DIST
            # v = cand ^ u, where u.bit_count() < MIN_DIST
            # So we iterate over all u with weight < MIN_DIST, and mark blocked[cand ^ u] = True
            
            # Precompute list of offsets with weight < MIN_DIST
            # This is constant for all words.
            offsets = [u for u in range(2**N) if u.bit_count() < MIN_DIST]
            
            for u in offsets:
                blocked[cand ^ u] = True
                
        return current_code

    # Precompute offsets once
    offsets = [u for u in range(2**N) if u.bit_count() < MIN_DIST]
    # Note: The greedy_packing function above re-computes offsets or uses global? 
    # Let's pass it or make it global.
    
    # Wait, the "blocked" array initialization for 0 can be done using offsets too.
    # blocked[u] = True for u in offsets.
    
    def greedy_packing_opt(order, offsets):
        current_code = [0]
        blocked = [False] * (2**N)
        
        # Mark neighbors of 0
        for u in offsets:
            blocked[u] = True
            
        for cand in order:
            if blocked[cand]:
                continue
            
            current_code.append(cand)
            
            # Mark neighbors of cand
            for u in offsets:
                blocked[cand ^ u] = True
                
        return current_code

    # We will try several random shuffles.
    # To save time, we can stop early if we exceed best_size and time is running out?
    # Or just run fixed number of restarts.
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time. Let's start with 5.
    # If we have time, we can do more.
    
    # However, 5 restarts * (300 * 9400 updates + 131k checks)
    # Updates: 300 * 9400 = 2.8M. 
    # Checks: 131k.
    # Total ops per restart ~ 3M.
    # 5 restarts ~ 15M ops. Very fast.
    # Wait, the update loop `for u in offsets` is 9400 iterations.
    # `blocked[cand ^ u] = True` is fast.
    # So 300 * 9400 = 2.8M assignments.
    # This is very fast in Python (approx 0.1s).
    # So we can do MANY restarts.
    
    # Let's do 50 restarts.
    # 50 * 0.1s = 5s.
    # We have 55s. We can do 500 restarts?
    # 500 * 0.1s = 50s.
    # Let's try to do as many as possible until time limit.
    
    best_code = [0]
    best_size = 1
    
    # To vary the result, we shuffle the candidate list.
    # The candidate list is [1, 2, ..., 2^N-1].
    
    for i in range(500):
        if time.time() - start_time > MAX_TIME:
            break
            
        # Create a shuffled order
        # Shuffling 131k items takes some time.
        # random.shuffle is O(N).
        order = candidates_pool[:]
        random.shuffle(order)
        
        code = greedy_packing_opt(order, offsets)
        
        if len(code) > best_size:
            best_size = len(code)
            best_code = code
            # We can print or log, but no output allowed.
            
    return best_code
