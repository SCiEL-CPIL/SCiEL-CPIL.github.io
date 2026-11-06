"""
Strategy: Randomized Greedy Packing with Multiple Restarts.
The previous approach used a deterministic sequential scan (0, 1, 2, ...),
which is highly sensitive to order and often yields suboptimal packings.
This strategy uses:
1. A better initial seed: The [18, 11, 6] extended quadratic residue code (512 words) 
   is a known strong linear code for A_2(18,6). I will construct it explicitly.
   Wait, constructing the QRY code requires knowing the specific generator matrix or 
   using a known list. Since I cannot fetch external data, I will rely on the 
   Reed-Muller seed (32 words) but enhance the search.
   
   Actually, let's stick to the Reed-Muller seed (32 words) as it is easy to generate
   and valid. The key improvement is the search method.
   
2. Randomized candidate order: Instead of scanning 0..2^N-1, we generate random 
   candidates or shuffle the range. Randomness helps break symmetries and find 
   different local optima.
3. Multiple Restarts: We run the greedy packing multiple times with different 
   random seeds/orders and keep the largest code found.
4. NumPy Acceleration: Vectorized XOR and popcount for fast distance checks.

Why this is different:
- Current: Deterministic, single pass, sequential order.
- New: Stochastic, multiple passes, randomized order, best-of-N selection.
"""

import time
import random
import numpy as np

MINIMUM_DISTANCE = 6
N = 18
TOTAL_SPACE = 1 << N

def _seed_words():
    """
    Generate the first-order Reed-Muller code RM(1, 4) extended by 2 zero bits.
    Length 18, size 32, min distance 8.
    """
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                # Affine function: dot(slope, point) + intercept
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The last 2 bits are 0 (extension)
            words.append(word)
    return words

def _greedy_pack(seed_code, candidates, deadline):
    """
    Greedily add candidates to the code if they maintain min distance.
    candidates: list of candidate integers to try.
    deadline: absolute time limit.
    Returns the final code list.
    """
    code = list(seed_code)
    if not code:
        return code
    
    # Convert to numpy array for fast vectorized operations
    code_np = np.array(code, dtype=np.uint32)
    
    # We iterate through the provided candidates
    for candidate in candidates:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Vectorized distance check
        # code_np ^ candidate gives XOR for all codewords
        # np.bit_count gives population count
        dists = np.bit_count(code_np ^ candidate)
        
        if dists.min() >= MINIMUM_DISTANCE:
            code.append(candidate)
            # Update numpy array
            # To avoid frequent reallocation, we could use a larger buffer, 
            # but for simplicity and correctness, we re-allocate.
            # Optimization: Only re-allocate if we added a word.
            code_np = np.array(code, dtype=np.uint32)
            
    return code

def construct():
    start_time = time.time()
    # Safety margin: 5 seconds before the 90s limit
    deadline = start_time + 85.0
    
    seed = _seed_words()
    best_code = seed
    
    # Strategy: Run multiple randomized greedy searches.
    # Each search uses a different random permutation of the candidate space
    # or a random subset.
    
    # Since 2^18 = 262,144 is small enough to shuffle in Python reasonably fast,
    # we can create a full list of candidates and shuffle it.
    # However, shuffling 262k items takes ~10-20ms.
    # Checking 262k candidates against M words takes time.
    # Let's estimate: 
    # If M ~ 100, 262k * 100 = 26M ops. NumPy does this in < 1s.
    # So we can afford multiple full scans with different orders.
    
    # How many restarts can we do in 85s?
    # Let's try to do as many as possible.
    
    candidates_all = list(range(TOTAL_SPACE))
    
    rng = random.Random(42) # Fixed seed for reproducibility of the 'random' strategy
    
    restart_count = 0
    while time.time() < deadline - 10.0: # Leave 10s buffer for final checks/overhead
        restart_count += 1
        
        # Shuffle candidates
        # Shuffling 262k items is O(N) and fast.
        shuffled_candidates = candidates_all.copy()
        rng.shuffle(shuffled_candidates)
        
        # Run greedy pack
        current_code = _greedy_pack(seed, shuffled_candidates, deadline)
        
        # Keep the best code
        if len(current_code) > len(best_code):
            best_code = current_code
            # Optional: Break if we reached a theoretical upper bound? 
            # A_2(18,6) is known to be at least 512 (from QRY code).
            # If we find > 512, that's a new record.
            # If we find 512, we are matching the known best linear code.
            # Nonlinear might be larger?
            # Actually, A_2(18,6) is likely 512 or slightly more.
            # Let's just keep searching.

        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
            
    # Final verification is done by the evaluator, but we should ensure 
    # best_code is valid. The greedy algorithm guarantees validity.
    
    return best_code

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} words in {t1-t0:.2f}s")
    # Verify min distance
    min_d = 999
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            d = (res[i] ^ res[j]).bit_count()
            if d < min_d:
                min_d = d
    print(f"Min distance: {min_d}")
