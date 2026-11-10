import numpy as np
import time
import bisect
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45  # 45 seconds margin to be safe within 60s (90s limit)
    
    N = 10**7
    
    # 1. Compute sigma(n) for n=1..N
    # Using a sieve approach with NumPy for speed
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # We can optimize the sieve loop slightly by skipping even numbers?
    # No, sigma is multiplicative but the sieve `sigma[i::i] += i` is simple and effective.
    # However, iterating 10^7 times in Python has overhead.
    # We can vectorize the loop? No, the step changes.
    # Let's just run the standard loop. It's usually ~2-5 seconds for 10^7.
    
    for i in range(1, N + 1):
        if time.time() > deadline - 5:
            # Fallback if sieve is too slow
            return [11, 6]
        sigma[i::i] += i

    # 2. Group indices by sigma value
    # We want to find pairs (a, b) such that sigma(a) == sigma(b).
    # Instead of storing all indices for every sigma (which uses lots of memory),
    # we can iterate through b and look for a.
    
    # Strategy:
    # We want to minimize |a/b - 1.5|.
    # This is equivalent to minimizing |2a - 3b| / (2b).
    # For a fixed b, we want a close to 1.5b.
    # Let target_a = 1.5 * b.
    # We need sigma(a) == sigma(b).
    
    # Approach A: Group by sigma, then for each group, find best pair.
    # Approach B: Iterate b, compute target_a, check candidates near target_a.
    
    # Approach B is risky if we don't check enough candidates.
    # Approach A is robust but requires building the groups.
    
    # Let's refine Approach A:
    # Build a dictionary: sigma_val -> list of n
    # But 10^7 entries is a lot.
    # However, many sigma values are unique.
    # We only care about groups with size >= 2.
    
    # To save memory and time, we can use a different strategy:
    # Sort indices by sigma value?
    # Or just use a dict.
    
    # Let's try the dict approach but be careful with time.
    
    groups = defaultdict(list)
    
    # Iterate from 1 to N
    # To speed up, we can use numpy to get unique sigma values and their indices?
    # np.unique(sigma) returns unique values and inverse indices.
    # But we need to map back to original indices.
    
    # Alternative:
    # We can iterate b from N down to 1.
    # For each b, we want to find a such that sigma(a) == sigma(b) and a is close to 1.5b.
    # If we have a data structure that allows us to query: "Given a sigma value S, find the number in the set closest to X",
    # we could do this efficiently.
    
    # Since we have the full sigma array, we can pre-process.
    # Let's stick to the group-by-sigma approach but optimize the search within groups.
    
    # To avoid the cost of building a huge dict with lists, we can:
    # 1. Sort indices by sigma value.
    # 2. Iterate through sorted indices, keeping track of contiguous blocks with same sigma.
    
    # Sorting 10^7 integers in numpy is fast (~0.5s).
    # argsort = np.argsort(sigma)
    # sorted_sigma = sigma[argsort]
    # Then iterate through sorted_sigma to find blocks.
    
    # This avoids the dict overhead and Python loop for building groups.
    
    argsort = np.argsort(sigma)
    sorted_sigma = sigma[argsort]
    sorted_indices = argsort
    
    # Now, iterate through sorted_indices to find blocks with same sigma.
    # For each block, we have a list of indices (from sorted_indices) that share the same sigma.
    # We want to find a pair (a, b) in this block that minimizes |a/b - 1.5|.
    
    # Note: The indices in `sorted_indices` are the original n values.
    # So for a block, we have a set of numbers.
    # We want to find two numbers a, b in this set such that |a/b - 1.5| is minimized.
    
    # Since the numbers in a block are not sorted by value (they are sorted by sigma),
    # we should sort the indices in each block to use bisect.
    
    # However, most blocks are size 1.
    # We can skip size 1 blocks quickly.
    
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # Iterate through the sorted array to find blocks
    i = 0
    n_elements = len(sorted_sigma)
    
    while i < n_elements:
        if time.time() > deadline - 5:
            break
            
        # Find the end of the current block
        current_sigma = sorted_sigma[i]
        j = i + 1
        while j < n_elements and sorted_sigma[j] == current_sigma:
            j += 1
            
        block_size = j - i
        
        if block_size >= 2:
            # Extract the indices for this block
            block_indices = sorted_indices[i:j]
            
            # Sort the indices by value to allow efficient search for a close to 1.5*b
            block_indices.sort()
            
            # Now, for each b in block_indices, find a in block_indices closest to 1.5*b
            # We can iterate through b and use bisect on block_indices.
            
            # Optimization: 
            # If the range of values in the block is small, we can do a simple double loop?
            # No, block_size can be large (e.g., for small sigma values).
            # But for large n, sigma values are large and unique, so blocks are small.
            # The expensive blocks are for small sigma values, which correspond to small n.
            # For small n, block_size might be large, but n is small, so 1.5*b is small.
            
            # Let's use the bisect approach.
            # To speed up, we can limit the search to a window around 1.5*b.
            # But bisect is O(log M) for M elements.
            
            # Iterate through b in block_indices
            # To find the best pair, we can iterate through all b?
            # That is O(M log M) for the block.
            
            # If M is large (e.g., 1000), this is fine.
            # How large can M be?
            # sigma(n) = n for n=1. sigma(1)=1.
            # sigma(2)=3.
            # sigma(3)=4.
            # sigma(4)=7.
            # sigma(6)=12. sigma(8)=15. sigma(10)=18. sigma(12)=28.
            # Collisions are rare for large n.
            
            # Let's just do the bisect search for each b in the block.
            
            for b in block_indices:
                target_a = 1.5 * b
                # Find position in block_indices
                pos = bisect.bisect_left(block_indices, target_a)
                
                # Check pos
                if pos < len(block_indices):
                    a = block_indices[pos]
                    if a != b:
                        # Calculate error exactly to avoid float issues?
                        # |a/b - 1.5| = |2a - 3b| / (2b)
                        # We want to minimize this.
                        # Compare |2a - 3b| / (2b) < best_err
                        # |2a - 3b| < 2b * best_err
                        
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue # Invalid, a/b = 1.5
                        
                        # Compare num / (2b) < best_err
                        # num < 2b * best_err
                        # Using float for comparison is risky for very close values.
                        # But best_err is a float.
                        # Let's use the rational comparison if possible or just float.
                        # Given the constraints, float should be okay for ranking,
                        # but we must ensure the final result is valid.
                        
                        err = num / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            
                # Check pos - 1
                if pos - 1 >= 0:
                    a = block_indices[pos - 1]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue
                        
                        err = num / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            
        i = j

    return [int(best_a), int(best_b)]
