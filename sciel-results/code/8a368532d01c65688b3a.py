"""Strategy: High-Throughput Local Search via Vectorized Bitset Evaluation.
1. Represent the set S as a bitset (Python int or NumPy array).
2. Evaluate the score (number of distinct d) using vectorized bitwise operations.
   For a given set S (bitset), a difference d is valid if (S & (S >> d) & (S >> 2d)) != 0.
   We can iterate d from 1 to max_possible_diff and count how many yield a non-zero intersection.
   This is much faster than set lookups in Python.
3. Perform aggressive Hill Climbing with many random restarts.
   - Start with random or structured sets.
   - In each step, try swapping one element for another (or moving an element).
   - Accept if score improves.
   - Since evaluation is fast, we can run millions of steps.
4. Use a time guard to return the best found set.
"""

import time
import random
import numpy as np

def _score_from_bitset(bitset, max_val=512):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    bitset: Python int where bit i is 1 if (i+1) is in the set.
    max_val: The maximum value in the universe (512).
    """
    count = 0
    # d can range from 1 to max_val // 2 (since a+2d <= max_val)
    # Actually, a >= 1, so a+2d <= 512 => 2d <= 511 => d <= 255.
    # But we can just check up to max_val.
    # Optimization: Pre-compute shifts? No, dynamic is fine.
    # We check if there exists a such that bits at a, a+d, a+2d are set.
    # This is equivalent to: (bitset & (bitset >> d) & (bitset >> 2*d)) != 0
    
    # We can vectorize this if we use numpy, but Python int bitwise ops are very fast (C speed).
    # Let's stick to Python int for simplicity and speed for single-threaded.
    
    # To speed up, we can limit the range of d.
    # If the set is sparse, large d are unlikely.
    # But we must check all d to be correct.
    # Max d is 255.
    # 255 iterations of bitwise ops on 512-bit ints is extremely fast.
    
    for d in range(1, max_val // 2 + 1):
        # Check if any triple exists for this d
        # S & (S >> d) & (S >> 2d)
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
    return count

def _bitset_to_list(bitset):
    """Convert bitset to list of integers (1-indexed)."""
    # Extract set bits
    res = []
    b = bitset
    idx = 0
    while b:
        if b & 1:
            res.append(idx + 1)
        b >>= 1
        idx += 1
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n_target = 32
    max_val = 512
    max_d = max_val // 2
    
    best_score = -1
    best_bitset = 0
    
    # Pre-generate a pool of random starting points and structured points
    # We will run multiple restarts.
    
    # Helper to create a random bitset with n_target bits set
    def random_bitset(n):
        # Choose n distinct indices from 0 to 511
        indices = random.sample(range(max_val), n)
        b = 0
        for i in indices:
            b |= (1 << i)
        return b

    # Helper to create a structured bitset (e.g., interval)
    def interval_bitset(start, n):
        # start is 0-indexed
        b = 0
        for i in range(start, start + n):
            if i < max_val:
                b |= (1 << i)
        return b

    # List of initial candidates
    candidates = []
    
    # 1. Dense intervals at various positions
    for start in range(0, max_val - n_target + 1, 10):
        candidates.append(interval_bitset(start, n_target))
        
    # 2. Random sets
    for _ in range(50):
        candidates.append(random_bitset(n_target))
        
    # 3. "Checkerboard" or modular structures
    for step in [2, 3, 4, 5]:
        # Pick n_target numbers with this step, wrapping or shifting
        base = random.randint(0, max_val)
        b = 0
        count = 0
        for i in range(max_val):
            if (i - base) % step == 0:
                b |= (1 << i)
                count += 1
                if count == n_target:
                    break
        if count == n_target:
            candidates.append(b)

    # Local Search / Hill Climbing
    # We will iterate through candidates. For each, we do a greedy/hill-climb phase.
    
    # To make hill climbing faster, we pre-compute the "neighborhood" effects?
    # No, just re-evaluate. With ~10us per eval, we can do ~100k evals/sec.
    # 145s * 100k = 14.5 million evals.
    # Each eval is a swap.
    
    # We need a fast way to pick a swap.
    # Random swap: pick one element to remove, one to add.
    
    all_indices = list(range(max_val))
    
    for start_b in candidates:
        if time.time() > deadline - 5.0:
            break
            
        current_b = start_b
        current_score = _score_from_bitset(current_b)
        
        if current_score > best_score:
            best_score = current_score
            best_bitset = current_b
            
        # Hill Climbing Phase
        # We try to improve current_b.
        # Stopping condition: no improvement in X steps, or time limit.
        
        no_improve_count = 0
        max_no_improve = 2000  # Adjust based on speed
        
        while no_improve_count < max_no_improve:
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random element to remove
            # We need to extract the set of indices from current_b for this.
            # To avoid overhead, we can maintain a list of indices alongside the bitset.
            # But for simplicity in this draft, let's extract indices.
            # Optimization: Maintain the list of indices.
            
            # Let's refactor to maintain indices list for the current state in the loop.
            # Actually, extracting 32 bits from a 512-bit int is fast.
            
            indices = _bitset_to_list(current_b)
            
            # Pick one to remove
            rem_idx = random.choice(indices)
            # Pick one to add (not in indices)
            # To pick a random valid add, we can pick random 1..512 and check.
            add_idx = random.randint(1, max_val)
            if add_idx in indices:
                # Retry or pick another
                # Simple retry
                for _ in range(10):
                    add_idx = random.randint(1, max_val)
                    if add_idx not in indices:
                        break
                else:
                    continue
                    
            # Perform swap
            new_b = current_b
            new_b &= ~(1 << (rem_idx - 1))
            new_b |= (1 << (add_idx - 1))
            
            new_score = _score_from_bitset(new_b)
            
            if new_score > current_score:
                current_b = new_b
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_bitset = current_b
            else:
                no_improve_count += 1
                
            # Optional: Simulated Annealing step?
            # If we just do hill climbing, we get stuck.
            # Let's add a small probability of accepting a worse move (SA-like) to escape.
            # But with many restarts, pure hill climbing might be enough if we have many random restarts.
            # Let's stick to strict hill climbing for speed and simplicity, relying on the 50+ restarts.
            
        # End of hill climb for this candidate

    # If we have time, do a final "polish" on the best found set
    if best_bitset and time.time() < deadline - 10.0:
        current_b = best_bitset
        current_score = _score_from_bitset(current_b)
        no_improve_count = 0
        max_no_improve = 5000
        
        while no_improve_count < max_no_improve and time.time() < deadline - 2.0:
            indices = _bitset_to_list(current_b)
            rem_idx = random.choice(indices)
            add_idx = random.randint(1, max_val)
            if add_idx in indices:
                continue
                
            new_b = current_b
            new_b &= ~(1 << (rem_idx - 1))
            new_b |= (1 << (add_idx - 1))
            
            new_score = _score_from_bitset(new_b)
            
            if new_score > current_score:
                current_b = new_b
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_bitset = current_b
            else:
                no_improve_count += 1

    if best_bitset == 0:
        return list(range(1, 33))
        
    return _bitset_to_list(best_bitset)

# EVOLVE-BLOCK-END
