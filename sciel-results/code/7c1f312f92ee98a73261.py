"""Rewritten program for low-discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute all arithmetic progressions
    # An AP is (start, step, length) where indices are start, start+step, ..., start+(length-1)*step
    # We need 0-indexed: start in [0, n), step in [1, n), length >= 1
    # But we only care about the sum of signs over the AP.
    # Let's precompute for each AP the list of indices.
    
    # Actually, for efficiency, let's precompute the APs as lists of indices.
    # There are O(n^2) APs roughly. For n=256, this is manageable.
    # step=1: 256 APs (lengths 1..256 starting at 0..255) -> actually for each start, multiple lengths
    # Wait, the task says "any start, positive step and length including one".
    # So for a given start and step, the AP can have any length >= 1 such that the indices are in [0, n).
    # This means for each (start, step), we have multiple APs of different lengths.
    # However, the maximum absolute sum over all APs will be achieved by some AP.
    # Note that for a fixed start and step, the sums for different lengths are related.
    # But we need to check all of them.
    
    # Let's count the number of APs.
    # For step=1: starts 0..255, lengths 1..(256-start). Total ~ 256*128 = 32768
    # For step=2: starts 0..255, lengths 1..ceil((256-start)/2). Total ~ 256*64 = 16384
    # ...
    # Total is roughly sum_{d=1}^{256} sum_{s=0}^{255} ceil((256-s)/d)
    # This is about n^2 * H_n ~ 256^2 * 6 ~ 400k. This is too many to precompute all explicitly and store.
    
    # Alternative approach: Use a heuristic that evaluates the max discrepancy efficiently.
    # For a given sign vector, the max discrepancy is the max over all APs of |sum|.
    # We can compute this by iterating over all (start, step) pairs and computing the running sum.
    
    # Let's precompute the list of (start, step) pairs.
    # For each (start, step), the APs are prefixes of the sequence start, start+step, ...
    # So we just need to compute the prefix sums for each (start, step) and take the max absolute value.
    
    # Number of (start, step) pairs: 256 * 256 = 65536. This is manageable.
    
    # Precompute the list of (start, step) pairs.
    ap_pairs = []
    for step in range(1, n + 1):
        for start in range(n):
            ap_pairs.append((start, step))
    
    # Function to compute the maximum absolute signed sum on any AP
    def max_discrepancy(signs):
        max_disc = 0
        for start, step in ap_pairs:
            s = 0
            idx = start
            while idx < n:
                s += signs[idx]
                abs_s = abs(s)
                if abs_s > max_disc:
                    max_disc = abs_s
                idx += step
        return max_disc
    
    # Initialize with a random sign vector
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    best_signs = signs[:]
    best_disc = max_discrepancy(signs)
    
    # Simulated annealing or hill climbing
    # Let's use a simple hill climbing with random restarts.
    # Or simulated annealing.
    
    # Let's try simulated annealing.
    temperature = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        idx = rng.randint(0, n - 1)
        old_sign = signs[idx]
        new_sign = -old_sign
        signs[idx] = new_sign
        
        new_disc = max_discrepancy(signs)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = signs[:]
        else:
            # Accept with probability exp(-(new_disc - best_disc) / temperature)
            # But we should compare with current, not best.
            # Let's use current_disc instead of best_disc for acceptance.
            # Actually, we need to track current_disc.
            # Let's restructure.
            
            # For now, let's just accept if new_disc <= current_disc, or with some probability.
            # This is a simplification. Let's do proper SA.
            
            # We need current_disc. Let's compute it.
            # But we already have new_disc. We need the old_disc (current before flip).
            # We don't have it stored. Let's store it.
            
            # Let's restart this loop with proper SA.
            break
    
    # Proper simulated annealing
    signs = best_signs[:]
    current_disc = max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs[:]
    
    temperature = 5.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        idx = rng.randint(0, n - 1)
        signs[idx] = -signs[idx]
        
        new_disc = max_discrepancy(signs)
        
        delta = new_disc - current_disc
        
        if delta < 0:
            # Accept
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
        else:
            # Accept with probability exp(-delta / temperature)
            if rng.random() < math.exp(-delta / temperature):
                current_disc = new_disc
            else:
                # Reject: flip back
                signs[idx] = -signs[idx]
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    return best_signs

# EVOLVE-BLOCK-END
