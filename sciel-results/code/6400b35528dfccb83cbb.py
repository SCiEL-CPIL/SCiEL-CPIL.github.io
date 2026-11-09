import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = float('inf')
    best_poly = None
    
    # Initialize with the seed polynomial to ensure we have a valid baseline
    # Seed: [1]*12 -> degrees 0..11 have coeff 1.
    # Square has 13 non-zeros.
    seed_poly = [0] * 41
    for i in range(12):
        seed_poly[i] = 1
    best_poly = seed_poly
    best_score = _count_nonzero_square(seed_poly)
    
    # Random restarts and local search
    while time.time() < deadline:
        # Create a random candidate
        # Choose 12 distinct indices from 0..40
        indices = random.sample(range(41), 12)
        poly = [0] * 41
        for idx in indices:
            poly[idx] = random.randint(-64, 64)
            if poly[idx] == 0:
                poly[idx] = 1 # Ensure non-zero
        
        # Local search
        current_score = _count_nonzero_square(poly)
        
        # If this random start is already better, update best
        if current_score < best_score:
            best_score = current_score
            best_poly = poly[:]
        
        # Local search loop
        no_improve_count = 0
        while time.time() < deadline and no_improve_count < 100:
            # Pick a random move
            move_type = random.choice(['change_coeff', 'move_index'])
            
            if move_type == 'change_coeff':
                # Pick a random non-zero index
                nz_indices = [i for i in range(41) if poly[i] != 0]
                idx = random.choice(nz_indices)
                old_val = poly[idx]
                new_val = random.randint(-64, 64)
                if new_val == 0:
                    # If we make it zero, we need to maintain 12 non-zeros? 
                    # The problem says "exactly 12 nonzero terms".
                    # So we cannot make it zero unless we introduce another non-zero.
                    # It's safer to just change to a different non-zero value.
                    new_val = old_val + random.choice([-1, 1])
                    if new_val == 0:
                        new_val = 1
                    if abs(new_val) > 64:
                        new_val = 64 if new_val > 0 else -64
                    
                poly[idx] = new_val
                new_score = _count_nonzero_square(poly)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_poly = poly[:]
                else:
                    poly[idx] = old_val
                    no_improve_count += 1
                    
            else: # move_index
                # Pick a random non-zero index and move it to a random zero index
                nz_indices = [i for i in range(41) if poly[i] != 0]
                z_indices = [i for i in range(41) if poly[i] == 0]
                
                if not z_indices:
                    no_improve_count += 1
                    continue
                    
                src_idx = random.choice(nz_indices)
                dst_idx = random.choice(z_indices)
                
                val = poly[src_idx]
                poly[src_idx] = 0
                poly[dst_idx] = val
                
                new_score = _count_nonzero_square(poly)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_poly = poly[:]
                else:
                    poly[dst_idx] = 0
                    poly[src_idx] = val
                    no_improve_count += 1

    return best_poly

def _count_nonzero_square(poly):
    # Compute square of poly
    # poly is a list of 41 ints
    # Square will have degree up to 80
    sq = [0] * 81
    for i in range(41):
        if poly[i] == 0:
            continue
        pi = poly[i]
        for j in range(41):
            if poly[j] == 0:
                continue
            pj = poly[j]
            sq[i+j] += pi * pj
            
    count = 0
    for c in sq:
        if c != 0:
            count += 1
    return count
