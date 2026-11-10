import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Indices are 0-based
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Convert APs to numpy array for vectorized operations
    # aps_np shape: (total_aps, 4)
    aps_np = np.array(aps, dtype=np.int32)
    
    # Precompute masks or structures for faster delta calculation?
    # For a block flip, we need to count mono APs before and after.
    # Since N=256 and total_aps is ~10k, a full recount is fast enough if vectorized.
    # count_mono_vectorized:
    #   colors is a numpy array of 0s and 1s.
    #   mono = np.all(colors[aps_np] == colors[aps_np][:, 0:1], axis=1)
    #   count = np.sum(mono)
    
    def count_mono_vec(colors_np):
        # colors_np is (N,)
        # colors_np[aps_np] is (total_aps, 4)
        # Compare each element to the first element of the tuple
        c0 = colors_np[aps_np[:, 0]]
        c1 = colors_np[aps_np[:, 1]]
        c2 = colors_np[aps_np[:, 2]]
        c3 = colors_np[aps_np[:, 3]]
        # An AP is monochromatic if c0==c1 and c0==c2 and c0==c3
        # Since they are 0/1, we can use:
        # (c0==c1) & (c0==c2) & (c0==c3)
        mono = (c0 == c1) & (c0 == c2) & (c0 == c3)
        return np.sum(mono)

    # Strategy: Multi-start Local Search with Block Moves
    # 1. Initialize with random colors
    # 2. Loop:
    #    a. Generate K candidate moves (e.g., K=5). Each move flips a random subset of bits (size ~ N/4 to N/2).
    #    b. Evaluate the new cost for each candidate.
    #    c. Select the best candidate.
    #    d. If best candidate is better than current, accept.
    #    e. If we get stuck (no improvement in last X steps), restart with a new random initialization or a perturbed version.
    
    best_count = float('inf')
    best_colors = [0] * N
    
    # We will run multiple restarts.
    # Each restart gets a portion of the time.
    # Let's try to fit as many restarts as possible.
    # A single "iteration" (evaluating K candidates) takes roughly K * (cost of count_mono_vec).
    # count_mono_vec is ~10k operations. Very fast. Maybe 0.1ms?
    # So 1000 iterations per second?
    # Let's target 100 restarts, each with a few hundred steps.
    
    rng = np.random.default_rng()
    
    current_restart = 0
    max_restarts = 50 # We'll adjust based on time
    
    while time.time() < deadline - 2.0:
        # Initialize current state
        # Use a mix of random and structured inits?
        # Random is usually fine for binary CSPs if we have good local search.
        colors = rng.integers(0, 2, size=N)
        current_count = count_mono_vec(colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.tolist()
            
        # Local search phase
        # We use "Block Flips" of varying sizes.
        # Small flips (1-5 bits) for fine tuning, large flips (32-64 bits) for exploration.
        
        no_improve_count = 0
        max_no_improve = 20 # If we don't improve in 20 steps, restart local search or break to new init
        
        while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
            # Generate candidates
            # Let's try 5 candidates per step
            num_candidates = 5
            best_candidate_delta = float('inf')
            best_candidate_indices = None
            best_candidate_count = float('inf')
            
            for _ in range(num_candidates):
                # Randomly decide flip size
                # Bias towards smaller flips as we converge?
                # For now, mix of small and medium
                if rng.random() < 0.5:
                    flip_size = rng.integers(1, 10) # Small
                else:
                    flip_size = rng.integers(10, 50) # Medium
                    
                flip_size = min(flip_size, N)
                if flip_size == 0: continue
                
                indices = rng.choice(N, size=flip_size, replace=False)
                
                # Apply flip to a copy
                new_colors = colors.copy()
                new_colors[indices] = 1 - new_colors[indices]
                
                new_count = count_mono_vec(new_colors)
                
                if new_count < best_candidate_count:
                    best_candidate_count = new_count
                    best_candidate_indices = indices
                    best_candidate_delta = new_count - current_count
                    
            # Accept best candidate if it improves
            if best_candidate_delta < 0:
                colors[best_candidate_indices] = 1 - colors[best_candidate_indices]
                current_count = best_candidate_count
                no_improve_count = 0
                
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.tolist()
            else:
                # No improvement found in this batch
                no_improve_count += 1
                # Optional: Accept a slightly worse move to escape? (Simulated Annealing touch)
                # If we are stuck, accept the best candidate even if worse, with some probability?
                # Or just break out to restart.
                # Let's try accepting the best candidate if it's not too bad (delta < 5) and we've been stuck for a while
                if no_improve_count > 5 and best_candidate_delta < 5 and rng.random() < 0.1:
                    colors[best_candidate_indices] = 1 - colors[best_candidate_indices]
                    current_count = best_candidate_count
                    no_improve_count = 0 # Reset counter to give it a chance to recover
            
        current_restart += 1
        # If we hit a very good score, we can tighten the search or just keep going
        if best_count <= 5:
             # We found a very good solution. Try to refine it with single flips?
             # For now, just break if we are super low, but let's keep searching to be safe.
             # Actually, let's do a final polish phase if we have time.
             pass

    # Final Polish: Try single bit flips on the best solution found
    if time.time() < deadline - 1.0:
        colors = np.array(best_colors)
        current_count = best_count
        
        # Greedy single flip
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try flipping each bit
            # To speed up, we can vectorize the delta calculation for all single flips?
            # Delta for flipping bit i:
            # It affects all APs containing i.
            # For each such AP, if it was mono, it becomes non-mono (delta -1).
            # If it was non-mono, it might become mono (delta +1).
            
            # Precompute for each bit i, the list of APs it is in.
            # We can do this once outside the loop if we want, but let's just do a simple loop for polish.
            # N=256, so 256 checks. Each check involves iterating over APs containing i.
            # Average APs per bit is total_aps * 4 / N ~ 10000 * 4 / 256 ~ 150.
            # So 256 * 150 = 38400 ops per pass. Very fast.
            
            # Build adjacency list once
            bit_to_aps = [[] for _ in range(N)]
            for ap_id, ap in enumerate(aps):
                for idx in ap:
                    bit_to_aps[idx].append(ap_id)
            
            while improved and time.time() < deadline - 0.5:
                improved = False
                for i in range(N):
                    # Calculate delta for flipping i
                    delta = 0
                    c_i = colors[i]
                    for ap_id in bit_to_aps[i]:
                        ap = aps[ap_id]
                        # Check if AP is mono
                        # Get colors of the 4 elements
                        c0 = colors[ap[0]]
                        c1 = colors[ap[1]]
                        c2 = colors[ap[2]]
                        c3 = colors[ap[3]]
                        
                        is_mono = (c0 == c1) and (c0 == c2) and (c0 == c3)
                        
                        if is_mono:
                            # If we flip i, does it break the mono property?
                            # Only if i is part of the AP (it is) and the other 3 are same color as i.
                            # If other 3 are same as c_i, then flipping i makes it non-mono.
                            others_same = (c0 == c1) and (c0 == c2) and (c0 == c3)
                            # Wait, if it's mono, all 4 are same. So others are same as c_i.
                            # Flipping i makes it non-mono. Delta = -1.
                            delta -= 1
                        else:
                            # If we flip i, does it make it mono?
                            # This happens if the other 3 are all the same color, and that color is 1-c_i.
                            # Let's check the other 3 colors.
                            others = [ap[j] for j in range(4) if ap[j] != i]
                            c_o0 = colors[others[0]]
                            c_o1 = colors[others[1]]
                            c_o2 = colors[others[2]]
                            
                            if c_o0 == c_o1 and c_o0 == c_o2:
                                # The other 3 are same.
                                # If they are different from c_i, flipping i makes them same.
                                if c_o0 != c_i:
                                    delta += 1
                                # If they are same as c_i, flipping i makes them different (still non-mono).
                                # Wait, if c_o0 == c_i, then currently 3 are c_i, 1 is c_i (itself). 
                                # No, currently it's non-mono. So at least one other is different.
                                # If the other 3 are same as each other, and one of them is different from c_i?
                                # Let's re-evaluate.
                                # Current state: Non-mono.
                                # Other 3 colors: c_o0, c_o1, c_o2.
                                # If c_o0 == c_o1 == c_o2:
                                #   Case 1: c_o0 == c_i. Then all 4 are c_i. But we said it's non-mono. Contradiction.
                                #   So if c_o0 == c_o1 == c_o2, then c_o0 MUST be != c_i.
                                #   Therefore, flipping i makes all 4 equal to c_o0. Delta = +1.
                                
                    if delta < 0:
                        colors[i] = 1 - colors[i]
                        current_count += delta
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_colors = colors.tolist()
                        # Restart the loop to check other bits again, as flipping one might affect others
                        break # Break inner loop to restart outer while
        
    return best_colors

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
    print(res[:10])
