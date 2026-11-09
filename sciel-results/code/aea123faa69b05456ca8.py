"""
Strategy: Vectorized Bitset Constraint Propagation
Instead of stochastic search (SA), we model the problem using bitsets of squared distances.
For a set of points P, let D_p be the set of squared distances from p to all other points in P.
The no-isosceles condition requires:
1. For any p in P, |D_p| == |P|-1 (no two other points at same distance from p).
2. For any distinct p, q in P, (D_p \ {dist(p,q)}) ∩ (D_q \ {dist(p,q)}) == ∅ 
   (This ensures no third point r exists such that dist(p,r) == dist(q,r)).

We use Python integers as arbitrary-precision bitsets for distance sets.
Bit i is set if squared distance i is present.
Max squared distance in [0, 64)^2 is 63^2 + 63^2 = 3969 + 3969 = 7938.
So we need bits up to ~8000. Python ints handle this natively and bitwise ops are fast.

Algorithm:
1. Pre-compute squared distances between all pairs of grid points.
2. Greedy construction with smart ordering (e.g., based on "potential" or just vectorized checks).
3. Local search using bitset operations to quickly identify conflicts and valid additions.
"""

import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START

# Pre-compute geometry
# Grid points
grid_points = [(x, y) for x in range(N) for y in range(N)]
num_points = len(grid_points)

# Map point index to (x, y)
# We will work with indices 0..num_points-1

# Pre-compute squared distances between all pairs
# dist_matrix[i, j] = squared distance between grid_points[i] and grid_points[j]
xs = np.array([p[0] for p in grid_points])
ys = np.array([p[1] for p in grid_points])

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist_matrix = dx * dx + dy * dy
dist_matrix = dist_matrix.astype(np.int32) # Max 7938, fits in int16 but int32 is safe

def _dist_to_bitset(distances):
    """Convert a list/array of distances to a bitset (Python int)."""
    bits = 0
    for d in distances:
        bits |= (1 << d)
    return bits

def _bitset_popcount(bits):
    """Count set bits in a Python int."""
    return bin(bits).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # State representation:
    # selected: list of indices of selected points
    # dist_bitsets: list of ints, dist_bitsets[i] is the bitset of distances from selected[i] to all other selected points
    
    # Initial greedy:
    # We want to pick points that are "far apart" in terms of distance diversity.
    # A simple heuristic: pick points that maximize the minimum distance to existing points?
    # Or just random shuffle + vectorized check.
    
    # Let's try a structured greedy first: 
    # Sort points by some metric? 
    # Actually, a good initial set can be found by trying to add points in a specific order.
    # Let's use a randomized greedy with multiple restarts if time permits, 
    # but the core improvement is the fast validation.
    
    best_selected = []
    best_size = 0

    # We will run a few iterations of "Greedy + Local Search"
    # Local Search: 
    # 1. Try to add a random point.
    # 2. If fails, identify conflicts and try to remove one conflicting point and add the new one (swap).
    
    for restart in range(10): # Limit restarts, each should be fast
        if time.time() > deadline:
            break
            
        # Initialize with a small random set or empty
        selected = []
        dist_bitsets = []
        
        # Greedy Init
        candidates = list(range(num_points))
        random.shuffle(candidates)
        
        # To make init faster, we can process in chunks or just loop.
        # With bitsets, checking a new point against K existing points is O(K) bitwise ops.
        # If K is ~100, it's very fast.
        
        for idx in candidates:
            if time.time() > deadline:
                break
                
            # Check if idx can be added
            # 1. Compute distances from idx to all in selected
            new_dists = dist_matrix[idx, selected]
            
            # Check for duplicates in new_dists (apex condition)
            # If len(set(new_dists)) != len(new_dists), it's invalid.
            # For speed, we can use a set or sort.
            # Since K is small during init, set is fine.
            if len(set(new_dists.tolist())) != len(new_dists):
                continue
                
            # Check base condition:
            # For each existing point i, dist(idx, i) must not be in dist_bitsets[i] 
            # EXCEPT for the distance to idx itself? No, dist_bitsets[i] contains distances to OTHER points.
            # So if dist(idx, i) is already in dist_bitsets[i], then there exists j != idx such that dist(i, j) == dist(i, idx).
            # This forms an isosceles triangle (i, j, idx) with apex i.
            
            valid = True
            for i, sel_idx in enumerate(selected):
                d = new_dists[i]
                if dist_bitsets[i] & (1 << d):
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # If valid, add idx
            # Update dist_bitsets for existing points
            new_bitset = 0
            for d in new_dists:
                new_bitset |= (1 << d)
                
            for i, d in enumerate(new_dists):
                dist_bitsets[i] |= (1 << d)
                
            selected.append(idx)
            dist_bitsets.append(new_bitset)
            
        current_size = len(selected)
        if current_size > best_size:
            best_size = current_size
            best_selected = selected[:]
            
        # Local Search Phase
        # Try to improve by swapping or adding.
        # We maintain a list of "failed" candidates or just sample random ones.
        
        # To speed up local search, we can pre-compute which points are currently "blocked" and why?
        # Too complex. Let's just do random swaps.
        
        ls_start = time.time()
        ls_deadline = deadline - 5 # Reserve 5s for final check
        
        while time.time() < ls_deadline:
            # Pick a random unselected point to try to add
            # Or pick a random selected point to remove and replace
            # Swap is usually more effective
            
            if len(selected) < 2:
                break
                
            # Choose a random point to remove
            remove_idx = random.randint(0, len(selected) - 1)
            remove_pos = selected[remove_idx]
            
            # Tentatively remove
            temp_selected = selected[:]
            temp_dist_bitsets = dist_bitsets[:]
            
            temp_selected.pop(remove_idx)
            temp_dist_bitsets.pop(remove_idx)
            
            # Update remaining bitsets: remove the distance to remove_pos
            d_to_removed = dist_matrix[remove_pos, temp_selected]
            for i, d in enumerate(d_to_removed):
                temp_dist_bitsets[i] &= ~(1 << d)
                
            # Now try to add a random new point
            new_pos = random.randint(0, num_points - 1)
            if new_pos in temp_selected:
                continue
                
            # Check validity of adding new_pos to temp_selected
            new_dists = dist_matrix[new_pos, temp_selected]
            
            if len(set(new_dists.tolist())) != len(new_dists):
                continue
                
            valid = True
            for i, sel_idx in enumerate(temp_selected):
                d = new_dists[i]
                if temp_dist_bitsets[i] & (1 << d):
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Valid swap!
            # Update state
            new_bitset = 0
            for d in new_dists:
                new_bitset |= (1 << d)
                
            for i, d in enumerate(new_dists):
                temp_dist_bitsets[i] |= (1 << d)
                
            temp_selected.append(new_pos)
            temp_dist_bitsets.append(new_bitset)
            
            # Accept the swap (size is same, but diversity might improve for future steps)
            # Actually, pure swap keeps size constant. We need to find a way to GROW.
            # The current logic only swaps. 
            # To grow, we need to find a point that fits into the current set.
            # The greedy init might leave "holes" that a swap can't fill if the swap doesn't change size.
            # We should prioritize ADD if possible.
            
            # Let's modify the loop:
            # 1. Try to add a random point.
            # 2. If fail, try a swap.
            
            # This restart loop structure makes it hard to interleave add/swap efficiently in one while loop 
            # without duplicating code. 
            # Let's refine the inner loop to try ADD first.
            
            # Reset for the "Add" attempt
            # We already have `selected` and `dist_bitsets`.
            # Let's try adding a random point to the CURRENT best set found in this restart.
            # If it works, we grow.
            
            # However, the code above did a swap on `temp_`. 
            # Let's restructure the local search to be cleaner.
            
            pass # The above swap logic was a draft. Let's implement a cleaner LS below.

        # Clean Local Search Implementation
        # We will run a dedicated LS loop here.
        
        # To avoid duplicating the restart logic, we will just run the LS on the `selected` found in greedy.
        
        ls_deadline = deadline - 2
        last_improvement = time.time()
        
        while time.time() < ls_deadline:
            # 1. Try to add a random point
            added = False
            for _ in range(100): # Try 100 random adds
                if time.time() > ls_deadline:
                    break
                new_pos = random.randint(0, num_points - 1)
                if new_pos in selected:
                    continue
                    
                new_dists = dist_matrix[new_pos, selected]
                if len(set(new_dists.tolist())) != len(new_dists):
                    continue
                    
                valid = True
                for i, sel_idx in enumerate(selected):
                    d = new_dists[i]
                    if dist_bitsets[i] & (1 << d):
                        valid = False
                        break
                        
                if not valid:
                    continue
                    
                # Add it
                new_bitset = 0
                for d in new_dists:
                    new_bitset |= (1 << d)
                for i, d in enumerate(new_dists):
                    dist_bitsets[i] |= (1 << d)
                selected.append(new_pos)
                dist_bitsets.append(new_bitset)
                
                added = True
                last_improvement = time.time()
                break
                
            if not added:
                # 2. Try a swap
                # Pick a random point to remove
                remove_idx = random.randint(0, len(selected) - 1)
                remove_pos = selected[remove_idx]
                
                # Temporarily remove
                temp_selected = selected[:]
                temp_dist_bitsets = dist_bitsets[:]
                temp_selected.pop(remove_idx)
                temp_dist_bitsets.pop(remove_idx)
                
                d_to_removed = dist_matrix[remove_pos, temp_selected]
                for i, d in enumerate(d_to_removed):
                    temp_dist_bitsets[i] &= ~(1 << d)
                    
                # Try to add a random point to the temp set
                swapped = False
                for _ in range(100):
                    if time.time() > ls_deadline:
                        break
                    new_pos = random.randint(0, num_points - 1)
                    if new_pos in temp_selected:
                        continue
                        
                    new_dists = dist_matrix[new_pos, temp_selected]
                    if len(set(new_dists.tolist())) != len(new_dists):
                        continue
                        
                    valid = True
                    for i, sel_idx in enumerate(temp_selected):
                        d = new_dists[i]
                        if temp_dist_bitsets[i] & (1 << d):
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
                    # Swap is valid
                    new_bitset = 0
                    for d in new_dists:
                        new_bitset |= (1 << d)
                    for i, d in enumerate(new_dists):
                        temp_dist_bitsets[i] |= (1 << d)
                    temp_selected.append(new_pos)
                    temp_dist_bitsets.append(new_bitset)
                    
                    # Accept swap
                    selected = temp_selected
                    dist_bitsets = temp_dist_bitsets
                    swapped = True
                    break
                    
                if not swapped:
                    # If we haven't improved (grown) in a long time, break to restart
                    if time.time() - last_improvement > 5.0:
                        break
                        
        # End of LS for this restart
        
        # Update best
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
            
    # Convert indices back to (x, y)
    result = [grid_points[i] for i in best_selected]
    return result

# EVOLVE-BLOCK-END
