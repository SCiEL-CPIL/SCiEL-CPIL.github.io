import time
import random
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    # Safety margin: 150s limit, use 145s
    deadline = time.time() + 145.0
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # --- 3D Grid-Based Packing Engine ---
    
    # The grid represents occupied space.
    # grid[x, y, z] == 1 if occupied, 0 if free.
    # Dimensions: W, H, D
    
    def pack_with_order(order, grid, placed_info):
        """
        Greedily packs boxes in the given order using the grid.
        Returns (placed_indices, total_volume, final_grid_state, final_placed_info)
        Note: This function is destructive to the local copies of grid/placed_info.
        """
        placed_indices = []
        total_vol = 0
        
        # We need to efficiently find a valid position.
        # Strategy: 
        # 1. For each box, try orientations.
        # 2. For each orientation, scan potential positions.
        #    To speed up scanning, we can use the fact that we want "lowest, then leftmost, then backmost" (or similar heuristic).
        #    Instead of scanning all W*H*D positions, we can scan in a heuristic order.
        #    However, with NumPy, we can check a whole slice at once? 
        #    No, checking a specific box (w,h,d) at (x,y,z) requires checking grid[x:x+w, y:y+h, z:z+d].sum() == 0.
        #    Doing this for every (x,y,z) is O(W*H*D * w*h*d) which is too slow if done naively in Python loops.
        
        # Optimization: 
        # We can maintain a "free space" structure? 
        # Or, we can just iterate z from 0 to D-h, y from 0 to H-h, x from 0 to W-w.
        # For each (z,y,x), check the sum. 
        # To make this fast, we can precompute a 3D prefix sum? 
        # But the grid changes as we place boxes. Updating a 3D prefix sum is O(W*H*D) per update, which is too slow.
        
        # Alternative: Use the grid directly but optimize the search.
        # Since N=60 and container size is likely moderate (e.g., < 100x100x100), 
        # W*H*D might be around 1e6. 
        # Checking 1e6 positions * 60 boxes = 60e6 checks. 
        # Each check is a numpy sum of a subarray. 
        # This might be too slow in Python if we loop over 1e6 positions in Python.
        
        # Better Strategy for Grid:
        # Instead of scanning all positions, we can use a "skyline" on the grid?
        # Or, we can restrict the search space.
        # Let's try a different approach: 
        # For a given box (w,h,d), we want to find if there is ANY free spot.
        # We can iterate z, then y, then x.
        # To speed up the inner loop, we can check if the entire plane at height z has enough free space?
        
        # Actually, let's look at the constraints. 
        # If the container is small (e.g., 20x20x20 = 8000 cells), scanning is fine.
        # If it's large (100x100x100 = 1,000,000 cells), scanning all positions for 60 boxes is 60,000,000 numpy slice sums. 
        # Each numpy sum is fast in C, but the overhead of the Python loop over 1M positions is high.
        
        # Let's optimize the search by only checking "candidate" positions derived from the grid structure?
        # No, the grid IS the structure.
        
        # Hybrid Idea:
        # Use the grid for O(1) overlap check.
        # But how to find the position?
        # We can use a randomized search or a heuristic scan.
        # Given the time limit, let's try to scan in a specific order (z, y, x) and stop at the first valid spot.
        # This is "First Fit" on the grid.
        
        # To make the scan faster, we can precompute for each z-level, the 2D free space?
        # No, let's just try the direct scan with NumPy and see if it's fast enough.
        # If W*H*D is ~1000, it's instant. If ~100,000, it's ~100k * 60 = 6M operations. 
        # Numpy sum on a small slice is very fast. The loop overhead is the bottleneck.
        
        # Let's implement a vectorized check for a specific orientation?
        # We can slide a window over the grid? 
        # np.lib.stride_tricks.sliding_window_view can create a view of all possible placements.
        # But we need to check if the sum is 0.
        
        # Let's use sliding_window_view for the inner loop (x,y) for a fixed z?
        # Or for the whole 3D?
        
        # Let's try a simpler approach first: 
        # Iterate z from 0 to D-d
        #   Iterate y from 0 to H-h
        #     Iterate x from 0 to W-w
        #       if grid[x:x+w, y:y+h, z:z+d].sum() == 0:
        #           place it
        #           return
        #
        # To speed this up, we can break early if we find a spot.
        # We can also skip z-levels that are completely full?
        
        # Let's implement this and add a timeout check inside the loop if it's too slow?
        # No, we need to be fast.
        
        # Optimization: 
        # Precompute a "free" array? No, grid is the free array (0 is free).
        
        # Let's use a trick: 
        # For a fixed z and y, we can check all x positions at once using convolution?
        # Or just a loop over x.
        
        # Let's assume the container is not huge (e.g., < 50x50x50).
        # If it is huge, this might time out. 
        # The problem statement says "5 generated 60-box instances". 
        # Usually, these are sized so that packing is non-trivial but solvable.
        
        # Let's try to optimize the placement function.
        
        def find_position(w, h, d):
            # Try to find a position for a box of size w, h, d
            # We scan in order of "lowest z, then lowest y, then lowest x"
            # This tends to pack tightly.
            
            # We can use sliding_window_view to get all possible positions for a fixed z?
            # Let's try to vectorize the x-y scan for a fixed z.
            
            for z in range(D - d + 1):
                # Check if this z-layer has any free space for the footprint w x h?
                # We can compute the 2D sum of the grid slice at height z..z+d for all x,y
                # slice = grid[:, :, z:z+d]
                # sums = np.sum(slice, axis=2) # Shape (W, H)
                # We need sums[x, y] == 0 for all (x,y) in the window (w, h)
                
                # Use sliding_window_view on the 2D sums?
                # Or just check the 2D slice directly.
                
                # Let's compute the 2D occupancy map for this z-range
                # occ_2d = np.sum(grid[:, :, z:z+d], axis=2)
                # Now we need to find if there is a w x h window in occ_2d with sum 0.
                
                # This is a 2D problem now.
                # We can use sliding_window_view on occ_2d.
                
                if D > 0:
                    occ_2d = np.sum(grid[:, :, z:z+d], axis=2)
                    # If the total sum of occ_2d is 0, we can place anywhere.
                    # If not, we need to find a free window.
                    
                    # Use sliding window view for efficiency
                    # If W or H is small, this is fine.
                    # If W or H is large, this might be memory intensive? 
                    # sliding_window_view doesn't copy data, it creates a view.
                    
                    # But we need to check if ANY window is 0.
                    # We can compute the sum of all windows?
                    # Or just check if min(window_sums) == 0.
                    
                    # Let's compute the sums of all w x h windows in occ_2d.
                    # This can be done with a 2D convolution or cumulative sum.
                    # Cumulative sum is faster.
                    
                    # Prefix sum of occ_2d
                    # ps = np.cumsum(np.cumsum(occ_2d, axis=0), axis=1)
                    # Then sum of window at (x,y) is ps[x+w, y+h] - ps[x, y+h] - ps[x+w, y] + ps[x, y]
                    # (with appropriate bounds)
                    
                    # Let's just use a simple loop if the size is small, or a vectorized check.
                    # Given N=60, and potential restarts, we need this to be fast.
                    
                    # Let's try the vectorized check with prefix sums.
                    if W >= w and H >= h:
                        # Add padding to handle bounds easily
                        # Actually, let's just use np.lib.stride_tricks.sliding_window_view
                        # It returns a view of shape (W-w+1, H-h+1, w, h)
                        # We can sum the last two axes.
                        
                        # This might be slow if W*H is large.
                        # Let's fallback to a simpler check if W*H > 10000?
                        
                        # For now, let's try the sliding window view.
                        try:
                            windows = np.lib.stride_tricks.sliding_window_view(occ_2d, (w, h))
                            window_sums = np.sum(windows, axis=(-2, -1))
                            # Find the first index where sum is 0
                            # We want the "lowest" x, y.
                            # So we can flatten and find the first 0?
                            # But we want the one with smallest x, then smallest y.
                            # The view is ordered by x, then y.
                            # So the first 0 in the flattened array is the best.
                            
                            flat_sums = window_sums.flatten()
                            zero_indices = np.where(flat_sums == 0)[0]
                            if len(zero_indices) > 0:
                                idx = zero_indices[0]
                                # Convert flat index to (x, y)
                                # The shape of window_sums is (W-w+1, H-h+1)
                                num_y = H - h + 1
                                x = idx // num_y
                                y = idx % num_y
                                return x, y, z
                        except Exception:
                            pass
            return None

        for idx in order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            placed = False
            
            # Try orientations
            # Sort orientations by some heuristic? e.g., volume, or just random?
            # Let's try all, pick the first valid one.
            # To improve packing, maybe try orientations that are "flatter" first?
            # Or just iterate.
            
            for (bw, bh, bd) in orient_cache[idx]:
                pos = find_position(bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    # Place the box
                    grid[x:x+bw, y:y+bh, z:z+bd] = 1
                    placed_info[idx] = (x, y, z, bw, bh, bd)
                    placed_indices.append(idx)
                    total_vol += bw * bh * bd
                    placed = True
                    break
            
            if not placed:
                # Box couldn't be placed
                # In this greedy pass, we just skip it.
                # But for the search, we might want to know which ones were skipped.
                pass
        
        return placed_indices, total_vol, grid, placed_info

    # --- Search Strategy ---
    
    # We will use Simulated Annealing on the permutation of boxes.
    # State: Permutation of box indices.
    # Objective: Maximize total volume packed.
    
    indexed = list(range(n))
    
    # Initial orders
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    
    best_vol = 0
    best_placed_info = {}
    best_order = order_vol[:]
    
    # Initialize grid
    grid = np.zeros((W, H, D), dtype=np.int8)
    placed_info = {}
    
    # Run initial greedy
    placed_indices, vol, grid, placed_info = pack_with_order(order_vol, grid, placed_info)
    if vol > best_vol:
        best_vol = vol
        best_placed_info = placed_info.copy()
        best_order = order_vol[:]
    
    # Simulated Annealing
    current_order = best_order[:]
    current_vol = best_vol
    current_grid = np.zeros((W, H, D), dtype=np.int8)
    current_placed_info = {}
    
    # Re-run to get state
    _, current_vol, current_grid, current_placed_info = pack_with_order(current_order, current_grid, current_placed_info)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Perturb: Swap two random boxes
        i, j = random.sample(range(n), 2)
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Evaluate
        new_grid = np.zeros((W, H, D), dtype=np.int8)
        new_placed_info = {}
        _, new_vol, new_grid, new_placed_info = pack_with_order(new_order, new_grid, new_placed_info)
        
        delta = new_vol - current_vol
        
        if delta > 0 or random.random() < np.exp(delta / (T * max(1, best_vol))):
            current_order = new_order
            current_vol = new_vol
            current_grid = new_grid
            current_placed_info = new_placed_info
            no_improve = 0
            
            if current_vol > best_vol:
                best_vol = current_vol
                best_placed_info = current_placed_info.copy()
                best_order = current_order[:]
                
        else:
            no_improve += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart with a random order if stuck?
            # For now, just keep going.

    # Finalize result
    result = [None] * n
    for idx, (x, y, z, w, h, d) in best_placed_info.items():
        result[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result

# EVOLVE-BLOCK-END
