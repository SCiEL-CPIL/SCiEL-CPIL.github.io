"""
Strategy: 3D Occupancy Grid with Z-Axis Binary Search and Multi-Strategy Ordering.

This approach fundamentally changes the spatial representation from a list of
placed boxes to a 3D numpy boolean grid. This allows:
1. O(1) overlap checking via slicing (vectorized).
2. A direct spatial lookup for the lowest empty slot by scanning z-levels 
   from bottom to top (or using binary search if the grid was pre-processed, 
   but here we scan because we need the exact lowest fit for each orientation).
   Actually, for "lowest empty slot", we can iterate z from 0 to D-bd and check 
   if the slice is empty. With numpy, this is fast.
3. Multiple ordering heuristics and local search on the ordering.

The key structural difference is the use of a dense grid for collision detection 
and placement search, rather than explicit pairwise box comparisons.
"""

import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 110.0

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]

    def pack(order):
        """
        Greedy packing using a 3D occupancy grid.
        Returns (placements_list, total_volume).
        """
        # Grid shape: (W, H, D)
        # Using uint8 for memory efficiency, 0/1 values
        grid = np.zeros((W, H, D), dtype=np.uint8)
        placed = []  # (idx, x, y, z, w, h, d)
        total_vol = 0

        for idx in order:
            w, h, d = boxes[idx]
            orients = orient_cache[idx]
            best_placement = None
            best_score = None

            # Try all orientations
            for (bw, bh, bd) in orients:
                if bw > W or bh > H or bd > D:
                    continue
                
                # Find the lowest (z, y, x) position that fits
                # We scan z from 0 to D-bd
                # For each z, scan y from 0 to H-bh
                # For each y, scan x from 0 to W-bw
                # This is O(W*H*D) in worst case, which is too slow if W,H,D are large.
                # However, for 60 boxes, the grid is mostly empty.
                # We can optimize by checking only "frontier" positions?
                # Or just iterate through z, and for each z, use numpy to find empty slots.
                
                found = False
                # To speed up, we can pre-compute for each z-level the empty x,y slots?
                # But the grid changes.
                
                # Let's try to find the first empty slot by scanning z, then y, then x.
                # We can use np.any on slices.
                for z in range(D - bd + 1):
                    # Check if any slice at this z has a free spot of size bw x bh
                    # We need to find if there exists x,y such that grid[x:x+bw, y:y+bh, z:z+bd] is all 0
                    # This is a 2D sliding window sum check.
                    # For simplicity and correctness, let's iterate x and y but break early.
                    # Optimization: If we just want ANY slot, we can check if the sum of the 
                    # sub-volumes is 0.
                    # Let's compute the "blocked" status for this z-level for the box size.
                    # We can use a cumulative sum trick or just simple iteration if W,H are small.
                    # Assuming W,H,D <= 1000, iteration might be slow.
                    # Let's use a helper that checks if a specific (x,y,z) is free.
                    
                    # To make it faster, we can scan x and y in a smart way?
                    # Or just accept the O(W*H) per z-level cost.
                    # If W=100, H=100, D=100, and we check 100 z-levels, that's 1,000,000 checks.
                    # Each check is a numpy slice sum, which is fast.
                    # 60 boxes * 6 orientations * 100 z-levels * (100*100 checks) = 360,000,000 operations?
                    # No, we break early when we find the first slot.
                    # The first slot is usually found quickly (low z, low y, low x).
                    
                    for y in range(H - bh + 1):
                        for x in range(W - bw + 1):
                            # Check if the box fits at (x, y, z)
                            # Slice: grid[x:x+bw, y:y+bh, z:z+bd]
                            if np.sum(grid[x:x+bw, y:y+bh, z:z+bd]) == 0:
                                # Found a slot
                                score = (z, y, x)
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_placement = (x, y, z, bw, bh, bd)
                                # Since we scan z, y, x in increasing order, 
                                # the first one found for this orientation is the best for it.
                                # But we need the best across ALL orientations.
                                # So we break out of x,y loops for this orientation.
                                found = True
                                break
                        if found:
                            break
                    if found:
                        # We found the best slot for this orientation.
                        # Compare with best across orientations.
                        # The score is (z, y, x).
                        # We continue to check other orientations to see if they offer a better (lower) score.
                        # Wait, if we find a slot at z=0, no other orientation can beat z=0.
                        # So we can break early if z==0?
                        # Yes, if z==0, it's the lowest possible z.
                        if z == 0:
                            # Can't get lower z. But maybe lower y or x with same z?
                            # We are scanning y and x in increasing order, so the first found at z=0 
                            # is the lowest y, and for that y, lowest x.
                            # So this is the global best possible slot.
                            # We can break out of the orientation loop too.
                            break
                # End z loop
                if best_placement is not None and best_placement[2] == 0:
                    # Found a slot at z=0, which is optimal.
                    break

            if best_placement is not None:
                bx, by, bz, bw, bh, bd = best_placement
                # Update grid
                grid[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                placed.append((idx, bx, by, bz, bw, bh, bd))
                total_vol += bw * bh * bd

        # Build result
        result = [None] * n
        for p in placed:
            idx, x, y, z, w, h, d = p
            result[idx] = (0, x, y, z, w, h, d)
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        return result, total_vol

    # Define several ordering heuristics
    indexed = list(range(n))
    orders = [
        sorted(indexed, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        sorted(indexed, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indexed, key=lambda i: (boxes[i][0]+boxes[i][1]+boxes[i][2]), reverse=True),
        sorted(indexed, key=lambda i: min(boxes[i]), reverse=True),
    ]

    best_result = None
    best_volume = -1
    best_order = None

    for order in orders:
        if time.time() > deadline:
            break
        res, vol = pack(order)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            best_order = order[:]

    # Local search: swap adjacent elements in the best order and re-pack
    if best_order is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_order) - 1):
                if time.time() > deadline:
                    break
                # Swap i and i+1
                best_order[i], best_order[i+1] = best_order[i+1], best_order[i]
                res, vol = pack(best_order)
                if vol > best_volume:
                    best_volume = vol
                    best_result = res
                    improved = True
                else:
                    # Revert
                    best_order[i], best_order[i+1] = best_order[i+1], best_order[i]

    # Randomized restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        rand_order = orders[0][:]
        random.shuffle(rand_order)
        res, vol = pack(rand_order)
        if vol > best_volume:
            best_volume = vol
            best_result = res

    if best_result is None:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    return best_result

# EVOLVE-BLOCK-END
