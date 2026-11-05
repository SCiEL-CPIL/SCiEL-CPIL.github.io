"""
3D Bin Packing: Grid-based Occupancy with BLF Heuristic and Local Search.
Strategy:
1. Represent each bin's state as a 3D NumPy boolean array for O(1) overlap checks.
2. Use a "Bottom-Left-Far" (BLF) heuristic for placement: minimize x, then y, then z.
   This tends to pack boxes tightly against the origin, reducing fragmentation.
3. Implement a local search phase: after the initial greedy pack, attempt to
   improve the solution by swapping box positions or re-sequencing boxes within
   a time budget.
4. Multi-start: run the greedy pack with different box orderings (volume, dims, random).
"""
import time
import random
import math
import numpy as np
from scipy import ndimage

DEADLINE = time.time() + 145.0  # Safety margin for 150s limit

def _fits_grid(grid, x, y, z, w, h, d):
    """Check if a box fits at (x, y, z) in the grid without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > grid.shape[0] or y + h > grid.shape[1] or z + d > grid.shape[2]:
        return False
    # Check if the volume is free (0)
    subgrid = grid[x:x+w, y:y+h, z:z+d]
    return not np.any(subgrid)

def _place_grid(grid, x, y, z, w, h, d):
    """Place a box in the grid."""
    grid[x:x+w, y:y+h, z:z+d] = 1

def _remove_grid(grid, x, y, z, w, h, d):
    """Remove a box from the grid."""
    grid[x:x+w, y:y+h, z:z+d] = 0

def _find_best_position(grid, w, h, d):
    """
    Find the best position (x, y, z) for a box of size (w, h, d) in the grid.
    Uses a heuristic: minimize x, then y, then z (Bottom-Left-Far).
    We scan candidate positions efficiently.
    """
    best_pos = None
    best_score = float('inf')
    
    # To speed up, we can limit the search space or use a smarter scan.
    # For small grids (bin dimensions), a full scan is feasible if optimized.
    # However, for 40 boxes, bin dims might be moderate.
    # Let's scan all valid (x, y, z) but prune early.
    
    # Optimization: Only check positions where the box *could* fit.
    # We can use a "skyline" or just iterate.
    # Given the time budget, let's iterate but break early if we find a "good" spot.
    # Actually, for BLF, we want the minimum x, then min y, then min z.
    
    # Efficient scan:
    for x in range(grid.shape[0] - w + 1):
        for y in range(grid.shape[1] - h + 1):
            for z in range(grid.shape[2] - d + 1):
                if _fits_grid(grid, x, y, z, w, h, d):
                    # Score: prioritize lower x, then lower y, then lower z
                    # This creates a tight pack against (0,0,0)
                    score = x * 10000 + y * 100 + z
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        # Optional: break early if x is 0 and y is 0 and z is 0?
                        # No, we want the *best* BLF, which is unique if we define it strictly.
                        # But finding the absolute minimum x,y,z requires scanning all.
                        # We can break inner loops if we found x=0, y=0, z=0.
                        if x == 0 and y == 0 and z == 0:
                            return best_pos
    return best_pos

def _pack_greedy(boxes, container, order):
    """
    Pack boxes using the grid-based BLF heuristic.
    Returns a list of bins, each bin is a list of (idx, x, y, z, w, h, d).
    """
    W, H, D = container
    bins = []
    bin_grids = []
    bin_vols = []
    
    for idx in order:
        w, h, d = boxes[idx]
        box_vol = w * h * d
        placed = False
        
        # Try to place in existing bins
        for bi in range(len(bins)):
            grid = bin_grids[bi]
            pos = _find_best_position(grid, w, h, d)
            if pos is not None:
                x, y, z = pos
                _place_grid(grid, x, y, z, w, h, d)
                bins[bi].append((idx, x, y, z, w, h, d))
                bin_vols[bi] += box_vol
                placed = True
                break
        
        if not placed:
            # Open new bin
            grid = np.zeros((W, H, D), dtype=np.uint8)
            # Place at origin
            _place_grid(grid, 0, 0, 0, w, h, d)
            bins.append([(idx, 0, 0, 0, w, h, d)])
            bin_grids.append(grid)
            bin_vols.append(box_vol)
            
    return bins

def _score(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    if B == 0:
        return 0.0
    last_bin = bins[-1]
    if not last_bin:
        return B
    last_vol = sum(w * h * d for (_, _, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill_ratio)

def _local_search(bins, container, boxes, time_limit=10.0):
    """
    Simple local search: try to move boxes from the last bin to previous bins
    or swap boxes to improve fill of the last bin or reduce bin count.
    This is a heuristic improvement step.
    """
    start_time = time.time()
    W, H, D = container
    C = W * H * D
    
    # Reconstruct grids from bins
    bin_grids = []
    for bin_content in bins:
        grid = np.zeros((W, H, D), dtype=np.uint8)
        for (idx, x, y, z, w, h, d) in bin_content:
            _place_grid(grid, x, y, z, w, h, d)
        bin_grids.append(grid)
        
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        B = len(bins)
        if B <= 1:
            break
            
        # Try to move a box from the last bin to a previous bin
        last_bin_idx = B - 1
        last_bin_content = bins[last_bin_idx][:] # Copy
        
        # Sort last bin content by volume desc to try moving large boxes first
        last_bin_content.sort(key=lambda item: item[4]*item[5]*item[6], reverse=True)
        
        for item in last_bin_content:
            idx, x, y, z, w, h, d = item
            # Remove from last bin
            _remove_grid(bin_grids[last_bin_idx], x, y, z, w, h, d)
            bins[last_bin_idx].remove(item)
            
            # Try to place in previous bins
            placed = False
            for bi in range(last_bin_idx):
                pos = _find_best_position(bin_grids[bi], w, h, d)
                if pos is not None:
                    px, py, pz = pos
                    _place_grid(bin_grids[bi], px, py, pz, w, h, d)
                    bins[bi].append((idx, px, py, pz, w, h, d))
                    placed = True
                    improved = True
                    break
            
            if not placed:
                # Put back in last bin
                _place_grid(bin_grids[last_bin_idx], x, y, z, w, h, d)
                bins[last_bin_idx].append(item)
                
            if time.time() - start_time > time_limit:
                break

    # Remove empty bins at the end
    final_bins = []
    for bi, bin_content in enumerate(bins):
        if bin_content:
            final_bins.append(bin_content)
            
    return final_bins

def _generate_orderings(boxes, n, rng):
    """Generate diverse orderings of box indices."""
    idx = list(range(n))
    
    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Sum of Dimensions Descending
    yield sorted(idx, key=lambda i: sum(boxes[i]), reverse=True)
    
    # 4. Aspect Ratio Descending
    yield sorted(idx, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True)
    
    # 5-15. Random Shuffles
    for _ in range(10):
        s = idx[:]
        rng.shuffle(s)
        yield s

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    best_bins = None
    best_score = float('inf')
    t_start = time.time()
    rng = random.Random(42)
    
    # Phase 1: Multi-start Greedy Packing
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE - 20: # Leave 20s for local search and safety
            break
            
        bins = _pack_greedy(boxes, container, order)
        s = _score(bins, container)
        
        if s < best_score:
            best_score = s
            best_bins = bins
            
    if best_bins is None:
        # Fallback
        best_bins = [[(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])] for i in range(n)]
        best_score = _score(best_bins, container)
        
    # Phase 2: Local Search on the best solution found
    if time.time() < DEADLINE - 5:
        ls_time_limit = min(10.0, DEADLINE - 5 - time.time())
        improved_bins = _local_search(best_bins, container, boxes, time_limit=ls_time_limit)
        s_ls = _score(improved_bins, container)
        
        if s_ls < best_score:
            best_bins = improved_bins
            best_score = s_ls
            
    # Build the result in original box order
    placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            placement_map[idx] = (bi, x, y, z, w, h, d)
            
    result = []
    for i in range(n):
        if i in placement_map:
            result.append(placement_map[i])
        else:
            # Should not happen if valid, but safety
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result

