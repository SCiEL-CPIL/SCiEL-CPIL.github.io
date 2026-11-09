"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq

def pack(boxes, container):
    """
    3D Bin Packing using a best-fit decreasing strategy with a bin-tree structure.
    
    Strategy:
    1. Sort boxes by volume (descending) to place large items first.
    2. Maintain a list of bins. Each bin tracks its occupied space using a simple
       interval tree or a list of placed boxes to check for overlap.
       For efficiency and simplicity with 40 boxes, we can use a grid or a list of
       placed boxes per bin and check for overlap in O(N^2) per placement attempt,
       which is fine for small N.
       
    However, a more robust approach for "best fit" is to maintain a list of
    "empty spaces" (cuboids) in each bin. When a box is placed, it splits the
    empty space into up to 6 new empty spaces. We pick the bin and empty space
    that minimizes the remaining space (or maximizes the fit).
    
    Given the small number of boxes (40), a simpler heuristic might work well:
    - Sort boxes by (w*h*d) descending, or by max dimension descending.
    - For each box, try to place it in the current bin (or next available) using
      a shelf-like or bottom-left-front heuristic.
    
    Let's implement a "Best Fit Decreasing" with a simplified empty-space tracking.
    Since the container is fixed and boxes are small, we can maintain a list of
    free cuboids per bin.
    
    Algorithm:
    1. Sort boxes by volume descending.
    2. Initialize a list of bins. Bin 0 is empty.
    3. For each box:
       a. Try to place it in existing bins. For each bin, find the best free space
          (smallest remaining volume after placement, or closest to bottom-left-front).
       b. If no existing bin can fit the box, open a new bin.
    
    To find the best free space, we maintain a list of free cuboids per bin.
    A free cuboid is defined by (x, y, z, w, h, d).
    When a box of size (w, h, d) is placed at (x, y, z) within a free cuboid
    (fx, fy, fz, fw, fh, fd), the remaining free space is split into up to 6 parts:
    - Left: (fx, fy, fz, x - fx, fh, fd) if x > fx
    - Right: (x + w, fy, fz, fx + fw - (x + w), fh, fd) if x + w < fx + fw
    - Bottom: (x, fy, z, w, y - fy, d) if y > fy
    - Top: (x, y + h, z, w, fy + fh - (y + h), d) if y + h < fy + fh
    - Front: (x, y, fz, w, h, z - fz) if z > fz
    - Back: (x, y, z + d, w, h, fz + fd - (z + d)) if z + d < fz + fd
    
    We choose the placement that minimizes the sum of the volumes of the new free
    cuboids, or simply the first fit in a specific order (bottom-left-front).
    
    For simplicity and speed, let's use a "First Fit Decreasing" with a
    bottom-left-front placement within each bin. This is often competitive.
    
    Actually, let's try a hybrid: sort by volume descending, and for each box,
    try to place it in the bin where it fits with the least waste. If no bin fits,
    open a new one.
    
    To determine if a box fits in a bin, we check against the list of free cuboids.
    We place it in the free cuboid that results in the smallest total free volume
    after placement.
    
    Let's implement this.
    """
    W, H, D = container
    # Sort boxes by volume descending
    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # bins[i] is a list of free cuboids: (x, y, z, w, h, d)
    bins = []
    # placements[i] = (bin_idx, x, y, z, w, h, d)
    placements = [None] * len(boxes)
    
    def add_bin():
        bins.append([(0, 0, 0, W, H, D)])
        return len(bins) - 1
    
    def try_place_in_bin(bin_idx, w, h, d):
        """
        Try to place a box of size (w, h, d) in bin bin_idx.
        Returns (x, y, z) if placed, else None.
        Also updates the free cuboids in bins[bin_idx].
        """
        free_cuboids = bins[bin_idx]
        best_pos = None
        best_waste = float('inf')
        
        for i, (fx, fy, fz, fw, fh, fd) in enumerate(free_cuboids):
            if w <= fw and h <= fh and d <= fd:
                # Try placing at bottom-left-front of this cuboid
                x, y, z = fx, fy, fz
                # Calculate new free cuboids
                new_cuboids = []
                # Left
                if x > fx:
                    new_cuboids.append((fx, fy, fz, x - fx, fh, fd))
                # Right
                if x + w < fx + fw:
                    new_cuboids.append((x + w, fy, fz, fx + fw - (x + w), fh, fd))
                # Bottom
                if y > fy:
                    new_cuboids.append((x, fy, z, w, y - fy, d))
                # Top
                if y + h < fy + fh:
                    new_cuboids.append((x, y + h, z, w, fy + fh - (y + h), d))
                # Front
                if z > fz:
                    new_cuboids.append((x, y, fz, w, h, z - fz))
                # Back
                if z + d < fz + fd:
                    new_cuboids.append((x, y, z + d, w, h, fz + fd - (z + d)))
                
                # Calculate total volume of new free cuboids
                waste = sum(cw * ch * cd for _, _, _, cw, ch, cd in new_cuboids)
                # Also consider the position to prefer bottom-left-front
                pos_score = x * 1000 + y * 100 + z
                total_score = waste + pos_score * 0.01
                
                if total_score < best_waste:
                    best_waste = total_score
                    best_pos = (x, y, z)
                    # Store the new cuboids for this option
                    # We need to reconstruct the list later
                    # For now, just store the index and the new list
                    # To avoid storing large lists, we can just store the index
                    # and recompute if needed, or store a reference.
                    # Let's store the index and the new list in a temporary structure
                    # But since we only need the best one, we can store the details
                    # We'll store the index i and the new_cuboids
                    # Actually, we can just update the list if we decide to place it.
                    # But we need to find the best first.
                    # Let's store the best index and the new cuboids
                    # We'll use a separate variable to store the best new cuboids
                    # This is inefficient for large lists, but N is small.
                    # Let's store the best index and the new cuboids in a list
                    if not hasattr(try_place_in_bin, '_temp'):
                        try_place_in_bin._temp = []
                    # We can't easily store multiple options in a static variable
                    # Let's just store the best index and the new cuboids in a local variable
                    # and return them if it's the best
                    # We'll change the structure to return (x, y, z, new_cuboids)
        
        # Re-implementing to return the best option
        best_pos = None
        best_waste = float('inf')
        best_new_cuboids = None
        best_idx = -1
        
        for i, (fx, fy, fz, fw, fh, fd) in enumerate(free_cuboids):
            if w <= fw and h <= fh and d <= fd:
                x, y, z = fx, fy, fz
                new_cuboids = []
                if x > fx:
                    new_cuboids.append((fx, fy, fz, x - fx, fh, fd))
                if x + w < fx + fw:
                    new_cuboids.append((x + w, fy, fz, fx + fw - (x + w), fh, fd))
                if y > fy:
                    new_cuboids.append((x, fy, z, w, y - fy, d))
                if y + h < fy + fh:
                    new_cuboids.append((x, y + h, z, w, fy + fh - (y + h), d))
                if z > fz:
                    new_cuboids.append((x, y, fz, w, h, z - fz))
                if z + d < fz + fd:
                    new_cuboids.append((x, y, z + d, w, h, fz + fd - (z + d)))
                
                waste = sum(cw * ch * cd for _, _, _, cw, ch, cd in new_cuboids)
                pos_score = x * 1000 + y * 100 + z
                total_score = waste + pos_score * 0.01
                
                if total_score < best_waste:
                    best_waste = total_score
                    best_pos = (x, y, z)
                    best_new_cuboids = new_cuboids
                    best_idx = i
        
        if best_pos is not None:
            # Update the free cuboids
            # Remove the old cuboid and add the new ones
            free_cuboids[best_idx] = best_new_cuboids[0] if len(best_new_cuboids) == 1 else best_new_cuboids[0]
            # This is tricky because we need to replace one element with a list
            # Let's rebuild the list
            new_free_cuboids = []
            for j, cuboid in enumerate(free_cuboids):
                if j == best_idx:
                    new_free_cuboids.extend(best_new_cuboids)
                else:
                    new_free_cuboids.append(cuboid)
            bins[bin_idx] = new_free_cuboids
            return best_pos
        return None
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        placed = False
        for bin_idx in range(len(bins)):
            pos = try_place_in_bin(bin_idx, w, h, d)
            if pos is not None:
                x, y, z = pos
                placements[idx] = (bin_idx, x, y, z, w, h, d)
                placed = True
                break
        if not placed:
            bin_idx = add_bin()
            placements[idx] = (bin_idx, 0, 0, 0, w, h, d)
            # Update the free cuboids for the new bin
            new_cuboids = []
            if w < W:
                new_cuboids.append((w, 0, 0, W - w, H, D))
            if h < H:
                new_cuboids.append((0, h, 0, w, H - h, D))
            if d < D:
                new_cuboids.append((0, 0, d, w, h, D - d))
            bins[bin_idx] = new_cuboids
    
    return placements
# EVOLVE-BLOCK-END
