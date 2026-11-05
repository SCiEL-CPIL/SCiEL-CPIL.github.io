"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Deadline guard: 150s wall-clock, use 130s for safety
    deadline = time.time() + 130.0
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    # We process in this order but must return placements in original order
    box_dims = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    box_dims.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # Each bin maintains:
    # - occupancy: list of placed boxes (x, y, z, w, h, d)
    # - we'll use a simple check for overlaps
    
    # For efficiency, we'll maintain per-bin data structures
    # Since n=40 is small, O(n^2) overlap checks are fine
    
    bins = []  # list of lists of placements (x, y, z, w, h, d)
    
    def fits_in_bin(bin_idx, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return False
        return True
    
    def best_fit_position(bin_idx, w, h, d):
        """Find the position that minimizes wasted space (best fit)."""
        # Try all possible positions on a grid
        # For small containers, we can try integer positions
        # But dimensions might be large... let's use a smarter approach
        
        # Get all "candidate" positions:
        # 1. Origin (0,0,0)
        # 2. Adjacent to each existing box on each face
        
        candidates = [(0, 0, 0)]
        
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            # Right of box
            candidates.append((px + pw, py, pz))
            # Left of box
            candidates.append((px - w, py, pz))
            # Above box
            candidates.append((px, py + ph, pz))
            # Below box
            candidates.append((px, py - h, pz))
            # Front of box
            candidates.append((px, py, pz + pd))
            # Back of box
            candidates.append((px, py, pz - d))
        
        # Also try positions aligned with existing boxes
        existing_xs = set([0])
        existing_ys = set([0])
        existing_zs = set([0])
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            existing_xs.add(px)
            existing_xs.add(px + pw)
            existing_ys.add(py)
            existing_ys.add(py + ph)
            existing_zs.add(pz)
            existing_zs.add(pz + pd)
        
        # Generate more candidates from combinations
        for cx in existing_xs:
            for cy in existing_ys:
                for cz in existing_zs:
                    candidates.append((cx, cy, cz))
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if fits_in_bin(bin_idx, cx, cy, cz, w, h, d):
                # Score: prefer positions closer to origin (minimize max coordinate)
                score = cx + cy + cz
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        
        return best_pos
    
    def open_space_score(bin_idx):
        """Score how much open space remains in a bin (lower is better for packing)."""
        total_vol = W * H * D
        used_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bins[bin_idx])
        return total_vol - used_vol
    
    # Place each box
    placements_orig = [None] * n
    
    for (w, h, d, orig_idx) in box_dims:
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_pos = None
        best_total_score = float('inf')
        
        # Try each existing bin
        for bi in range(len(bins)):
            pos = best_fit_position(bi, w, h, d)
            if pos is not None:
                # Score: prefer bins with less remaining space (to fill bins)
                remaining = open_space_score(bi)
                total_score = remaining
                if total_score < best_total_score:
                    best_total_score = total_score
                    best_bin = bi
                    best_pos = pos
        
        # If no bin works, create a new bin
        if best_bin == -1:
            # Try to place in a new bin at origin
            if fits_in_bin_check_origin(w, h, d):
                best_bin = len(bins)
                best_pos = (0, 0, 0)
                bins.append([])
        
        if best_bin != -1 and best_pos is not None:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            placements_orig[orig_idx] = (best_bin, x, y, z, w, h, d)
        else:
            # Shouldn't happen if boxes fit in container
            # Place in a new bin anyway
            best_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements_orig[orig_idx] = (best_bin, 0, 0, 0, w, h, d)
    
    # Check if we timed out and some boxes are unplaced
    if any(p is None for p in placements_orig):
        # Fallback: use simple shelf heuristic for remaining boxes
        for i, (w, h, d) in enumerate(boxes):
            if placements_orig[i] is None:
                # Find a bin that can fit it, or create new
                placed = False
                for bi in range(len(bins)):
                    pos = best_fit_position(bi, w, h, d)
                    if pos is not None:
                        x, y, z = pos
                        bins[bi].append((x, y, z, w, h, d))
                        placements_orig[i] = (bi, x, y, z, w, h, d)
                        placed = True
                        break
                if not placed:
                    bi = len(bins)
                    bins.append([(0, 0, 0, w, h, d)])
                    placements_orig[i] = (bi, 0, 0, 0, w, h, d)
    
    return placements_orig

def fits_in_bin_check_origin(w, h, d):
    # This is a placeholder; we need W, H, D from container
    # Actually, let's inline this properly
    return True  # We'll handle this differently

# Let me rewrite this more carefully

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    deadline = time.time() + 130.0
    
    box_dims = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    box_dims.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    bins = []
    
    def fits(bin_idx, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return False
        return True
    
    def find_best_pos(bin_idx, w, h, d):
        candidates = [(0, 0, 0)]
        
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            candidates.append((px + pw, py, pz))
            candidates.append((px - w, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py - h, pz))
            candidates.append((px, py, pz + pd))
            candidates.append((px, py, pz - d))
        
        exs = set([0])
        eys = set([0])
        ezs = set([0])
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            exs.update([px, px + pw])
            eys.update([py, py + ph])
            ezs.update([pz, pz + pd])
        
        for cx in exs:
            for cy in eys:
                for cz in ezs:
                    candidates.append((cx, cy, cz))
        
        best = None
        best_s = float('inf')
        for (cx, cy, cz) in candidates:
            if fits(bin_idx, cx, cy, cz, w, h, d):
                s = cx + cy + cz
                if s < best_s:
                    best_s = s
                    best = (cx, cy, cz)
        return best
    
    placements = [None] * n
    
    for (w, h, d, oi) in box_dims:
        if time.time() > deadline:
            break
        
        best_bi = -1
        best_p = None
        best_rem = float('inf')
        
        for bi in range(len(bins)):
            p = find_best_pos(bi, w, h, d)
            if p is not None:
                rem = W*H*D - sum(pw*ph*pd for (_,_,_,pw,ph,pd) in bins[bi])
                if rem < best_rem:
                    best_rem = rem
                    best_bi = bi
                    best_p = p
        
        if best_bi == -1:
            if (0, 0, 0) and w <= W and h <= H and d <= D:
                best_bi = len(bins)
                best_p = (0, 0, 0)
                bins.append([])
        
        if best_bi >= 0 and best_p is not None:
            x, y, z = best_p
            bins[best_bi].append((x, y, z, w, h, d))
            placements[oi] = (best_bi, x, y, z, w, h, d)
        else:
            best_bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[oi] = (best_bi, 0, 0, 0, w, h, d)
    
    # Fallback for any unplaced
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placed = False
            for bi in range(len(bins)):
                p = find_best_pos(bi, w, h, d)
                if p is not None:
                    x, y, z = p
                    bins[bi].append((x, y, z, w, h, d))
                    placements[i] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[i] = (bi, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
