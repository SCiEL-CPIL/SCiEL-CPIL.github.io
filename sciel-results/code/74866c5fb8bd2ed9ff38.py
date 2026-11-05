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
    
    deadline = time.time() + 125.0
    
    container_vol = W * H * D
    
    # Generate multiple sort strategies for diversity
    strategies = [
        lambda x: x[0]*x[1]*x[2],  # volume
        lambda x: max(x[0], x[1], x[2]),  # max dimension
        lambda x: x[0]*x[1]*x[2] + 0.1*max(x[0], x[1], x[2]),  # combined
    ]
    
    def solve_with_strategy(sort_key_fn):
        """Solve using a specific sorting strategy."""
        box_dims = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
        box_dims.sort(key=lambda x: sort_key_fn((x[0], x[1], x[2])), reverse=True)
        
        bins = []  # list of lists of placements (x, y, z, w, h, d)
        bin_volumes = []  # track used volume per bin
        
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
            """Find best position using candidate generation."""
            candidates = [(0, 0, 0)]
            
            for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
                # Six face-adjacent positions
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
                # Also try positions that align with box edges
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))
            
            # Generate candidates from all unique coordinate combinations
            exs = {0}
            eys = {0}
            ezs = {0}
            for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
                exs.update([px, px + pw])
                eys.update([py, py + ph])
                ezs.update([pz, pz + pd])
            
            # Filter to valid ranges
            exs = {x for x in exs if 0 <= x <= W}
            eys = {y for y in eys if 0 <= y <= H}
            ezs = {z for z in ezs if 0 <= z <= D}
            
            # Add combinations (limit to avoid explosion)
            if len(exs) * len(eys) * len(ezs) < 1000:
                for cx in exs:
                    for cy in eys:
                        for cz in ezs:
                            candidates.append((cx, cy, cz))
            
            best = None
            best_score = float('inf')
            
            for (cx, cy, cz) in candidates:
                if fits(bin_idx, cx, cy, cz, w, h, d):
                    # Score: prefer positions that minimize wasted space
                    # Use a combination of proximity to origin and tightness
                    score = cx * 1.0 + cy * 1.0 + cz * 1.0
                    if score < best_score:
                        best_score = score
                        best = (cx, cy, cz)
            
            return best
        
        placements = [None] * n
        
        for (w, h, d, oi) in box_dims:
            if time.time() > deadline:
                break
            
            best_bi = -1
            best_p = None
            best_rem = float('inf')
            
            # Try each existing bin
            for bi in range(len(bins)):
                p = find_best_pos(bi, w, h, d)
                if p is not None:
                    rem = container_vol - bin_volumes[bi]
                    if rem < best_rem:
                        best_rem = rem
                        best_bi = bi
                        best_p = p
            
            # If no existing bin works, create new bin
            if best_bi == -1:
                if w <= W and h <= H and d <= D:
                    best_bi = len(bins)
                    best_p = (0, 0, 0)
                    bins.append([])
                    bin_volumes.append(0)
            
            if best_bi >= 0 and best_p is not None:
                x, y, z = best_p
                vol = w * h * d
                bins[best_bi].append((x, y, z, w, h, d))
                bin_volumes[best_bi] += vol
                placements[oi] = (best_bi, x, y, z, w, h, d)
            else:
                # Fallback: force into new bin
                best_bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                placements[oi] = (best_bi, 0, 0, 0, w, h, d)
        
        # Fallback for any unplaced boxes
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placed = False
                for bi in range(len(bins)):
                    p = find_best_pos(bi, w, h, d)
                    if p is not None:
                        x, y, z = p
                        vol = w * h * d
                        bins[bi].append((x, y, z, w, h, d))
                        bin_volumes[bi] += vol
                        placements[i] = (bi, x, y, z, w, h, d)
                        placed = True
                        break
                if not placed:
                    bi = len(bins)
                    bins.append([(0, 0, 0, w, h, d)])
                    bin_volumes.append(w * h * d)
                    placements[i] = (bi, 0, 0, 0, w, h, d)
        
        return placements
    
    # Try multiple strategies and pick the best
    best_placements = None
    best_score = float('inf')
    
    for i, strategy in enumerate(strategies):
        if time.time() > deadline:
            break
        
        placements = solve_with_strategy(strategy)
        
        # Calculate score
        if placements is not None and all(p is not None for p in placements):
            # Count occupied bins
            occupied_bins = set(p[0] for p in placements)
            B = len(occupied_bins)
            
            # Find last bin (largest label)
            if occupied_bins:
                last_bin = max(occupied_bins)
                # Calculate occupied volume in last bin
                last_bin_vol = sum(w * h * d for (b, x, y, z, w, h, d) in placements if b == last_bin)
                last_frac = last_bin_vol / container_vol if container_vol > 0 else 0
            else:
                last_frac = 0
            
            score = B + 0.1 * (1 - last_frac)
            
            if score < best_score:
                best_score = score
                best_placements = placements
    
    # If no strategy worked, use the first one as fallback
    if best_placements is None:
        best_placements = solve_with_strategy(strategies[0])
    
    return best_placements
# EVOLVE-BLOCK-END
