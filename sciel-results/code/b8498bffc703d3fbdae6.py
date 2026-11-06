"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]:
        rots.add(perm)
    return list(rots)

class Container:
    """Represents the container with a 3D occupancy grid for fast overlap checking."""
    def __init__(self, W, H, D):
        self.W = W
        self.H = H
        self.D = D
        # Using a 3D numpy array for O(1) range sum checks (approximate) or direct indexing
        # Since boxes are integer dimensions, we can use a grid.
        # However, if dimensions are large, this might be memory heavy. 
        # Assuming reasonable dimensions for 60 boxes.
        self.grid = np.zeros((W, H, D), dtype=np.uint8)
        self.boxes = [] # List of (idx, x, y, z, w, h, d)
        
    def is_free(self, x, y, z, w, h, d):
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
        # Check if the region is empty
        # Numpy slicing is fast
        region = self.grid[x:x+w, y:y+h, z:z+d]
        return not region.any()
        
    def place(self, idx, x, y, z, w, h, d):
        self.grid[x:x+w, y:y+h, z:z+d] = 1
        self.boxes.append((idx, x, y, z, w, h, d))
        
    def remove(self, idx):
        for i, (bidx, x, y, z, w, h, d) in enumerate(self.boxes):
            if bidx == idx:
                self.grid[x:x+w, y:y+h, z:z+d] = 0
                self.boxes.pop(i)
                break

def _blb_place(boxes, container, sort_key='volume_desc', deadline=None, seed=None):
    """
    Bottom-Left-Back placement heuristic.
    Scans the container from (0,0,0) to find the first valid position for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    indexed = list(enumerate(boxes))
    if sort_key == 'volume_desc':
        indexed.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
    elif sort_key == 'max_dim_desc':
        indexed.sort(key=lambda t: max(t[1]), reverse=True)
    elif sort_key == 'min_dim_asc':
        indexed.sort(key=lambda t: min(t[1]))
    elif sort_key == 'aspect_desc':
        indexed.sort(key=lambda t: max(t[1])/max(min(t[1]),1e-9), reverse=True)
    elif sort_key == 'random':
        if seed is not None:
            random.seed(seed)
        random.shuffle(indexed)
    else:
        indexed.sort(key=lambda t: t[1][0]*t[1][1]*t[1][2], reverse=True)
        
    cont = Container(W, H, D)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    used_volume = 0.0
    
    for idx, (bw, bh, bd) in indexed:
        if deadline and time.time() > deadline:
            break
            
        rots = _get_rotations(bw, bh, bd)
        best_pos = None
        best_score = float('inf')
        
        # Try all rotations
        for (rw, rh, rd) in rots:
            # Scan for BLB position
            # Optimization: We can break early if we find a position with score 0 (0,0,0)
            found = False
            # Iterate z, y, x in increasing order to find the "lowest, leftmost, backmost"
            # This is O(W*H*D) in worst case, which is too slow if dimensions are large.
            # However, for 60 boxes, the occupied space is limited.
            # A better way: Maintain a list of "extreme points" (corners of placed boxes) and check those first.
            # But BLB scan is more robust for finding global minimums in the free space.
            # Let's use a hybrid: Check extreme points first, then scan if needed? 
            # Actually, for this problem size, a simple scan might be too slow if W,H,D are large.
            # Let's stick to the Extreme Point method but with a denser set of candidates.
            
            # Candidate points: Origin + corners of all placed boxes
            candidates = set()
            candidates.add((0, 0, 0))
            for (_, px, py, pz, pw, ph, pd) in cont.boxes:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                # Also add some intermediate points?
                # To improve BLB behavior, we should scan from 0,0,0.
                # Let's implement a quick scan for the specific box size.
                # If the container is small (e.g., < 50^3), full scan is fine.
                # If large, we rely on extreme points.
                
            # Sort candidates by (x+y+z) to prefer compactness, but we want the "first" in BLB order (x, then y, then z? or z, then y, then x?)
            # Standard BLB: Minimize x, then y, then z? Or Minimize z (height), then x, then y?
            # Usually, we want to fill from the bottom (z=0) up. So minimize z, then x, then y.
            # Let's sort candidates by (z, x, y).
            candidates = sorted(candidates, key=lambda c: (c[2], c[0], c[1]))
            
            for (cx, cy, cz) in candidates:
                if not cont.is_free(cx, cy, cz, rw, rh, rd):
                    continue
                score = cz * 10000 + cx * 100 + cy # Weight z heavily
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz, rw, rh, rd)
                    # We can't break immediately because a later candidate might have lower z?
                    # No, we sorted by z, so the first valid one with lowest z is best.
                    # But there might be multiple with same z, we want lowest x, then y.
                    # The sort key (z, x, y) ensures the first valid one is the best BLB for this rotation.
                    break
            
            if best_pos:
                # Check if this rotation's position is better than previous rotations' positions
                # We already compare scores, so this is handled.
                pass
        
        if best_pos:
            cx, cy, cz, rw, rh, rd = best_pos
            cont.place(idx, cx, cy, cz, rw, rh, rd)
            result[idx] = (0, cx, cy, cz, rw, rh, rd)
            used_volume += rw * rh * rd
            
    return result

def _shake_local_search(boxes, container, initial_result, deadline, num_shakes=50):
    """
    Local search by removing a subset of boxes and re-packing them.
    """
    W, H, D = container
    n = len(boxes)
    
    # Parse initial result into a Container object
    cont = Container(W, H, D)
    placed_indices = []
    for i in range(n):
        r = initial_result[i]
        if r[0] == 0:
            cont.place(i, r[1], r[2], r[3], r[4], r[5], r[6])
            placed_indices.append(i)
            
    best_result = [tuple(r) for r in initial_result]
    best_vol = sum(r[4]*r[5]*r[6] for r in best_result if r[0] == 0)
    
    # Keep a list of placed boxes for quick removal
    # We need to be able to remove and re-add specific boxes
    
    for _ in range(num_shakes):
        if time.time() > deadline:
            break
            
        # Randomly select a few boxes to remove (e.g., 10% of placed boxes)
        if not placed_indices:
            break
            
        k = max(1, len(placed_indices) // 10)
        # Shuffle placed_indices to pick random subset
        temp_list = placed_indices[:]
        random.shuffle(temp_list)
        to_remove = temp_list[:k]
        
        # Remove them
        removed_boxes = []
        for idx in to_remove:
            r = best_result[idx]
            removed_boxes.append((idx, r[4], r[5], r[6])) # Store original dims? No, we need to re-rotate.
            cont.remove(idx)
            best_result[idx] = (-1, 0, 0, 0, 0, 0, 0)
            
        # Sort removed boxes by volume desc to place largest first
        removed_boxes.sort(key=lambda t: t[1]*t[2]*t[3], reverse=True)
        
        # Re-insert them using BLB logic
        for idx, bw, bh, bd in removed_boxes:
            rots = _get_rotations(bw, bh, bd)
            best_pos = None
            best_score = float('inf')
            
            candidates = set()
            candidates.add((0, 0, 0))
            for (_, px, py, pz, pw, ph, pd) in cont.boxes:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                
            candidates = sorted(candidates, key=lambda c: (c[2], c[0], c[1]))
            
            for (rw, rh, rd) in rots:
                for (cx, cy, cz) in candidates:
                    if not cont.is_free(cx, cy, cz, rw, rh, rd):
                        continue
                    score = cz * 10000 + cx * 100 + cy
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, rw, rh, rd)
                        break # Best BLB for this rotation found
                
                if best_pos:
                    # Check if this is better than current best for this box
                    # We are just trying to place it, so any valid spot is good.
                    # But we want the best spot.
                    pass
                
            if best_pos:
                cx, cy, cz, rw, rh, rd = best_pos
                cont.place(idx, cx, cy, cz, rw, rh, rd)
                best_result[idx] = (0, cx, cy, cz, rw, rh, rd)
                placed_indices.append(idx)
        
        # Check if volume improved
        current_vol = sum(r[4]*r[5]*r[6] for r in best_result if r[0] == 0)
        if current_vol > best_vol:
            best_vol = current_vol
            # We already updated best_result and cont
        else:
            # Revert? It's complex to revert. 
            # Instead, we should have worked on a copy or saved state.
            # For simplicity and speed, we only keep the change if it's better.
            # If it's not better, we need to restore the previous state.
            # This is expensive. 
            # Alternative: Always keep the change if it's valid, even if volume is same?
            # No, we want to maximize volume.
            # Let's do a "commit or rollback" strategy.
            # Save state before shake.
            pass # Rollback logic omitted for brevity in this thought, but implemented in code below.

    return best_result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_vol = -1
    
    # 1. Multi-start Greedy
    strategies = [
        'volume_desc',
        'max_dim_desc',
        'min_dim_asc',
        'aspect_desc',
    ]
    
    for strat in strategies:
        if time.time() > deadline:
            break
        result = _blb_place(boxes, container, sort_key=strat, deadline=deadline)
        vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            
    # 2. Random starts
    if time.time() < deadline - 10:
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _blb_place(boxes, container, sort_key='random', deadline=deadline, seed=seed)
            vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == 0)
            if vol > best_vol:
                best_vol = vol
                best_result = result

    # 3. Local Search (Shake)
    if time.time() < deadline - 5 and best_result is not None:
        # Implement a proper shake with rollback
        W, H, D = container
        n = len(boxes)
        
        # Helper to get volume
        def get_vol(res):
            return sum(r[4]*r[5]*r[6] for r in res if r[0] == 0)
            
        current_result = [tuple(r) for r in best_result]
        current_vol = best_vol
        
        # Pre-compute rotations for all boxes
        all_rots = [_get_rotations(b[0], b[1], b[2]) for b in boxes]
        
        # Local search loop
        while time.time() < deadline - 2:
            # Select boxes to shake
            loaded = [i for i in range(n) if current_result[i][0] == 0]
            if not loaded:
                break
                
            k = max(1, len(loaded) // 5) # Shake 20%
            random.shuffle(loaded)
            to_shake = loaded[:k]
            
            # Save state
            saved_result = [tuple(r) for r in current_result]
            
            # Build occupancy for current state
            cont = Container(W, H, D)
            for i in range(n):
                if current_result[i][0] == 0:
                    r = current_result[i]
                    cont.place(i, r[1], r[2], r[3], r[4], r[5], r[6])
            
            # Remove shaken boxes
            for i in to_shake:
                r = current_result[i]
                cont.remove(i)
                current_result[i] = (-1, 0, 0, 0, 0, 0, 0)
                
            # Sort shaken boxes by volume desc
            to_shake.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
            
            # Re-insert
            for i in to_shake:
                bw, bh, bd = boxes[i]
                rots = all_rots[i]
                best_pos = None
                best_score = float('inf')
                
                candidates = set()
                candidates.add((0, 0, 0))
                for (_, px, py, pz, pw, ph, pd) in cont.boxes:
                    candidates.add((px + pw, py, pz))
                    candidates.add((px, py + ph, pz))
                    candidates.add((px, py, pz + pd))
                    
                candidates = sorted(candidates, key=lambda c: (c[2], c[0], c[1]))
                
                for (rw, rh, rd) in rots:
                    for (cx, cy, cz) in candidates:
                        if not cont.is_free(cx, cy, cz, rw, rh, rd):
                            continue
                        score = cz * 10000 + cx * 100 + cy
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, rw, rh, rd)
                        break # Found best BLB for this rotation
                
                if best_pos:
                    cx, cy, cz, rw, rh, rd = best_pos
                    cont.place(i, cx, cy, cz, rw, rh, rd)
                    current_result[i] = (0, cx, cy, cz, rw, rh, rd)
            
            # Check improvement
            new_vol = get_vol(current_result)
            if new_vol > current_vol:
                current_vol = new_vol
                best_vol = new_vol
                best_result = current_result
            else:
                # Rollback
                current_result = saved_result

    if best_result is None:
        # Fallback
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
        
    return best_result
# EVOLVE-BLOCK-END
