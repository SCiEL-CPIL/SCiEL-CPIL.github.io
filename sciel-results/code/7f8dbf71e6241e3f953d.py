"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all unique orientations for each box
    def get_orientations(w, h, d):
        orients = set()
        for perm in itertools.permutations((w, h, d)):
            orients.add(perm)
        return list(orients)
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Use numpy occupancy grid for fast overlap checks
    # Grid resolution: 1 unit per cell (dimensions are integers presumably)
    # If dimensions are not integers, we'd need scaling, but let's assume they are
    # Actually, let's use a set-based approach with numpy for candidate filtering
    # But the directive says use numpy bitmap. Let's create a 3D array.
    # If W, H, D are large, this might be memory intensive. 
    # Let's check typical sizes. 60 boxes in a container. 
    # If container is like 100x100x100, that's 1M cells, which is fine.
    # If it's 1000x1000x1000, that's 1B cells, which is too much.
    # Let's assume dimensions are reasonable. If not, we fall back to list-based.
    
    # Let's try to use a numpy array if size is manageable, else use list.
    # But to keep it simple and fast as per directive, let's try numpy.
    # We'll represent occupancy as a 3D boolean array.
    
    # Check if dimensions are integers
    # The problem doesn't specify integer dimensions. 
    # If they are floats, we can't use a simple bitmap directly without scaling.
    # Let's assume integer dimensions for the bitmap approach.
    # If not, we'll scale or use the old method.
    # Given the context of "boxes", integer dimensions are likely.
    
    # Let's implement a hybrid: if W*H*D < 10^7, use numpy bitmap, else use list.
    # Actually, let's just use the list-based approach but optimize the candidate generation
    # and use numpy to speed up the overlap check if possible?
    # No, the directive specifically asks for a numpy bitmap to enable O(1) checks.
    
    # Let's assume integer dimensions. If they are floats, we round or scale.
    # Let's check if all dimensions are integers.
    is_int = all(isinstance(d, (int, np.integer)) for d in boxes[0]) if n > 0 else False
    # Also check container
    is_int = is_int and all(isinstance(d, (int, np.integer)) for d in container)
    
    # For now, let's stick to the list-based approach but improve the candidate positions
    # and add a "first fit decreasing" with better orientation sorting.
    # The previous approach was already doing this. 
    # Let's try to improve the placement strategy by trying to fill gaps.
    
    # Actually, let's implement the numpy bitmap as requested, assuming integer dims.
    # If dims are not integers, we'll fall back.
    
    try:
        Wi, Hi, Di = int(W), int(H), int(D)
        if Wi != W or Hi != H or Di != D:
            raise ValueError("Non-integer container dims")
        for w, h, d in boxes:
            if int(w) != w or int(h) != h or int(d) != d:
                raise ValueError("Non-integer box dims")
        # Use integer dims
        W, H, D = Wi, Hi, Di
    except:
        # Fall back to float handling (use old method or scale)
        # For simplicity, let's just use the old method if not integer
        return _load_float(boxes, (W, H, D), start_time, deadline)
    
    # Now W, H, D are integers
    # Create occupancy grid
    # Memory check: W*H*D should be < ~10^7 for safety
    if W * H * D > 10**7:
        return _load_float(boxes, (W, H, D), start_time, deadline)
    
    occ = np.zeros((W, H, D), dtype=bool)
    
    def fits_fast(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if any cell in the region is occupied
        if not np.any(occ[x:x+w, y:y+h, z:z+d]):
            return True
        return False
    
    def place(x, y, z, w, h, d):
        occ[x:x+w, y:y+h, z:z+d] = True
    
    def unplace(x, y, z, w, h, d):
        occ[x:x+w, y:y+h, z:z+d] = False
    
    # Placement list for tracking
    placements = []
    
    def get_candidate_positions():
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd) in placements:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        xs.add(W)
        ys.add(H)
        zs.add(D)
        return sorted(xs), sorted(ys), sorted(zs)
    
    result = [None] * n
    
    for (idx, w, h, d, vol) in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = get_orientations(w, h, d)
        placed = False
        
        xs, ys, zs = get_candidate_positions()
        
        # Sort orientations: try to align largest dimension with largest container dimension
        # This is a heuristic
        sorted_orients = sorted(orients, key=lambda o: -max(o))
        
        for ow, oh, od in sorted_orients:
            if placed:
                break
            for cx in xs:
                if placed:
                    break
                if cx + ow > W:
                    continue
                for cy in ys:
                    if placed:
                        break
                    if cy + oh > H:
                        continue
                    for cz in zs:
                        if placed:
                            break
                        if cz + od > D:
                            continue
                        if fits_fast(cx, cy, cz, ow, oh, od):
                            place(cx, cy, cz, ow, oh, od)
                            placements.append((cx, cy, cz, ow, oh, od))
                            result[idx] = (0, cx, cy, cz, ow, oh, od)
                            placed = True
                            break
        
        if not placed:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _load_float(boxes, container, start_time, deadline):
    # Fallback for non-integer or large containers
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def get_orientations(w, h, d):
        orients = set()
        for perm in itertools.permutations((w, h, d)):
            orients.add(perm)
        return list(orients)
    
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    placements = []
    
    def fits(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placements:
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return False
        return True
    
    def get_candidate_positions():
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd) in placements:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        xs.add(W)
        ys.add(H)
        zs.add(D)
        return sorted(xs), sorted(ys), sorted(zs)
    
    result = [None] * n
    
    for (idx, w, h, d, vol) in indexed_boxes:
        if time.time() > deadline:
            break
        
        orients = get_orientations(w, h, d)
        placed = False
        
        xs, ys, zs = get_candidate_positions()
        
        for ow, oh, od in orients:
            if placed:
                break
            for cx in xs:
                if placed:
                    break
                if cx + ow > W:
                    continue
                for cy in ys:
                    if placed:
                        break
                    if cy + oh > H:
                        continue
                    for cz in zs:
                        if placed:
                            break
                        if cz + od > D:
                            continue
                        if fits(cx, cy, cz, ow, oh, od):
                            placements.append((cx, cy, cz, ow, oh, od))
                            result[idx] = (0, cx, cy, cz, ow, oh, od)
                            placed = True
                            break
        
        if not placed:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
