import time
import itertools

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin
    
    # Generate all 6 unique rotations for each box (dedup if symmetric)
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    box_rots = []
    for i, (w, h, d) in enumerate(boxes):
        box_rots.append(get_rotations(w, h, d))
    
    # Sorting strategies
    def sort_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def sort_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -max(boxes[i]))
    
    def sort_min_dim_asc(idx_list):
        return sorted(idx_list, key=lambda i: min(boxes[i]))
    
    def sort_ratio_desc(idx_list):
        # volume / max_dim^2, prefer "cube-like" or flat efficient
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(1, max(boxes[i])**2))
    
    strategies = [sort_volume_desc, sort_max_dim_desc, sort_min_dim_asc, sort_ratio_desc]
    
    def try_place(box_idx, rots, container_dims, placed):
        """Try to place box with given rotations in container. Returns (x,y,z,w,h,d) or None."""
        cW, cH, cD = container_dims
        for rw, rh, rd in rots:
            # Try all positions: this is expensive, so we use a smarter approach
            # For now, try a grid of candidate positions based on existing placements
            # But for simplicity and speed, let's try the "skyline" or "shelf" approach in 3D
            pass
        return None
    
    # A more practical approach: use a 3D shelf/skyline algorithm
    # We'll maintain a list of "shelves" or use a point-based free space check
    
    def place_box_3d(box_dims, rots, container_dims, placed_rects):
        """
        Try to place a box in the container given existing placed rectangles.
        Uses a candidate position approach: try positions at (0,0,0), and at the
        edges of existing boxes.
        Returns (x, y, z, w, h, d) or None.
        """
        cW, cH, cD = container_dims
        
        # Generate candidate x, y, z positions
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (px, py, pz, pw, ph, pd) in placed_rects:
            if px + pw <= cW:
                x_candidates.append(px + pw)
            if py + ph <= cH:
                y_candidates.append(py + ph)
            if pz + pd <= cD:
                z_candidates.append(pz + pd)
        
        # Remove duplicates and sort
        x_candidates = sorted(set(x for x in x_candidates if x <= cW))
        y_candidates = sorted(set(y for y in y_candidates if y <= cH))
        z_candidates = sorted(set(z for z in z_candidates if z <= cD))
        
        # Limit candidates to keep it fast
        if len(x_candidates) > 30:
            x_candidates = x_candidates[:30] + [cW - max(boxes[0][0], 1)]
            x_candidates = sorted(set(x for x in x_candidates if 0 <= x <= cW))
        if len(y_candidates) > 30:
            y_candidates = y_candidates[:30] + [cH - max(boxes[0][1], 1)]
            y_candidates = sorted(set(y for y in y_candidates if 0 <= y <= cH))
        if len(z_candidates) > 30:
            z_candidates = z_candidates[:30] + [cD - max(boxes[0][2], 1)]
            z_candidates = sorted(set(z for z in z_candidates if 0 <= z <= cD))
        
        for rw, rh, rd in rots:
            for x in x_candidates:
                if x + rw > cW:
                    continue
                for y in y_candidates:
                    if y + rh > cH:
                        continue
                    for z in z_candidates:
                        if z + rd > cD:
                            continue
                        # Check overlap with all placed boxes
                        overlap = False
                        for (px, py, pz, pw, ph, pd) in placed_rects:
                            if (x < px + pw and x + rw > px and
                                y < py + ph and y + rh > py and
                                z < pz + pd and z + rd > pz):
                                overlap = True
                                break
                        if not overlap:
                            return (x, y, z, rw, rh, rd)
        return None
    
    def run_strategy(sort_func):
        """Run a packing strategy and return placements."""
        placed_rects = []
        result = [None] * n
        order = sort_func(list(range(n)))
        
        for i in order:
            if time.time() > deadline:
                break
            rots = box_rots[i]
            pos = place_box_3d(boxes[i], rots, (W, H, D), placed_rects)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed_rects.append((x, y, z, rw, rh, rd))
                result[i] = (0, x, y, z, rw, rh, rd)
            else:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Fill in any that weren't processed due to time
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    # New: Time-sliced layer decomposition approach
    def run_layer_decomposition(sort_func, num_layers):
        """
        Decompose container height into horizontal slabs.
        Pack boxes into each slab using 2D shelf heuristic.
        """
        result = [None] * n
        order = sort_func(list(range(n)))
        layer_height = H / num_layers if num_layers > 0 else H
        
        # For each layer, maintain 2D shelf state
        # Shelf state: list of (x_start, width, shelf_height) for each shelf
        # We'll use a simple 2D shelf packing per layer
        
        for layer_idx in range(num_layers):
            z_start = layer_idx * layer_height
            z_end = min(z_start + layer_height, H)
            layer_h = z_end - z_start
            
            if layer_h <= 0:
                continue
            
            # 2D shelf packing for this layer
            # Shelves: each has a starting x, current x, and height
            shelves = []  # list of [x_start, x_current, height]
            
            for i in order:
                if time.time() > deadline:
                    break
                
                bw, bh, bd = boxes[i]
                
                # Find a rotation that fits in the layer height
                valid_rots = []
                for rw, rh, rd in box_rots[i]:
                    if rh <= layer_h:
                        valid_rots.append((rw, rh, rd))
                
                if not valid_rots:
                    continue
                
                # Try to place on existing shelves or create new shelf
                placed = False
                
                for shelf in shelves:
                    x_start, x_current, shelf_h = shelf
                    shelf_width = W - x_start
                    
                    # Try to fit box on this shelf
                    for rw, rh, rd in valid_rots:
                        if rw <= shelf_width and rh <= shelf_h:
                            # Place box
                            placed_rects = []  # We'll track 2D placements
                            # For simplicity, just place at x_current
                            result[i] = (0, x_start + x_current, z_start, 0, rw, rh, rd)
                            # Actually we need y coordinate too - let's use a simpler approach
                            # Let's just use the 3D approach but with z constrained
                            break
                    
                    if placed:
                        break
                
                if not placed:
                    # Try to create a new shelf
                    for rw, rh, rd in valid_rots:
                        if rw <= W and rh <= layer_h:
                            # Create new shelf
                            shelves.append([0, rw, rh])
                            result[i] = (0, 0, z_start, 0, rw, rh, rd)
                            placed = True
                            break
            
            # The above is too simplistic. Let's use a better 2D approach.
            # Actually, let's just use the existing 3D approach but with z constrained to layer.
        
        # The layer decomposition as described above is flawed. Let's use a different approach:
        # Use the existing 3D placement but sort by z-coordinate preference.
        
        # Actually, let's implement a proper layer decomposition:
        # 1. Divide height into layers
        # 2. For each layer, do 2D packing in the W x D plane
        # 3. Assign boxes to layers based on their height dimension
        
        result = [None] * n
        order = sort_func(list(range(n)))
        
        # Assign boxes to layers based on their minimum dimension
        # Sort boxes by their minimum dimension (to fit in thin layers)
        box_layer_assignment = {}
        
        for i in order:
            if time.time() > deadline:
                break
            bw, bh, bd = boxes[i]
            min_dim = min(bw, bh, bd)
            
            # Assign to the thinnest layer that can fit this box
            for layer_idx in range(num_layers):
                z_start = layer_idx * layer_height
                z_end = min(z_start + layer_height, H)
                layer_h = z_end - z_start
                
                if min_dim <= layer_h:
                    box_layer_assignment[i] = layer_idx
                    break
        
        # Now pack each layer
        for layer_idx in range(num_layers):
            z_start = layer_idx * layer_height
            z_end = min(z_start + layer_height, H)
            layer_h = z_end - z_start
            
            if layer_h <= 0:
                continue
            
            # Get boxes assigned to this layer
            layer_boxes = [i for i in order if box_layer_assignment.get(i) == layer_idx]
            
            # 2D shelf packing in W x D plane
            # Shelves: list of [y_start, y_current, shelf_width]
            # Actually, let's use a simpler grid-based approach for 2D
            
            placed_in_layer = []  # list of (x, y, w, d) for 2D placements
            
            for i in layer_boxes:
                if time.time() > deadline:
                    break
                
                bw, bh, bd = boxes[i]
                
                # Find rotation that fits in layer height
                valid_rots = []
                for rw, rh, rd in box_rots[i]:
                    if rh <= layer_h:
                        valid_rots.append((rw, rd))  # (x_dim, z_dim) for 2D
                        break  # Take first valid rotation for speed
                
                if not valid_rots:
                    continue
                
                rw, rd = valid_rots[0]
                
                # Try to place in 2D
                # Use a simple grid of candidate positions
                x_candidates = [0]
                y_candidates = [0]
                
                for (px, py, pw, pd) in placed_in_layer:
                    x_candidates.append(px + pw)
                    y_candidates.append(py + pd)
                
                x_candidates = sorted(set(x for x in x_candidates if x <= W))
                y_candidates = sorted(set(y for y in y_candidates if y <= D))
                
                placed = False
                for x in x_candidates:
                    if x + rw > W:
                        continue
                    for y in y_candidates:
                        if y + rd > D:
                            continue
                        
                        # Check overlap
                        overlap = False
                        for (px, py, pw, pd) in placed_in_layer:
                            if (x < px + pw and x + rw > px and
                                y < py + pd and y + rd > py):
                                overlap = True
                                break
                        
                        if not overlap:
                            placed_in_layer.append((x, y, rw, rd))
                            result[i] = (0, x, y, z_start, rw, layer_h, rd)
                            placed = True
                            break
                    if placed:
                        break
        
        # Fill in any that weren't processed
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    best_placements = None
    best_score = -1
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
        
        placements = run_strategy(sort_func)
        
        # Score: fraction of container volume filled
        total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        score = total_vol / (W * H * D) if W * H * D > 0 else 0
        
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # Try layer decomposition with different numbers of layers
    if time.time() < deadline:
        for num_layers in [2, 3, 4]:
            if time.time() > deadline:
                break
            
            for sort_func in strategies:
                if time.time() > deadline:
                    break
                
                placements = run_layer_decomposition(sort_func, num_layers)
                
                # Score: fraction of container volume filled
                total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
                score = total_vol / (W * H * D) if W * H * D > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_placements = placements
    
    if best_placements is None:
        # Fallback: try the original simple approach
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
