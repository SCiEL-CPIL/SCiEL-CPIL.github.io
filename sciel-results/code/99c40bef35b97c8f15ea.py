"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    best_volume = 0
    
    def get_all_orientations(w, h, d):
        """Return all 6 unique orientations of a box."""
        orientations = set()
        for perm in itertools.permutations([w, h, d]):
            orientations.add(perm)
        return list(orientations)
    
    def try_pack(ordered_boxes, W, H, D):
        """Try to pack boxes using simple shelf-based approach with rotations."""
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
        total_volume = 0
        
        # Shelf-based packing
        shelves = []  # list of (y, height, next_x)
        y_pos = 0
        
        for idx, (orig_idx, w, h, d) in enumerate(ordered_boxes):
            if time.time() > deadline:
                break
            
            orientations = get_all_orientations(w, h, d)
            placed = False
            
            for ow, oh, od in orientations:
                # Try to place on existing shelves
                for shelf in shelves:
                    y, shelf_h, next_x = shelf
                    if next_x + ow <= W and y + oh <= H:
                        # Check if this orientation fits in height
                        if oh <= shelf_h:
                            placements[orig_idx] = (0, next_x, y, 0, ow, oh, od)
                            shelf[2] = next_x + ow
                            total_volume += ow * oh * od
                            placed = True
                            break
                
                if placed:
                    continue
                
                # Try to create a new shelf
                for ow, oh, od in orientations:
                    if y_pos + oh <= H and ow <= W:
                        # Check if this orientation fits
                        if oh > 0:
                            new_shelf = [0, oh, ow]  # y=0 for new layer, height, next_x
                            # Actually, we need to track y properly
                            # Let's use a different approach: layers
                            break
                
                # Create new layer
                if y_pos + min(oh for ow, oh, od in orientations) <= H:
                    # Find best orientation for new layer
                    best_orient = None
                    for ow, oh, od in orientations:
                        if ow <= W and y_pos + oh <= H:
                            if best_orient is None or oh < best_orient[1]:
                                best_orient = (ow, oh, od)
                    
                    if best_orient:
                        ow, oh, od = best_orient
                        placements[orig_idx] = (0, 0, y_pos, 0, ow, oh, od)
                        shelves.append([y_pos, oh, ow])
                        y_pos += oh
                        total_volume += ow * oh * od
                        placed = True
        
        return placements, total_volume
    
    # Simple shelf-based packing with rotation
    placements, total_volume = try_pack(indexed_boxes, W, H, D)
    best_placements = placements
    best_volume = total_volume
    
    # Try a few different orderings if time permits
    if time.time() < deadline - 20:
        # Try sorting by different criteria
        for sort_key in [
            lambda x: x[1] * x[2] * x[3],  # volume
            lambda x: x[1],  # width
            lambda x: x[2],  # height
            lambda x: x[3],  # depth
            lambda x: max(x[1], x[2], x[3]),  # max dimension
        ]:
            if time.time() > deadline - 10:
                break
            
            ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
            p, v = try_pack(ordered, W, H, D)
            if v > best_volume:
                best_placements = p
                best_volume = v
    
    return best_placements
# EVOLVE-BLOCK-END
