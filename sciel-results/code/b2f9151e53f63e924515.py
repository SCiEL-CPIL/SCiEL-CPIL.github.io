"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 150s limit, keep 15s margin for safety/overhead
    deadline = start_time + 130.0
    
    # Preprocess: store box dimensions and generate unique rotations
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        # Generate all unique permutations of dimensions
        dims = list({tuple(sorted([w, h, d])), (w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h)})
        # Actually just get unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        # Filter to only valid orientations (all positive)
        rots = [(rw, rh, rd) for rw, rh, rd in rots if rw > 0 and rh > 0 and rd > 0]
        # Remove duplicates while preserving order
        seen = set()
        unique_rots = []
        for r in rots:
            if r not in seen:
                seen.add(r)
                unique_rots.append(r)
        box_data.append((i, unique_rots))
    
    def get_rotations(idx):
        return box_data[idx][1]
    
    def pack_with_order(order, boxes_dims):
        """Pack boxes in given order using First Fit Decreasing with best corner."""
        W, H, D = container
        # Track occupied space using a simple 3D occupancy grid approach
        # For efficiency, use a list of placed boxes and check overlap
        placed = []  # list of (x, y, z, w, h, d)
        
        def can_place(x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for px, py, pz, pw, ph, pd in placed:
                # Check overlap
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True
        
        def find_best_position(w, h, d):
            """Find the best position for a box with given dimensions."""
            best_score = -1
            best_pos = None
            
            # Try all possible positions on a grid of potential anchor points
            # Use existing placed boxes to generate candidate positions
            candidate_xs = [0]
            candidate_ys = [0]
            candidate_zs = [0]
            
            for px, py, pz, pw, ph, pd in placed:
                candidate_xs.append(px + pw)
                candidate_ys.append(py + ph)
                candidate_zs.append(pz + pd)
            
            # Also try positions near the edges
            candidate_xs.append(max(0, W - w))
            candidate_ys.append(max(0, H - h))
            candidate_zs.append(max(0, D - d))
            
            # Deduplicate and filter
            candidate_xs = sorted(set([x for x in candidate_xs if 0 <= x <= W - w]))
            candidate_ys = sorted(set([y for y in candidate_ys if 0 <= y <= H - h]))
            candidate_zs = sorted(set([z for z in candidate_zs if 0 <= z <= D - d]))
            
            for x in candidate_xs:
                for y in candidate_ys:
                    for z in candidate_zs:
                        if can_place(x, y, z, w, h, d):
                            # Score: prefer lower z, then lower y, then lower x (fill from bottom)
                            score = z * 10000 + y * 100 + x
                            if score < best_score or best_pos is None:
                                if best_pos is None or score < best_score:
                                    best_score = score
                                    best_pos = (x, y, z)
            
            return best_pos
        
        result = [-1] * len(boxes_dims)
        for idx in order:
            w, h, d = boxes_dims[idx]
            # Try all rotations
            best_rotation = None
            best_pos = None
            best_score = -1
            
            rots = get_rotations(idx)
            for rw, rh, rd in rots:
                pos = find_best_position(rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer tighter fit (lower z, y, x)
                    score = z * 10000 + y * 100 + x
                    if best_pos is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rotation = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rotation
                placed.append((x, y, z, rw, rh, rd))
                result[idx] = 0
            else:
                result[idx] = -1
        
        return result, placed
    
    def simulate_annealing(initial_order, boxes_dims, max_iterations=5000):
        """Simulated annealing to improve the packing order."""
        W, H, D = container
        current_order = list(initial_order)
        current_result, current_placed = pack_with_order(current_order, boxes_dims)
        current_volume = sum(boxes_dims[i][0] * boxes_dims[i][1] * boxes_dims[i][2] for i in range(len(boxes_dims)) if current_result[i] == 0)
        
        best_order = list(current_order)
        best_volume = current_volume
        best_result = list(current_result)
        
        T = 1.0
        cooling = 0.9995
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Generate neighbor by swapping two boxes
            if len(current_order) < 2:
                break
            i, j = random.sample(range(len(current_order)), 2)
            new_order = list(current_order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_result, new_placed = pack_with_order(new_order, boxes_dims)
            new_volume = sum(boxes_dims[i][0] * boxes_dims[i][1] * boxes_dims[i][2] for i in range(len(boxes_dims)) if new_result[i] == 0)
            
            delta = new_volume - current_volume
            if delta > 0 or random.random() < math.exp(delta / (T * max(1, current_volume))):
                current_order = new_order
                current_result = new_result
                current_volume = new_volume
                if current_volume > best_volume:
                    best_volume = current_volume
                    best_order = list(current_order)
                    best_result = list(current_result)
            
            T *= cooling
            if T < 0.001:
                break
        
        return best_order, best_result, best_volume
    
    # Initial greedy order: sort by volume descending
    box_volumes = [(boxes[i][0] * boxes[i][1] * boxes[i][2], i) for i in range(n)]
    box_volumes.sort(reverse=True)
    initial_order = [idx for _, idx in box_volumes]
    
    # Run initial packing
    initial_result, initial_placed = pack_with_order(initial_order, boxes)
    initial_volume = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n) if initial_result[i] == 0)
    
    best_order = list(initial_order)
    best_result = list(initial_result)
    best_volume = initial_volume
    
    # Try simulated annealing with multiple restarts
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create perturbed initial order
        perturbed_order = list(initial_order)
        # Randomly swap some elements
        for _ in range(min(10, n // 2)):
            i, j = random.sample(range(n), 2)
            perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
        
        # Run simulated annealing
        sa_order, sa_result, sa_volume = simulate_annealing(perturbed_order, boxes, max_iterations=2000)
        
        if sa_volume > best_volume:
            best_volume = sa_volume
            best_order = sa_order
            best_result = sa_result
    
    # Build final placements
    placements = []
    # We need to reconstruct actual positions from the best result
    # Re-run packing with best order to get positions
    final_result, final_placed = pack_with_order(best_order, boxes)
    
    # Map back to original box indices
    # final_placed is in order of best_order, so we need to map
    placed_by_box = {}
    for pos_idx, box_idx in enumerate(best_order):
        if final_result[box_idx] == 0:
            placed_by_box[box_idx] = final_placed[pos_idx]
    
    for i in range(n):
        if i in placed_by_box:
            x, y, z, w, h, d = placed_by_box[i]
            placements.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
