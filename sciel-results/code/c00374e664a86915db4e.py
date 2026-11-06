"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)
    
    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # 3D Occupancy Grid
    use_grid = (W * H * D) < 50_000_000
    if use_grid:
        grid = np.zeros((W, H, D), dtype=np.uint8)
    else:
        grid = None
    
    def is_valid_grid(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid is not None:
            if np.any(grid[x:x+w, y:y+h, z:z+d] > 0):
                return False
            return True
        else:
            return False
    
    def place_in_grid(x, y, z, w, h, d):
        if grid is not None:
            grid[x:x+w, y:y+h, z:z+d] = 1
    
    def unplace_in_grid(x, y, z, w, h, d):
        if grid is not None:
            grid[x:x+w, y:y+h, z:z+d] = 0
    
    def count_contact(x, y, z, w, h, d):
        """Count how many faces of the new box touch occupied cells (max-contact heuristic)."""
        if grid is None:
            return 0
        contact = 0
        # Check bottom face (z-1)
        if z > 0:
            contact += np.sum(grid[x:x+w, y:y+h, z-1] > 0)
        # Check top face (z+d)
        if z + d < D:
            contact += np.sum(grid[x:x+w, y:y+h, z+d] > 0)
        # Check front face (y-1)
        if y > 0:
            contact += np.sum(grid[x:x+w, y-1, z:z+d] > 0)
        # Check back face (y+h)
        if y + h < H:
            contact += np.sum(grid[x:x+w, y+h, z:z+d] > 0)
        # Check left face (x-1)
        if x > 0:
            contact += np.sum(grid[x-1, y:y+h, z:z+d] > 0)
        # Check right face (x+w)
        if x + w < W:
            contact += np.sum(grid[x+w, y:y+h, z:z+d] > 0)
        return contact
    
    def pack_with_order(order):
        if grid is not None:
            grid[:] = 0
        
        placed = []
        result = [None] * n
        current_time = time.time()
        
        # Candidate points: start with origin
        candidates = [(0, 0, 0)]
        candidate_set = {(0, 0, 0)}
        
        for idx in order:
            if current_time > deadline:
                break
            box = boxes[idx]
            orients = orient_cache[idx]
            
            best_pos = None
            best_score = float('inf')
            best_contact = -1
            
            for cx, cy, cz in candidates:
                for ow, oh, od in orients:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    if is_valid_grid(cx, cy, cz, ow, oh, od):
                        # BFD scoring: minimize z, then y, then x
                        # Plus max-contact bonus to fill gaps
                        contact = count_contact(cx, cy, cz, ow, oh, od)
                        # Weight: z is most important, then y, then x, then contact
                        score = 1000000 * cz + 10000 * cy + 100 * cx - 10 * contact
                        if score < best_score or (score == best_score and contact > best_contact):
                            best_score = score
                            best_contact = contact
                            best_pos = (cx, cy, cz, ow, oh, od)
            
            if best_pos:
                x, y, z, ow, oh, od = best_pos
                placed.append(best_pos)
                result[idx] = best_pos
                place_in_grid(x, y, z, ow, oh, od)
                
                # Update candidates: add corners of the new box
                new_corners = [
                    (x, y, z), (x+ow, y, z), (x, y+oh, z), (x+ow, y+oh, z),
                    (x, y, z+od), (x+ow, y, z+od), (x, y+oh, z+od), (x+ow, y+oh, z+od)
                ]
                
                # Filter and add new candidates
                for nc in new_corners:
                    nx, ny, nz = nc
                    if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                        if grid is not None and grid[nx, ny, nz] == 0:
                            if nc not in candidate_set:
                                candidates.append(nc)
                                candidate_set.add(nc)
                
                # Prune candidates that are now occupied
                if grid is not None:
                    new_candidates = []
                    new_candidate_set = set()
                    for c in candidates:
                        if grid[c[0], c[1], c[2]] == 0:
                            new_candidates.append(c)
                            new_candidate_set.add(c)
                    candidates = new_candidates
                    candidate_set = new_candidate_set
                
                current_time = time.time()
            else:
                result[idx] = None
        
        return result
    
    # Define sorting orders
    indexed = list(range(n))
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    def aspect_ratio(i):
        w, h, d = boxes[i]
        m = min(w, h, d)
        if m == 0:
            return 0
        return max(w, h, d) / m
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    # Add a few random orders for diversity
    random.seed(42)
    random_orders = []
    for _ in range(5):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
    
    # Add orders sorted by different criteria
    order_min = sorted(indexed, key=lambda i: min(boxes[i]), reverse=True)
    order_surface = sorted(indexed, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    orders = [order_vol, order_max, order_sum, order_aspect, order_min, order_surface] + random_orders
    
    best_result = None
    best_volume = 0
    best_order = None
    
    for order in orders:
        if time.time() > deadline:
            break
        
        res = pack_with_order(order)
        
        vol = 0
        for i in range(n):
            if res[i] is not None:
                x, y, z, w, h, d = res[i]
                vol += w * h * d
        
        if vol > best_volume:
            best_volume = vol
            best_result = res
            best_order = order
    
    # Phase 2: Local search with perturbation
    if best_order is not None and time.time() < deadline - 15:
        # Use simulated annealing-style acceptance for better exploration
        current_order = best_order[:]
        current_vol = best_volume
        temp = 1.0
        cooling_rate = 0.95
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Perturb the current order by swapping a few elements
            perturbed_order = current_order[:]
            num_swaps = random.randint(2, 6)
            for _ in range(num_swaps):
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
            res = pack_with_order(perturbed_order)
            
            vol = 0
            for i in range(n):
                if res[i] is not None:
                    x, y, z, w, h, d = res[i]
                    vol += w * h * d
            
            # Accept if better, or with probability based on temperature
            if vol > current_vol:
                current_order = perturbed_order
                current_vol = vol
                if vol > best_volume:
                    best_volume = vol
                    best_result = res
                    best_order = perturbed_order
            else:
                # Accept worse solutions with decreasing probability
                delta = current_vol - vol
                if delta > 0:
                    accept_prob = np.exp(-delta / (temp * max(1, current_vol)))
                    if random.random() < accept_prob:
                        current_order = perturbed_order
                        current_vol = vol
            
            temp *= cooling_rate
        
        # Final polish: try small perturbations on the best order
        if time.time() < deadline - 5:
            for _ in range(30):
                if time.time() > deadline - 3:
                    break
                
                perturbed_order = best_order[:]
                # Small perturbation: swap 1-2 elements
                num_swaps = random.randint(1, 2)
                for _ in range(num_swaps):
                    i, j = random.randint(0, n-1), random.randint(0, n-1)
                    perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
                
                res = pack_with_order(perturbed_order)
                
                vol = 0
                for i in range(n):
                    if res[i] is not None:
                        x, y, z, w, h, d = res[i]
                        vol += w * h * d
                
                if vol > best_volume:
                    best_volume = vol
                    best_result = res
                    best_order = perturbed_order
    
    if best_result is None:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Format output
    final_placements = []
    for i in range(n):
        if best_result[i] is not None:
            x, y, z, w, h, d = best_result[i]
            final_placements.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            final_placements.append((-1, 0, 0, 0, w, h, d))
            
    return final_placements

# EVOLVE-BLOCK-END
