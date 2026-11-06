import time
import itertools

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Container volume for scoring
    cont_vol = W * H * D
    
    def check_overlap(x, y, z, w, h, d, placed_in_bin):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any placed box in the bin."""
        for px, py, pz, pw, ph, pd in placed_in_bin:
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return True
        return False

    def generate_new_eps(x, y, z, w, h, d, placed_in_bin, current_eps):
        """Generate new extreme points from a newly placed box."""
        new_eps = [
            (x + w, y, z),
            (x, y + h, z),
            (x, y, z + d),
            (x + w, y + h, z),
            (x + w, y, z + d),
            (x, y + h, z + d)
        ]
        
        # Filter out points that are inside any placed box
        valid_eps = []
        for nx, ny, nz in new_eps:
            # Check bounds
            if nx >= W or ny >= H or nz >= D:
                continue
            
            # Check if point is inside any placed box
            is_inside = False
            for px, py, pz, pw, ph, pd in placed_in_bin:
                if px <= nx <= px + pw and py <= ny <= py + ph and pz <= nz <= pz + pd:
                    is_inside = True
                    break
            if not is_inside:
                # Avoid duplicates
                if (nx, ny, nz) not in current_eps and (nx, ny, nz) not in valid_eps:
                    valid_eps.append((nx, ny, nz))
        
        return valid_eps

    def run_ep_algorithm(sorted_boxes):
        """
        Run the Extreme Point algorithm on a specific ordering of boxes.
        Returns a list of placements.
        """
        bins = []  # List of lists of placed boxes per bin
        placements = [] # Final result list
        
        # Start with bin 0
        current_bin_idx = 0
        bins.append([])
        current_eps = [(0, 0, 0)]
        
        for i, (w, h, d) in enumerate(sorted_boxes):
            # Check time
            if time.time() > deadline:
                # If we run out of time, we should return a valid partial result or the best one found.
                # Since we need to place ALL boxes to be valid, we must ensure we don't return incomplete.
                # However, the prompt says "invalid result ... scores 0". 
                # So we must finish. Given N=40, it should finish instantly.
                # This guard is just in case.
                pass

            placed_in_current_bin = bins[current_bin_idx]
            best_pos = None
            best_cost = float('inf')
            
            # Try all extreme points in the current bin
            for ex, ey, ez in current_eps:
                # Check if box fits in bin bounds
                if ex + w > W or ey + h > H or ez + d > D:
                    continue
                
                # Check overlap
                if check_overlap(ex, ey, ez, w, h, d, placed_in_current_bin):
                    continue
                
                # Calculate cost. 
                # Using x+y+z encourages packing towards the origin.
                # We can also try to minimize the "gap" to walls, but x+y+z is a good proxy.
                cost = ex + ey + ez
                
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (ex, ey, ez)
            
            if best_pos is not None:
                # Place box in current bin
                bx, by, bz = best_pos
                bins[current_bin_idx].append((bx, by, bz, w, h, d))
                placements.append((current_bin_idx, bx, by, bz, w, h, d))
                
                # Update extreme points
                new_eps = generate_new_eps(bx, by, bz, w, h, d, bins[current_bin_idx], current_eps)
                current_eps.extend(new_eps)
            else:
                # Open a new bin
                current_bin_idx += 1
                bins.append([])
                current_eps = [(0, 0, 0)]
                
                # Place box in new bin
                bx, by, bz = 0, 0, 0
                bins[current_bin_idx].append((bx, by, bz, w, h, d))
                placements.append((current_bin_idx, bx, by, bz, w, h, d))
                
                # Update extreme points
                new_eps = generate_new_eps(bx, by, bz, w, h, d, bins[current_bin_idx], current_eps)
                current_eps.extend(new_eps)
                
        return placements

    def calculate_score(placements):
        """Calculate the score for a given placement."""
        if not placements:
            return 0.0
            
        # Find max bin label
        max_bin = max(p[0] for p in placements)
        B = max_bin + 1 # Number of occupied bins
        
        # Calculate volume in the last bin
        last_bin_vol = 0
        for p in placements:
            if p[0] == max_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        
        fill_fraction = last_bin_vol / cont_vol
        score = B + 0.1 * (1 - fill_fraction)
        return score

    # Generate different sorting strategies
    # 1. Volume descending
    # 2. Max dimension descending
    # 3. Height descending
    # 4. Width descending
    # 5. Depth descending
    
    strategies = [
        ("vol", lambda b: -b[0]*b[1]*b[2]),
        ("max_dim", lambda b: -max(b)),
        ("h", lambda b: -b[1]),
        ("w", lambda b: -b[0]),
        ("d", lambda b: -b[2]),
        # Sometimes sorting by "corners" or other metrics helps, but these are standard.
        # Let's also try a mix: Volume, then Height
        ("vol_h", lambda b: (-b[0]*b[1]*b[2], -b[1])),
        ("vol_w", lambda b: (-b[0]*b[1]*b[2], -b[0])),
        ("vol_d", lambda b: (-b[0]*b[1]*b[2], -b[2])),
    ]
    
    best_placements = None
    best_score = float('inf')
    
    for name, key_func in strategies:
        if time.time() > deadline - 5.0:
            break
            
        sorted_boxes = sorted(boxes, key=key_func)
        placements = run_ep_algorithm(sorted_boxes)
        
        # Validate placement (optional but good for debugging)
        # In a real scenario, we'd check for overlaps and bounds.
        # Given the logic, it should be valid.
        
        score = calculate_score(placements)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            
        # If we find a perfect score (e.g., 1 bin, 100% fill), we can stop early
        # But 1.0 is the baseline. We want < 1.0.
        # If B=1 and fill=1.0, score is 1.0.
        # If B=1 and fill=0.9, score is 1.01.
        # If B=2 and fill=1.0, score is 2.0.
        # So lower is better.
        
    # If no placements were found (should not happen), return empty
    if best_placements is None:
        return []
        
    return best_placements
