import time

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Create list of (box_dims, original_index)
    indexed_boxes = [(b, i) for i, b in enumerate(boxes)]
    
    # Sort by volume descending, then height, then depth, then width
    # This is a heuristic to place large/important boxes first
    indexed_boxes.sort(key=lambda x: (x[0][0]*x[0][1]*x[0][2], x[0][1], x[0][2], x[0][0]), reverse=True)
    
    # Bins structure
    # Each bin is a dict: {'boxes': [(w,h,d,x,y,z,orig_idx)], 'eps': [(x,y,z)]}
    bins = []
    
    def is_feasible(x, y, z, w, h, d, bin_boxes):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bw, bh, bd, bx, by, bz, _ in bin_boxes:
            # Check overlap
            if x < bx + bw and x + w > bx and \
               y < by + bh and y + h > by and \
               z < bz + bd and z + d > bz:
                return False
        return True

    def update_bin_eps(bin_data, x, y, z, w, h, d):
        new_corners = [
            (x + w, y, z),
            (x, y + h, z),
            (x, y, z + d),
            (x + w, y + h, z),
            (x, y + h, z + d),
            (x + w, y, z + d),
            (x + w, y + h, z + d)
        ]
        
        candidates = bin_data['eps'] + new_corners
        
        # Filter candidates
        valid_candidates = []
        seen = set()
        for p in candidates:
            if p in seen: continue
            seen.add(p)
            px, py, pz = p
            # Check if strictly inside any box in this bin
            inside = False
            for bw, bh, bd, bx, by, bz, _ in bin_data['boxes']:
                if bx < px < bx + bw and by < py < by + bh and bz < pz < bz + bd:
                    inside = True
                    break
            if not inside:
                valid_candidates.append(p)
                
        # Remove dominated points
        # Sort by x, y, z
        valid_candidates.sort()
        non_dominated = []
        for p in valid_candidates:
            # Check if p is dominated by any point in non_dominated
            dominated = False
            for q in non_dominated:
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                    if q != p:
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(p)
                
        # Final check: remove any point that is dominated by another in non_dominated
        # (The above loop might leave some if p dominates q but q was added earlier? 
        #  No, if q was added earlier, q.x <= p.x. If p dominates q, p.x <= q.x, so p.x == q.x.
        #  If p.x == q.x and p.y <= q.y and p.z <= q.z, then p dominates q.
        #  In the loop, when we process p, we check against q.
        #  q.x <= p.x is true. q.y <= p.y? If p.y < q.y, then q does NOT dominate p.
        #  So p is added. But q is dominated by p.
        #  So we need a final pass to remove dominated points from the list.)
        
        final_eps = []
        for p in non_dominated:
            is_dom = False
            for q in non_dominated:
                if p == q: continue
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                    is_dom = True
                    break
            if not is_dom:
                final_eps.append(p)
                
        bin_data['eps'] = final_eps

    # Place boxes
    for box, orig_idx in indexed_boxes:
        w, h, d = box
        placed = False
        
        best_bin_idx = -1
        best_pos = None
        best_cost = float('inf')
        
        for i in range(len(bins)):
            bin_data = bins[i]
            for x, y, z in bin_data['eps']:
                if is_feasible(x, y, z, w, h, d, bin_data['boxes']):
                    # Cost: minimize z, then y, then x
                    cost = z * W * H + y * W + x
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = i
                        best_pos = (x, y, z)
                        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx]['boxes'].append((w, h, d, x, y, z, orig_idx))
            update_bin_eps(bins[best_bin_idx], x, y, z, w, h, d)
        else:
            # New bin
            bins.append({'boxes': [], 'eps': [(0, 0, 0)]})
            bins[-1]['boxes'].append((w, h, d, 0, 0, 0, orig_idx))
            update_bin_eps(bins[-1], 0, 0, 0, w, h, d)

    # Construct result in original order
    placements = [None] * n
    for i, bin_data in enumerate(bins):
        for w, h, d, x, y, z, orig_idx in bin_data['boxes']:
            placements[orig_idx] = (i, x, y, z, w, h, d)
            
    return placements
