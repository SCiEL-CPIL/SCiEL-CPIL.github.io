import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    bins = []
    placements = [None] * n
    
    def is_dominated(point, other_points):
        px, py, pz = point
        for ox, oy, oz in other_points:
            if ox <= px and oy <= py and oz <= pz:
                if ox < px or oy < py or oz < pz:
                    return True
        return False

    def update_skyline(skyline, x, y, z, w, h, d):
        filtered_skyline = []
        for p in skyline:
            px, py, pz = p
            if not (x <= px < x + w and y <= py < y + h and z <= pz < z + d):
                filtered_skyline.append(p)
        
        candidates = list(filtered_skyline)
        candidates.append((x + w, y, z))
        candidates.append((x, y + h, z))
        candidates.append((x, y, z + d))
        
        final_skyline = []
        for p in candidates:
            if not is_dominated(p, candidates):
                final_skyline.append(p)
                
        return final_skyline

    def can_place(bin_data, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        
        placed_boxes = bin_data['boxes']
        for box in placed_boxes:
            px, py, pz, pw, ph, pd = box[:6]
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    for orig_idx, w, h, d in indexed_boxes:
        best_bin_idx = -1
        best_pos = None
        best_score = None
        
        for bin_idx, bin_data in enumerate(bins):
            skyline = bin_data['skyline']
            for (sx, sy, sz) in skyline:
                if can_place(bin_data, sx, sy, sz, w, h, d):
                    score = (sz, sy, sx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_bin_idx = bin_idx
                        best_pos = (sx, sy, sz)
                        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx]['boxes'].append((x, y, z, w, h, d, orig_idx))
            bins[best_bin_idx]['skyline'] = update_skyline(bins[best_bin_idx]['skyline'], x, y, z, w, h, d)
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            new_bin = {
                'boxes': [(0, 0, 0, w, h, d, orig_idx)],
                'skyline': update_skyline([(0, 0, 0)], 0, 0, 0, w, h, d)
            }
            bins.append(new_bin)
            placements[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            
    return placements
