import time
import itertools

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Sort boxes by volume descending to place large items first
    # We keep track of original indices to map back to the required output format
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
    placed_boxes = []  # List of (x, y, z, w, h, d)
    
    # Initial candidate point
    candidates = [(0, 0, 0)]
    
    for idx, w_orig, h_orig, d_orig in indexed_boxes:
        if time.time() > deadline:
            break
            
        # Generate unique rotations
        rotations = list(set(itertools.permutations((w_orig, h_orig, d_orig))))
        
        best_pos = None
        best_rot = None
        
        # Sort candidates by z, then y, then x to prefer lower positions
        # This helps in "filling from the bottom up"
        sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
        
        found = False
        for cx, cy, cz in sorted_candidates:
            if time.time() > deadline:
                break
                
            for bw, bh, bd in rotations:
                # Check bounds
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                overlap = False
                for px, py, pz, pw, ph, pd in placed_boxes:
                    # Separation check: if they don't overlap in any dimension, no overlap
                    if not (cx + bw <= px or px + pw <= cx):
                        if not (cy + bh <= py or py + ph <= cy):
                            if not (cz + bd <= pz or pz + pd <= cz):
                                overlap = True
                                break
                if not overlap:
                    best_pos = (cx, cy, cz)
                    best_rot = (bw, bh, bd)
                    found = True
                    break
            if found:
                break
        
        if found:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placements[idx] = (0, bx, by, bz, bw, bh, bd)
            placed_boxes.append((bx, by, bz, bw, bh, bd))
            
            # Generate new candidate points
            # The corners of the newly placed box that might be useful for future boxes
            # We add points adjacent to the box
            new_candidates = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd),
                (bx + bw, by + bh, bz),
                (bx + bw, by, bz + bd),
                (bx, by + bh, bz + bd)
            ]
            
            # Add to candidates if not already present (optional optimization, but set logic is slower)
            # Given small N, we can just append and let duplicates exist or use a set
            # Using a set for candidates is better to avoid explosion
            # But we need to preserve order? No, we sort every time.
            # Let's use a set to store candidates, then convert to list for sorting
            # Actually, maintaining a list and checking existence is O(N).
            # Let's just append. Duplicates are handled by sorting and first-fit.
            # To avoid O(N^2) growth of candidates list, we can prune.
            # But N=60, max candidates ~ 6*60 = 360. This is small.
            candidates.extend(new_candidates)
            
            # Optional: Prune candidates that are out of bounds or blocked?
            # Hard to do efficiently without spatial index.
            # We'll rely on the overlap check to reject them quickly.
            
    return placements
