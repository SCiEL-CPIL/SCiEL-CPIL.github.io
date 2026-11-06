"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 135.0 # Safety margin
    
    # 1. Preprocess: Generate rotations and sort by volume descending
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        # Generate unique rotations
        rots = [(w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)]
        # Deduplicate rotations
        unique_rots = list(set(rots))
        # Store (index, volume, rotations)
        vol = w * h * d
        box_data.append((i, vol, unique_rots))
        
    # Sort by volume descending (Largest First)
    box_data.sort(key=lambda x: x[1], reverse=True)
    
    # 2. State
    placed = [] # List of (x, y, z, w, h, d)
    candidates = [(0, 0, 0)] # List of candidate (x, y, z) points
    
    results = [-1] * n # Initialize all as left out
    # We need to map back to original indices
    
    for idx, vol, rots in box_data:
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Check if rotation fits in container at all
            if rw > W or rh > H or rd > D:
                continue
                
            # Try each candidate position
            for cx, cy, cz in candidates:
                # Check bounds
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                    
                # Check overlap with all placed boxes
                valid = True
                for px, py, pz, pw, ph, pd in placed:
                    # If intervals overlap in all 3 dims, it's a collision
                    if not (cx + rw <= px or px + pw <= cx or 
                            cy + rh <= py or py + ph <= cy or 
                            cz + rd <= pz or pz + pd <= cz):
                        valid = False
                        break
                
                if valid:
                    # Score: prefer lower x, then lower y, then lower z (Bottom-Left-Back)
                    # Using a weighted sum to prioritize Z (height) most, then Y, then X
                    # Or simply lexicographical ordering via tuple comparison
                    score = (cz, cy, cx)
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
                        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            # Record placement
            placed.append((bx, by, bz, bw, bh, bd))
            results[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidate points
            # The new "corners" created by this placement
            new_candidates = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd)
            ]
            # Add to candidates if not already present (optional optimization, but list is small)
            for nc in new_candidates:
                if nc not in candidates:
                    candidates.append(nc)
                    
            # Time check
            if time.time() > deadline:
                break
        else:
            # Box could not be placed
            results[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            
    return results
# EVOLVE-BLOCK-END
