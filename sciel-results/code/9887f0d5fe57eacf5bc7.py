import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    W, H, D = container
    
    # 1. Preprocess boxes: generate unique rotations
    # A box is defined by (w, h, d).
    # Rotations are permutations of (w, h, d).
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(perm)
        box_rotations.append(list(rots))
    
    # 2. Sort boxes by volume descending
    # We want to place large boxes first
    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # 3. Greedy placement using Extremal Points
    placed = []  # List of (x, y, z, w, h, d, original_index)
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
    
    # Extremal points are candidate positions for the bottom-front-left corner of a new box.
    # We start with (0, 0, 0).
    # When a box is placed at (x,y,z) with size (w,h,d), we add its 8 corners to the candidate list.
    # To avoid duplicates and invalid points, we check if a point is "inside" any placed box before adding,
    # or just keep a set and filter during placement.
    # Given N=60, keeping a list of all corners is fine.
    
    extremal_points = [(0, 0, 0)]
    
    for idx in indices:
        w_orig, h_orig, d_orig = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try all rotations for this box
        for rw, rh, rd in box_rotations[idx]:
            # Check if rotation fits in container dimensions
            if rw > W or rh > H or rd > D:
                continue
                
            # Try all extremal points
            for px, py, pz in extremal_points:
                nx, ny, nz = px, py, pz
                
                # Check bounds
                if nx + rw > W or ny + rh > H or nz + rd > D:
                    continue
                
                # Check collision with all placed boxes
                collision = False
                for bx, by, bz, bw, bh, bd, _ in placed:
                    # Overlap check:
                    # [nx, nx+rw) vs [bx, bx+bw)
                    if not (nx + rw <= bx or bx + bw <= nx):
                        if not (ny + rh <= by or by + bh <= ny):
                            if not (nz + rd <= bz or bz + bd <= nz):
                                collision = True
                                break
                if collision:
                    continue
                
                # Valid placement found. Calculate score.
                # Heuristic: Minimize height (z), then depth (y), then width (x).
                # This tends to keep the pack flat and compact.
                score = (nz, ny, nx)
                if score < best_score:
                    best_score = score
                    best_pos = (nx, ny, nz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            nx, ny, nz = best_pos
            rw, rh, rd = best_rot
            placed.append((nx, ny, nz, rw, rh, rd, idx))
            placements[idx] = (0, nx, ny, nz, rw, rh, rd)
            
            # Generate new extremal points from the 8 corners of the placed box
            # We only add points that are "exposed" or at least not inside another box.
            # For efficiency with small N, we can just add all 8 corners.
            # We should avoid adding points that are strictly dominated?
            # Actually, just adding all corners works well enough for N=60.
            corners = [
                (nx, ny, nz),
                (nx + rw, ny, nz),
                (nx, ny + rh, nz),
                (nx + rw, ny + rh, nz),
                (nx, ny, nz + rd),
                (nx + rw, ny, nz + rd),
                (nx, ny + rh, nz + rd),
                (nx + rw, ny + rh, nz + rd)
            ]
            
            # Optional: Filter corners that are inside other placed boxes to keep list small
            # But checking inside is O(N). We are already O(N) per placement for collision.
            # Let's just add them. Duplicates in the list just cause redundant checks.
            # To avoid list explosion, we can use a set for extremal points?
            # No, list is fine. 60 boxes * 8 corners = 480 points max.
            extremal_points.extend(corners)
            
            # Check time
            if time.time() > deadline:
                break
        else:
            # Box could not be placed
            placements[idx] = (-1, 0, 0, 0, 0, 0, 0)
            if time.time() > deadline:
                break

    return placements
