import numpy as np
import time
from math import gcd
from itertools import combinations

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 625.0  # safety margin
    
    # Try 7x8 grid (56 points), remove 2
    nx, ny = 7, 8
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_pts = len(base_points)
    
    # Generate candidate lines
    candidates = set()
    
    # Vertical
    for c in range(nx):
        candidates.add((1, 0, -c))
        
    # Horizontal
    for c in range(ny):
        candidates.add((0, 1, -c))
        
    # Diagonals - expand range to include more slopes
    for dx in range(1, nx + 1):
        for dy in range(-(ny - 1), ny):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            
            # A=dy, B=-dx
            # C = dx*y - dy*x
            C_vals = dx * base_points[:, 1] - dy * base_points[:, 0]
            vals = np.unique(C_vals)
            for C in vals:
                candidates.add((dy, -dx, C))
                
    # Also try lines with larger slopes that might still hit multiple points
    for dx in range(1, nx + 2):
        for dy in range(-(ny + 1), ny + 1):
            if dy == 0:
                continue
            if gcd(dx, abs(dy)) != 1:
                continue
            if dx > nx or abs(dy) >= ny:
                # Only include if it can still pass through at least 2 grid points
                # A line with slope -dy/dx through (x0,y0) hits (x0+dx, y0-dy)
                # For it to hit 2 points in [0,nx) x [0,ny):
                # need 0 <= x0 < nx-dx and 0 <= y0-dy < ny (if dy>0) or similar
                pass
            
            # A=dy, B=-dx
            C_vals = dx * base_points[:, 1] - dy * base_points[:, 0]
            vals = np.unique(C_vals)
            for C in vals:
                candidates.add((dy, -dx, C))
                
    # Normalize lines
    def normalize_line(A, B, C):
        if A == 0 and B == 0:
            return None
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
        
    unique_lines = []
    for line in candidates:
        norm = normalize_line(*line)
        if norm:
            unique_lines.append(norm)
            
    unique_lines = np.array(unique_lines, dtype=np.int64)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Precompute A, B, C arrays
    A = unique_lines[:, 0]
    B = unique_lines[:, 1]
    C = unique_lines[:, 2]
    
    # For each point, precompute which lines it lies on
    # point_line_incidence[k, m] = 1 if point k lies on line m
    point_line_incidence = np.zeros((n_pts, len(unique_lines)), dtype=np.int8)
    for k in range(n_pts):
        x_k = base_points[k, 0]
        y_k = base_points[k, 1]
        vals = A * x_k + B * y_k + C
        point_line_incidence[k, vals == 0] = 1
    
    # Iterate over pairs of points to remove
    for i, j in combinations(range(n_pts), 2):
        if time.time() > deadline:
            break
        
        # Incidences for each line = sum over active points
        # = total_incidence - point_line_incidence[i] - point_line_incidence[j]
        total_incidence = point_line_incidence.sum(axis=0)
        line_incidents = total_incidence - point_line_incidence[i] - point_line_incidence[j]
        
        # Select top 27
        if len(line_incidents) >= 27:
            top_indices = np.argsort(-line_incidents)[:27]
            score = int(np.sum(line_incidents[top_indices]))
        else:
            score = int(np.sum(line_incidents))
            top_indices = np.arange(len(line_incidents))
            
        if score > best_score:
            best_score = score
            mask = np.ones(n_pts, dtype=bool)
            mask[i] = False
            mask[j] = False
            best_points = base_points[mask].copy()
            best_lines = unique_lines[top_indices].copy()
            
    # Fallback if no pair found or time out
    if best_points is None:
        best_points = base_points[:54]
        A = unique_lines[:, 0]
        B = unique_lines[:, 1]
        C = unique_lines[:, 2]
        x = best_points[:, 0]
        y = best_points[:, 1]
        vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
        incidences = np.sum(vals == 0, axis=1)
        top_indices = np.argsort(-incidences)[:27]
        best_lines = unique_lines[top_indices]
    
    # Local refinement: try moving individual points to nearby positions
    # to increase incidence count with the fixed set of 27 lines
    A_l = best_lines[:, 0]
    B_l = best_lines[:, 1]
    C_l = best_lines[:, 2]
    
    def count_incidence(pt, lines_A, lines_B, lines_C):
        return int(np.sum(lines_A * pt[0] + lines_B * pt[1] + lines_C == 0))
    
    # Compute current total
    current_total = sum(count_incidence(best_points[k], A_l, B_l, C_l) for k in range(54))
    
    # Try moving each point to nearby positions
    for k in range(54):
        if time.time() > deadline:
            break
        best_x, best_y = best_points[k, 0], best_points[k, 1]
        best_local = current_total - count_incidence(best_points[k], A_l, B_l, C_l)
        
        # Search nearby positions (radius 2)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx_pt = (int(best_x + dx), int(best_y + dy))
                # Check if this position is already occupied by another point
                if any(best_points[j, 0] == nx_pt[0] and best_points[j, 1] == nx_pt[1] for j in range(54) if j != k):
                    continue
                new_local = count_incidence(nx_pt, A_l, B_l, C_l)
                if new_local > best_local:
                    best_local = new_local
                    best_x, best_y = nx_pt
        
        if (best_x, best_y) != (int(best_points[k, 0]), int(best_points[k, 1])):
            old_inc = count_incidence(best_points[k], A_l, B_l, C_l)
            new_inc = count_incidence((best_x, best_y), A_l, B_l, C_l)
            current_total += new_inc - old_inc
            best_points[k] = [best_x, best_y]
    
    # Also try swapping: move one point to a position that was removed
    # The removed points are the 2 from the 56-point grid
    for i, j in combinations(range(n_pts), 2):
        if time.time() > deadline:
            break
        removed = base_points[i]
        removed2 = base_points[j]
        
        # Try adding back removed points and removing current points
        # This is more complex; skip for now to save time
    
    return best_points, best_lines
