"""
Program to construct 30 distinct integer coordinate pairs in [-30,30]^2
maximizing the largest multiplicity of a single positive doubled area.
"""

import time
import itertools
import numpy as np
from collections import Counter
import random

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _compute_areas(points):
    """
    Compute all doubled areas for the given points.
    Returns a list of (i, j, k, area) tuples for all i<j<k with area>0.
    """
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Precompute all triples
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    # Filter out zero areas
    positive_mask = areas > 0
    i_pos = i_idx[positive_mask]
    j_pos = j_idx[positive_mask]
    k_pos = k_idx[positive_mask]
    a_pos = areas[positive_mask]
    
    return i_pos, j_pos, k_pos, a_pos

def _score(points):
    """
    Compute the maximum multiplicity of a single positive doubled area.
    """
    if not _valid(points):
        return 0
    
    i_pos, j_pos, k_pos, a_pos = _compute_areas(points)
    
    if len(a_pos) == 0:
        return 0
        
    # Count frequencies
    unique_areas, counts = np.unique(a_pos, return_counts=True)
    max_mult = np.max(counts)
    
    return int(max_mult)

def _fast_score(points, old_point_idx, old_point, new_point, prev_areas):
    """
    Fast incremental scoring when one point changes.
    prev_areas: the previous area array and indices from _compute_areas
    """
    i_pos, j_pos, k_pos, a_pos = prev_areas
    
    # Find all triples involving old_point_idx
    old_i = old_point_idx
    old_x, old_y = old_point
    new_x, new_y = new_point
    
    # We need to recompute areas for all triples where old_point_idx is i, j, or k
    # And remove triples that become zero, add new ones
    
    # This is complex to do incrementally. Let's use a simpler approach:
    # Just recompute all areas but use the fact that most don't change.
    # Actually, for 30 points, full recompute is fast enough with NumPy.
    # The key insight from knowledge: incremental scoring reduces per-move cost.
    
    # Let me implement a proper incremental version:
    # Precompute which triples involve each point
    
    # For now, use full recompute but optimized
    pts = list(points)
    pts[old_point_idx] = new_point
    pts_arr = np.array(pts, dtype=np.int64)
    
    n = len(pts_arr)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    xi = pts_arr[i_idx, 0]
    yi = pts_arr[i_idx, 1]
    xj = pts_arr[j_idx, 0]
    yj = pts_arr[j_idx, 1]
    xk = pts_arr[k_idx, 0]
    yk = pts_arr[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    positive_areas = areas[areas > 0]
    
    if len(positive_areas) == 0:
        return 0, (i_idx, j_idx, k_idx, positive_areas)
        
    unique_areas, counts = np.unique(positive_areas, return_counts=True)
    max_mult = int(np.max(counts))
    
    # Return new areas for next incremental step
    pos_mask = areas > 0
    return max_mult, (i_idx[pos_mask], j_idx[pos_mask], k_idx[pos_mask], positive_areas)

def _generate_candidate(type_id, params=None):
    """
    Generate structured candidates known to have high symmetry.
    """
    if params is None:
        params = {}
        
    points = []
    
    if type_id == 'regular_polygon_concentric':
        # Concentric regular polygons
        # e.g., 1 ring of 15, or 2 rings of 10, 3 rings of 10, etc.
        # We need 30 points total.
        # Strategy: Try rings of size 10, 10, 10 or 15, 15.
        ring_sizes = params.get('ring_sizes', [10, 10, 10])
        radii = params.get('radii', [10, 20, 30])
        
        for r, size in zip(radii, ring_sizes):
            for k in range(size):
                angle = 2 * np.pi * k / size
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'triangular_lattice':
        # Select 30 points from a triangular lattice
        # Triangular lattice points: (i, 0) and (i+0.5, h)
        # Let's use a grid of (i, j) and map to triangular coords
        # Or simpler: generate a large triangular grid and pick the center 30.
        pts = []
        # Generate points in a hexagonal-ish shape
        for x in range(-20, 21):
            for y in range(-20, 21):
                # Check if in triangular lattice?
                # Actually, let's just use a standard grid but shifted
                # or use a specific lattice generator.
                pass
        
        # Simpler: Use a 6x5 grid but with triangular offsets
        # Row 0: y=0, x = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 (12 pts)
        # Row 1: y=h, x = -9, -7, ... (11 pts)
        # This is getting complex. Let's use a known good lattice:
        # Eisenstein integers lattice points.
        # z = i + j * w where w = (-1 + i*sqrt(3))/2
        # We want integer coordinates in [-30, 30].
        
        lattice_pts = []
        for i in range(-15, 16):
            for j in range(-15, 16):
                # Approximate triangular lattice
                x = i + j // 2
                y = j * 3 // 2 # Scale y to keep aspect ratio reasonable for int grid
                # Better:
                # x = i + j
                # y = j
                # This is just a sheared square grid, which is equivalent to square grid.
                
                # Let's use the standard triangular lattice vectors:
                # v1 = (1, 0)
                # v2 = (0.5, sqrt(3)/2) ~ (0.5, 0.866)
                # To keep integer coords, we can use:
                # P = i * (2, 0) + j * (1, 3)  -- This is a distorted triangle
                # Let's try a subset of the integer grid that forms a nice pattern.
                
                # Actually, the "Regular Polygon" approach is often better for high multiplicity.
                # Let's add a "Hexagonal Lattice" generator.
                pass
        
        # Generate a hexagonal lattice of side length 3 (30 points)
        # Center at (0,0).
        # Rows of length 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 -> Sum = 36 (too many)
        # Rows: 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 is 36.
        # Remove 6 corner points to get 30?
        # Or use a 5x6 grid of triangular lattice?
        
        # Let's use a simpler structured grid:
        # A 5x6 grid where every other row is shifted by 0.5 (scaled to 1).
        # y = 0, 2, 4, 6, 8
        # x = 0, 2, 4, 6, 8, 10
        # Shift row 1, 3 by 1.
        # This creates a "brick wall" or "triangular" feel.
        
        pts = []
        for r in range(5):
            y = (r - 2) * 4 # -8, -4, 0, 4, 8
            for c in range(6):
                x = (c - 2.5) * 4 # -10, -6, -2, 2, 6, 10
                if r % 2 == 1:
                    x += 2 # Shift by 1 unit in x (which is 2 in our scaled space? No, 2/4 = 0.5 shift)
                    # If x step is 4, shift by 2 is half step.
                pts.append((int(x), y))
        points = pts

    elif type_id == 'circle_grid':
        # Points on concentric circles
        # 10 points on r=10, 10 on r=20, 10 on r=30
        for r, count in [(10, 10), (20, 10), (30, 10)]:
            for k in range(count):
                angle = 2 * np.pi * k / count
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'square_grid':
        # Standard 5x6 grid
        pts = []
        for x in range(-10, 11, 4):
            for y in range(-8, 9, 4):
                pts.append((x, y))
        points = pts

    return points

def _local_search(points, time_limit=10.0):
    """
    Hill climbing local search with larger neighborhoods and swap operations.
    """
    start_time = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    
    if best_score == 0:
        return best_points
    
    # Precompute some constants
    occupied = set(best_points)
    n = len(best_points)
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < time_limit:
        improved = False
        move_type = random.random()
        
        if move_type < 0.7:
            # Single point move with larger neighborhood
            random_idx = random.randint(0, n - 1)
            px, py = best_points[random_idx]
            
            # Try multiple random neighbor positions with larger range
            for _ in range(30):
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)
                nx, ny = px + dx, py + dy
                
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                
                # Apply move
                old_point = best_points[random_idx]
                best_points[random_idx] = new_point
                occupied.discard(old_point)
                occupied.add(new_point)
                
                s = _score(best_points)
                
                if s > best_score:
                    best_score = s
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    best_points[random_idx] = old_point
                    occupied.discard(new_point)
                    occupied.add(old_point)
                    
        elif move_type < 0.9:
            # Swap two points
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            if idx1 == idx2:
                continue
            
            p1 = best_points[idx1]
            p2 = best_points[idx2]
            
            # Try swapping positions
            best_points[idx1] = p2
            best_points[idx2] = p1
            
            s = _score(best_points)
            
            if s > best_score:
                best_score = s
                improved = True
                no_improve_count = 0
            else:
                # Revert
                best_points[idx1] = p1
                best_points[idx2] = p2
        
        else:
            # Large jump for one point
            random_idx = random.randint(0, n - 1)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            new_point = (nx, ny)
            
            if new_point not in occupied:
                old_point = best_points[random_idx]
                best_points[random_idx] = new_point
                occupied.discard(old_point)
                occupied.add(new_point)
                
                s = _score(best_points)
                
                if s > best_score:
                    best_score = s
                    improved = True
                    no_improve_count = 0
                else:
                    best_points[random_idx] = old_point
                    occupied.discard(new_point)
                    occupied.add(old_point)
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a few more aggressive moves before giving up
                for _ in range(5):
                    random_idx = random.randint(0, n - 1)
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_point = (nx, ny)
                    
                    if new_point not in occupied:
                        old_point = best_points[random_idx]
                        best_points[random_idx] = new_point
                        occupied.discard(old_point)
                        occupied.add(new_point)
                        
                        s = _score(best_points)
                        
                        if s > best_score:
                            best_score = s
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            best_points[random_idx] = old_point
                            occupied.discard(new_point)
                            occupied.add(old_point)
                
                if not improved:
                    break  # Local optimum found
                    
    return best_points

def construct():
    """
    Construct 30 points maximizing the multiplicity of a single doubled area.
    """
    start_time = time.time()
    deadline = start_time + 85 # Leave 5s margin
    
    candidates = []
    
    # 1. Regular Polygons / Concentric Rings
    # Try various ring configurations
    # 15+15
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [15, 15], 'radii': [15, 30]}))
    # 10+10+10
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [10, 10, 10], 'radii': [10, 20, 30]}))
    # 6+6+6+6+6 (Pentagonal rings? No, hexagonal)
    # 5x6 Grid (Rectangular)
    candidates.append(_generate_candidate('square_grid'))
    # Triangular-ish Grid
    candidates.append(_generate_candidate('triangular_lattice'))
    # Circle Grid (Concentric)
    candidates.append(_generate_candidate('circle_grid'))
    
    # Add some random symmetric configurations
    for _ in range(5):
        # Random regular polygon subset?
        # Just use random points for diversity
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        candidates.append(pts)
        
    best_points = None
    best_score = 0
    
    # Score all candidates
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    if best_points is None:
        # Fallback
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Run local search on the best candidate
    # Allocate time based on remaining budget
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run local search with the majority of the time
        search_time = min(remaining_time - 2, 70) # Cap at 70s for search
        best_points = _local_search(best_points, time_limit=search_time)
        best_score = _score(best_points)
    
    # If we have significant time left, try local search on other candidates
    if remaining_time > 20:
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            if not _valid(cand):
                continue
            if cand == best_points:
                continue
            s = _score(cand)
            if s >= best_score - 2:  # Close to best, worth exploring
                remaining = deadline - time.time() - 3
                if remaining > 5:
                    cand_search = _local_search(cand, time_limit=min(remaining, 15))
                    s2 = _score(cand_search)
                    if s2 > best_score:
                        best_score = s2
                        best_points = cand_search
    
    # Final check
    if not _valid(best_points):
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points

# EVOLVE-BLOCK-END
