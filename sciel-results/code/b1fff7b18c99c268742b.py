"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    def get_triangle_areas(points):
        """
        Compute a Counter of absolute determinants for all triples.
        points: list of (x, y) tuples
        """
        n = len(points)
        counts = Counter()
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] += 1
        return counts

    def evaluate(points):
        """
        Compute the maximum multiplicity of any positive doubled area.
        """
        counts = get_triangle_areas(points)
        if not counts:
            return 0
        return max(counts.values())

    # Initial configuration: two horizontal lines y=0, y=1
    # 15 points on each, x in 0..14
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)
    
    # Try a few other grid configurations to see if any beat 420
    grid_configs = [
        (5, 6), (6, 5), (10, 3), (3, 10), (15, 2), (2, 15)
    ]
    for w, h in grid_configs:
        if w * h != 30:
            continue
        pts = [(x, y) for x in range(w) for y in range(h)]
        # Center the grid slightly to allow more room for perturbations?
        # Or just keep it in the corner. The range is [-30,30].
        # Current pts are in [0, w) x [0, h).
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    # Local Search with efficient update
    # We maintain a global counter of areas.
    # When moving point i, we subtract areas of triangles involving i,
    # move i, then add new areas.
    
    # Helper to get areas of triangles involving a specific index
    def get_areas_involving(points, idx):
        """
        Returns a list of (area, count) for triangles involving points[idx].
        Actually, just returns a Counter of areas for triangles involving idx.
        """
        n = len(points)
        counts = Counter()
        x1, y1 = points[idx]
        for i in range(n):
            if i == idx:
                continue
            x2, y2 = points[i]
            for j in range(i + 1, n):
                if j == idx:
                    continue
                x3, y3 = points[j]
                # Calculate determinant for (idx, i, j)
                # Note: order doesn't matter for abs det
                dx1 = x2 - x1
                dy1 = y2 - y1
                dx2 = x3 - x1
                dy2 = y3 - y1
                det = abs(dx1 * dy2 - dy1 * dx2)
                if det > 0:
                    counts[det] += 1
        return counts

    # Initialize global counter for best_points
    global_counts = get_triangle_areas(best_points)
    current_score = best_score
    
    # Simulated Annealing / Hill Climbing
    # We try to move individual points to small neighboring positions.
    # Since the grid is small, we can try all valid moves for a point.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order to avoid bias
        indices = list(range(30))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            
            # Try all possible moves in a 5x5 neighborhood
            found_better = False
            best_local_score = current_score
            best_local_points = None
            best_local_counts = None
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    
                    # Check collision
                    new_point = (new_x, new_y)
                    if new_point in best_points:
                        continue
                    
                    # Update points
                    old_point = best_points[idx]
                    best_points[idx] = new_point
                    
                    # Update counts
                    # 1. Remove old areas involving idx
                    old_areas = get_areas_involving(best_points, idx)
                    # Wait, we just changed best_points[idx] to new_point.
                    # We need to remove areas based on OLD point.
                    # Let's revert for a moment or calculate carefully.
                    
                    # Better approach:
                    # 1. Calculate old_areas using OLD point.
                    # 2. Subtract from global_counts.
                    # 3. Change point to NEW.
                    # 4. Calculate new_areas using NEW point.
                    # 5. Add to global_counts.
                    # 6. Evaluate score.
                    
                    # Revert to old to calculate old_areas
                    best_points[idx] = old_point
                    old_areas = get_areas_involving(best_points, idx)
                    for area, count in old_areas.items():
                        global_counts[area] -= count
                        if global_counts[area] == 0:
                            del global_counts[area]
                    
                    # Apply new point
                    best_points[idx] = new_point
                    new_areas = get_areas_involving(best_points, idx)
                    for area, count in new_areas.items():
                        global_counts[area] += count
                    
                    # Check if valid (no zero area issues? No, we just exclude 0)
                    # Calculate new max score
                    if not global_counts:
                        new_score = 0
                    else:
                        new_score = max(global_counts.values())
                    
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_points = list(best_points)
                        best_local_counts = global_counts.copy()
                        # Don't break, keep looking for better in this neighborhood?
                        # Or break to accept first improvement?
                        # Let's keep looking to find the best move for this idx.
                        
            if best_local_points:
                # Accept the best move found for this idx
                best_points = best_local_points
                global_counts = best_local_counts
                current_score = best_local_score
                if current_score > best_score:
                    best_score = current_score
                improved = True
                # If we found a global improvement, we might want to restart the loop
                # or continue. Continuing is fine.
                
            # If no move improved, best_points is already reverted to old state
            # because we reverted inside the loop before calculating old_areas.
            # Wait, if best_local_points is None, we reverted to old_point in the last iteration?
            # No, if no move was found, we reverted to old_point in the loop.
            # But if a move was found, best_points was changed to new_point in the loop.
            # We need to ensure best_points is correct at the end of the idx loop.
            
            # If best_local_points was found, we set best_points = best_local_points.
            # If not, best_points should be the original old_point.
            # In the loop, we set best_points[idx] = new_point for evaluation.
            # If we don't accept it, we need to revert it.
            # The logic above:
            # 1. best_points[idx] = old_point (revert for calc)
            # 2. best_points[idx] = new_point (apply for calc)
            # So at the end of the inner loop, best_points[idx] is new_point.
            # If we don't accept, we must revert it back to old_point.
            
            if not best_local_points:
                best_points[idx] = old_point
            # else best_points is already set to best_local_points (which is a copy)
            # Wait, best_local_points is a copy. best_points is the live list.
            # If we accept, we do best_points = best_local_points.
            # If we don't, we must revert best_points[idx] to old_point.
            # The code above `if not best_local_points: best_points[idx] = old_point` handles this.
            
    return best_points
# EVOLVE-BLOCK-END
