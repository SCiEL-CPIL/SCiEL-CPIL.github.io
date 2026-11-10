import numpy as np
from math import gcd
import time
import random

def normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    # Make first non-zero coefficient positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def get_lines_from_points(points):
    """Get all unique lines determined by pairs of points."""
    lines_set = set()
    n_pts = len(points)
    for i in range(n_pts):
        x1, y1 = points[i]
        for j in range(i + 1, n_pts):
            x2, y2 = points[j]
            A = int(y1 - y2)
            B = int(x2 - x1)
            C = int(x1 * y2 - x2 * y1)
            norm = normalize_line(A, B, C)
            if norm != (0, 0, 0):
                lines_set.add(norm)
    return np.array(list(lines_set), dtype=np.int64)

def count_incidences(points, lines):
    """Count total incidences between points and lines."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    x = points[:, 0]
    y = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    # Compute A*x + B*y + C for all pairs
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    inc = np.sum(vals == 0)
    return inc

def select_top_lines(points, k=27):
    """Select the k lines with the highest incidence counts."""
    lines_all = get_lines_from_points(points)
    if len(lines_all) == 0:
        return np.empty((0, 3), dtype=np.int64)
    
    x = points[:, 0]
    y = points[:, 1]
    A = lines_all[:, 0]
    B = lines_all[:, 1]
    C = lines_all[:, 2]
    
    vals = A[:, np.newaxis] * x[np.newaxis, :] + B[:, np.newaxis] * y[np.newaxis, :] + C[:, np.newaxis]
    counts = np.sum(vals == 0, axis=1)
    
    # Sort indices by counts descending
    sorted_indices = np.argsort(counts)[::-1]
    
    # Take top k
    top_indices = sorted_indices[:k]
    selected_lines = lines_all[top_indices]
    
    return selected_lines

def local_search(points, lines, max_iters=50, time_limit=500):
    """Perform local search to improve incidences."""
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = count_incidences(current_points, current_lines)
    
    # Precompute neighbors for point moves
    # We will try moving each point to a nearby integer coordinate
    # To ensure uniqueness, we check against other points
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # 1. Try moving points
        for i in range(len(current_points)):
            x, y = current_points[i]
            best_neighbor = (x, y)
            best_neighbor_score = -1
            
            # Try a small grid around the point
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    # Check if new position is already occupied
                    # This is O(N) for each candidate, which is fine for N=54
                    # We can optimize by maintaining a set of points
                    # But for simplicity and safety, we just check
                    # Actually, let's maintain a set for O(1) lookup
                    pass
            
            # More efficient: Maintain a set of points
            # But we need to update the set when we move a point.
            # Let's stick to a simpler approach:
            # For each point, try moving it.
            # If the move improves the score, do it.
            
            # To speed up, we can just try a few random moves or a fixed set of moves.
            # Let's try moves in a 5x5 neighborhood.
            # We need to ensure the new point is not identical to any other point.
            
            # Let's precompute the set of other points
            # This is getting complex. Let's just do a simple hill climbing.
            
            # Try moving point i to (x+dx, y+dy)
            # Calculate delta in score.
            # Score = sum of incidences.
            # Moving point i affects:
            # 1. Incidences of point i with all lines.
            # 2. Incidences of other points with lines? No, other points don't move.
            # So we only need to check if the new point lies on more lines.
            
            # Let's compute the number of lines the current point lies on.
            # And the number of lines a candidate point lies on.
            
            # Current point incidence:
            # A*x + B*y + C == 0
            
            # We can compute this vectorized for all lines.
            
            # Let's just evaluate the total score for the candidate configuration.
            # This is O(N*M) = 54*27 = 1458 operations.
            # 54 points * 25 neighbors * 1458 = ~2M operations per iteration.
            # 50 iterations = 100M operations. This might be slow in Python.
            
            # Optimization:
            # For a fixed set of lines, the score of a point (x,y) is the number of lines it satisfies.
            # We can precompute for a region of interest.
            
            # Let's try a different approach:
            # Randomly swap two points? No.
            # Randomly move one point?
            
            # Let's just do a simple random restart and grid search for points.
            # The grid search in the original code was limited to rectangular grids.
            # We can allow points to be on a lattice but with some random offsets.
            
            # Actually, the "local search" idea is good if we optimize it.
            # Let's try moving one point at a time to the best nearby position.
            
            # To make it fast, we only check a small number of neighbors.
            # And we only accept if the score strictly increases.
            
            # Let's implement a simple move:
            # Pick a random point.
            # Try moving it to a random nearby integer coordinate (within distance 5).
            # If the new coordinate is not occupied and the score increases, move it.
            
            idx = random.randint(0, len(current_points) - 1)
            x, y = current_points[idx]
            
            # Generate a few random neighbors
            for _ in range(20):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = x + dx, y + dy
                if nx == x and ny == y:
                    continue
                
                # Check if occupied
                # We can check if (nx, ny) is in current_points
                # Using numpy: np.any((current_points[:, 0] == nx) & (current_points[:, 1] == ny))
                # This is O(N).
                # If occupied, skip.
                
                # To make this faster, we can use a set of tuples.
                # But updating the set is overhead.
                
                # Let's just do the check.
                is_occupied = np.any((current_points[:, 0] == nx) & (current_points[:, 1] == ny))
                if is_occupied:
                    continue
                
                # Try the move
                old_point = current_points[idx].copy()
                current_points[idx] = [nx, ny]
                
                new_score = count_incidences(current_points, current_lines)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    current_points[idx] = old_point
        
        # 2. Update lines
        # After moving points, the best lines might change.
        # Recompute top 27 lines.
        new_lines = select_top_lines(current_points, 27)
        if len(new_lines) > 0:
            # Check if the set of lines changed
            # If it did, the score might change because we are now using different lines.
            # Actually, the score is defined by the (points, lines) pair.
            # So we must use the new lines.
            new_score = count_incidences(current_points, new_lines)
            
            # It is possible that the new lines yield a lower score than the old lines
            # with the new points? No, select_top_lines picks the best lines for the points.
            # So new_score should be >= old score with new points and old lines.
            # And since we only moved points if it improved score with old lines,
            # the new score should be >= current_score.
            
            if new_score >= current_score:
                current_lines = new_lines
                current_score = new_score
            else:
                # This shouldn't happen if select_top_lines is correct.
                # But just in case, keep old lines?
                # No, we want the best lines.
                # Maybe the old lines were better?
                # Let's just keep the new lines as they are the top 27.
                current_lines = new_lines
                current_score = new_score
        
        if not improved:
            # Try a few random perturbations to escape local minima
            for _ in range(10):
                idx = random.randint(0, len(current_points) - 1)
                x, y = current_points[idx]
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                nx, ny = x + dx, y + dy
                if nx == x and ny == y:
                    continue
                is_occupied = np.any((current_points[:, 0] == nx) & (current_points[:, 1] == ny))
                if is_occupied:
                    continue
                
                old_point = current_points[idx].copy()
                current_points[idx] = [nx, ny]
                new_score = count_incidences(current_points, current_lines)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_points[idx] = old_point
            
            if not improved:
                break
                
    return current_points, current_lines, current_score

def szemeredi_trotter_incidences():
    best_points = None
    best_lines = None
    best_score = -1
    
    start_time = time.time()
    deadline = 600  # Leave some margin
    
    # Try multiple initializations
    # 1. Rectangular grids
    factors = [1, 2, 3, 6, 9, 18, 27, 54]
    
    # 2. Random lattices
    # 3. Perturbed grids
    
    initializations = []
    
    for W in factors:
        H = 54 // W
        xs = np.arange(W)
        ys = np.arange(H)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        initializations.append(points)
        
    # Add a few random points sets
    for _ in range(5):
        # Random points in a box
        points = np.random.randint(0, 20, (54, 2)).astype(np.int64)
        # Ensure unique
        unique_points = np.unique(points, axis=0)
        if len(unique_points) < 54:
            # Fill with more random points
            while len(unique_points) < 54:
                p = np.random.randint(0, 20, 2).astype(np.int64)
                if not np.any((unique_points[:, 0] == p[0]) & (unique_points[:, 1] == p[1])):
                    unique_points = np.vstack([unique_points, p])
        points = unique_points[:54]
        initializations.append(points)
        
    for init_points in initializations:
        if time.time() - start_time > deadline:
            break
            
        # Select top lines for this initialization
        lines = select_top_lines(init_points, 27)
        
        if len(lines) < 27:
            continue
            
        # Local search
        pts, lns, score = local_search(init_points, lines, max_iters=100, time_limit=60)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lns.copy()
            
    # Fallback
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
        
    return best_points, best_lines
