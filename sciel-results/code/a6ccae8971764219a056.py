# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, top_k=27):
    """
    Count the number of points on each line defined by pairs of points.
    Returns the sum of the counts of the top_k lines.
    Also returns the top_k lines themselves.
    """
    n = len(points)
    line_counts = {}
    
    # Iterate over all pairs to find lines
    # This is O(N^2). For N=54, this is ~1500 pairs.
    # We need to count points on each line.
    # A more efficient way:
    # For each point i, and each point j > i, define line L_ij.
    # Increment count for L_ij.
    # But this only counts pairs. We need total points on the line.
    # If a line has k points, it contributes C(k, 2) pairs.
    # So we can map pair-count to point-count?
    # No, multiple lines can pass through the same pair? No, 2 points define 1 line.
    # So, for each unique line, the number of pairs on it is C(k, 2).
    # We can invert this: k = (1 + sqrt(1 + 8 * pair_count)) / 2.
    # This is exact for integer k.
    
    pair_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients for (x1, y1) and (x2, y2)
            # A*x + B*y + C = 0
            # (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen if valid
                
            key = _normalize_line(A, B, C)
            pair_counts[key] = pair_counts.get(key, 0) + 1
            
    # Convert pair counts to point counts
    line_point_counts = {}
    for key, pc in pair_counts.items():
        # Solve k(k-1)/2 = pc
        # k^2 - k - 2*pc = 0
        # k = (1 + sqrt(1 + 8*pc)) / 2
        disc = 1 + 8 * pc
        sqrt_disc = int(np.sqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k * (k - 1) // 2 == pc:
                line_point_counts[key] = k
            else:
                # Should not happen for integer geometry, but just in case
                line_point_counts[key] = 2 # Fallback
        else:
            # This case implies the pair counts don't form a perfect triangle number,
            # which is impossible if the counts are correct for distinct lines.
            # However, if we have floating point errors or logic bugs, fallback.
            # Actually, if we count pairs correctly, it MUST be a triangle number.
            line_point_counts[key] = 2

    # Get top 27 lines
    # Sort by count descending
    sorted_lines = sorted(line_point_counts.items(), key=lambda item: item[1], reverse=True)
    
    total_inc = 0
    top_lines = []
    for i in range(min(top_k, len(sorted_lines))):
        key, count = sorted_lines[i]
        total_inc += count
        top_lines.append(key)
        
    # If there are fewer than 27 lines with >= 2 points,
    # we might need to add lines with 1 point?
    # The problem says "lines a (27, 3) array".
    # If a line has 1 point, it contributes 1 to the incidence.
    # If we have < 27 lines with >= 2 points, we should fill the rest with any other lines
    # that pass through at least 1 point (to get 1 incidence each) or 0.
    # But any line not passing through a point gives 0.
    # So we should pick lines passing through 1 point if we have space.
    # However, there are infinitely many such lines.
    # Does it matter? Yes, 1 incidence > 0.
    
    # How many lines with >= 2 points do we have?
    num_lines_ge2 = len(sorted_lines)
    
    if num_lines_ge2 < top_k:
        # We need (top_k - num_lines_ge2) more lines.
        # We can pick any lines passing through single points.
        # To be valid, we just need to return 27 unique lines.
        # We can generate arbitrary lines passing through unused points?
        # Or just arbitrary lines?
        # If we pick a line with 0 points, it adds 0.
        # If we pick a line with 1 point, it adds 1.
        # We should maximize this.
        
        # Let's find lines passing through exactly 1 point.
        # This is complex.
        # Alternative: Just return the top lines found, and pad with random valid lines 
        # that pass through at least one point if possible.
        
        # For now, let's assume we have enough lines with >= 2 points.
        # In a dense cluster, we will have many.
        
        # If not, we pad with lines passing through single points.
        # We can construct a line through point p and a random other point q?
        # No, that would make it a 2-point line, which would have been found.
        # So a line through p and a "virtual" point far away?
        # Simplest: Generate random lines and check if they hit any point.
        
        # For the purpose of this optimization, we assume we find enough high-incidence lines.
        # If we don't, the score is just the sum of what we have.
        # But we MUST return 27 lines.
        
        pass # Padding logic handled in return

    return total_inc, top_lines

def _generate_padding_lines(points, existing_keys, num_needed):
    """Generate valid lines that are not in existing_keys."""
    padding = []
    # Try to find lines passing through single points
    # We can just generate random lines and check incidence.
    # To save time, we can use lines like x = c or y = c or x+y=c.
    
    # Let's just use lines x = k, y = k, x+y=k, x-y=k for various k.
    # And check if they are already in existing_keys.
    
    candidates = []
    coords = points
    min_x = np.min(coords[:, 0])
    max_x = np.max(coords[:, 0])
    min_y = np.min(coords[:, 1])
    max_y = np.max(coords[:, 1])
    
    # Range of possible intercepts
    range_x = range(min_x - 5, max_x + 6)
    range_y = range(min_y - 5, max_y + 6)
    range_sum = range(min_x + min_y - 10, max_x + max_y + 11)
    range_diff = range(min_x - max_y - 10, max_x - min_y + 11)
    
    for c in range_x:
        key = _normalize_line(1, 0, -c)
        if key not in existing_keys:
            candidates.append(key)
    for c in range_y:
        key = _normalize_line(0, 1, -c)
        if key not in existing_keys:
            candidates.append(key)
    for c in range_sum:
        key = _normalize_line(1, 1, -c)
        if key not in existing_keys:
            candidates.append(key)
    for c in range_diff:
        key = _normalize_line(1, -1, -c)
        if key not in existing_keys:
            candidates.append(key)
            
    # Randomize and pick
    random.shuffle(candidates)
    for k in candidates:
        if len(padding) >= num_needed:
            break
        padding.append(k)
        
    # If still not enough, generate random garbage lines (score 0)
    while len(padding) < num_needed:
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        C = random.randint(-100, 100)
        key = _normalize_line(A, B, C)
        if key not in existing_keys and key != (0,0,0):
            padding.append(key)
            
    return padding

def _score_and_get_lines(points):
    """
    Compute the score (sum of top 27 line incidences) and the actual 27 lines.
    """
    inc, top_keys = _count_incidences(points, 27)
    
    # We need exactly 27 lines.
    if len(top_keys) < 27:
        num_needed = 27 - len(top_keys)
        padding = _generate_padding_lines(points, set(top_keys), num_needed)
        top_keys.extend(padding)
        
    # Convert keys to array
    lines_arr = np.array(top_keys, dtype=np.int64)
    
    return inc, lines_arr

def _is_valid_points(points):
    """Check for duplicate points."""
    # Convert to set of tuples
    pts = [tuple(p) for p in points]
    if len(set(pts)) != len(pts):
        return False
    return True

def _simulated_annealing(points, start_temp=10.0, end_temp=0.1, steps=5000):
    """
    Optimize the point set using Simulated Annealing.
    """
    best_points = points.copy()
    best_score, best_lines = _score_and_get_lines(points)
    
    current_points = points.copy()
    current_score, _ = _score_and_get_lines(current_points)
    
    n = len(current_points)
    
    # Bounding box for moves
    # Start with a box around current points
    min_x = np.min(current_points[:, 0]) - 5
    max_x = np.max(current_points[:, 0]) + 5
    min_y = np.min(current_points[:, 1]) - 5
    max_y = np.max(current_points[:, 1]) + 5
    
    for step in range(steps):
        # Cool temperature
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = current_points[idx]
        
        # Propose a new position
        # Mix of local and global moves
        if random.random() < 0.8:
            # Local move
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        else:
            # Global move within bounding box
            new_x = random.randint(min_x, max_x)
            new_y = random.randint(min_y, max_y)
            
        # Check bounds (optional, but keeps numbers small)
        # We can expand bounds if we want, but let's keep them fixed for now
        # or update them occasionally.
        
        # Check validity: no duplicate points
        # Create a set of all points except the one being moved
        # For efficiency, we can just check against the list
        # Since N=54, a linear scan is fine.
        
        is_dup = False
        for j in range(n):
            if j == idx:
                continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_dup = True
                break
                
        if is_dup:
            continue
            
        # Apply move
        current_points[idx] = [new_x, new_y]
        
        # Calculate new score
        new_score, _ = _score_and_get_lines(current_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
                best_lines, _ = _score_and_get_lines(best_points)
                # Update bounds if we found a better solution?
                # Let's keep bounds static for stability, or expand slightly.
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            
    return best_points, best_score, best_lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses Simulated Annealing to find a point set that maximizes the number of
    incidences with its top 27 lines.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 590  # Safety margin
    
    # Initial configuration: 9x6 Grid
    # Points: x in {1..9}, y in {1..6}
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Shuffle initial points to avoid symmetry traps?
    # The grid is symmetric. SA should break it.
    
    best_points, best_score, best_lines = _simulated_annealing(points, steps=2000)
    
    # If we have time, do another round with different parameters or restart
    while time.time() - start_time < timeout:
        # Try a different initial structure?
        # Or just continue SA from the best found?
        # Let's restart SA with the best found as the starting point,
        # but with a slightly higher temperature to escape local minima.
        
        # Check time
        if time.time() - start_time > timeout - 60:
            break
            
        # Perturb the best points slightly to create a new start
        perturbed_points = best_points.copy()
        for _ in range(5):
            idx = random.randint(0, 53)
            perturbed_points[idx] += np.array([random.randint(-1, 1), random.randint(-1, 1)])
            
        # Ensure validity
        if not _is_valid_points(perturbed_points):
            continue
            
        # Run SA
        new_best_points, new_best_score, new_best_lines = _simulated_annealing(
            perturbed_points, start_temp=5.0, end_temp=0.1, steps=1000
        )
        
        if new_best_score > best_score:
            best_points = new_best_points
            best_score = new_best_score
            best_lines = new_best_lines
            
    # Final validation and formatting
    # Ensure points are int64
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Double check validity of returned objects
    # 1. Shapes
    assert points_out.shape == (54, 2)
    assert lines_out.shape == (27, 3)
    
    # 2. No duplicate points
    pts_set = set([tuple(p) for p in points_out])
    assert len(pts_set) == 54
    
    # 3. No invalid lines (A=B=0)
    for l in lines_out:
        assert not (l[0] == 0 and l[1] == 0)
        
    # 4. No duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    assert len(lines_set) == 27
    
    return points_out, lines_out

# EVOLVE-BLOCK-END
