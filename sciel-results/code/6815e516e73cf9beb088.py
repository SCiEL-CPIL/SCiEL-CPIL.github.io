"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np

def _compute_score(points):
    """Compute the largest multiplicity of a single positive doubled area using vectorized NumPy."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all determinants vectorized
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorize over k > j
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3s = pts[j+1:, 0]
            y3s = pts[j+1:, 1]
            dets = np.abs(x1 * (y2 - y3s) + x2 * (y3s - y1) + x3s * (y1 - y2))
            mask = dets > 0
            if np.any(mask):
                areas.extend(dets[mask].tolist())
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _inbounds(points, limit=30):
    for x, y in points:
        if x < -limit or x > limit or y < -limit or y > limit:
            return False
    return True

def _build_grid(w, h, x0=0, y0=0):
    return [(x0 + x, y0 + y) for x in range(w) for y in range(h)]

def _build_lattice(w, h, x0=0, y0=0, dx=1, dy=1, skew=0):
    """Build a lattice/parallelogram pattern."""
    return [(x0 + x * dx + y * skew, y0 + y * dy) for x in range(w) for y in range(h)]

def _local_improve(points, max_iters=2000, timeout=45):
    """Local improvement via point moves with neighborhood search."""
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    n = len(best_points)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Try moving a random point to a nearby location (local move)
        idx = random.randint(0, n - 1)
        old_point = best_points[idx]
        ox, oy = old_point
        # Small perturbation
        new_x = ox + random.randint(-3, 3)
        new_y = oy + random.randint(-3, 3)
        new_point = (new_x, new_y)
        
        # Check bounds and uniqueness
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        if new_point in best_points:
            continue
        
        best_points[idx] = new_point
        new_score = _compute_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
        else:
            best_points[idx] = old_point
    
    return best_points, best_score

def _local_improve_global(points, max_iters=500, timeout=20):
    """Local improvement with occasional global jumps."""
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    n = len(best_points)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        if random.random() < 0.7:
            # Local move
            idx = random.randint(0, n - 1)
            old_point = best_points[idx]
            ox, oy = old_point
            new_x = ox + random.randint(-5, 5)
            new_y = oy + random.randint(-5, 5)
            new_point = (new_x, new_y)
        else:
            # Global jump
            idx = random.randint(0, n - 1)
            old_point = best_points[idx]
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        if new_point in best_points:
            continue
        
        best_points[idx] = new_point
        new_score = _compute_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
        else:
            best_points[idx] = old_point
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin
    
    best_points = None
    best_score = 0
    
    # Try structured configurations
    grid_configs = [
        (6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15),
        (30, 1), (1, 30), (5, 5)
    ]
    
    # Also try parallelogram/lattice patterns
    for w, h in grid_configs:
        if time.time() > deadline - 5:
            break
        for _ in range(2):
            if time.time() > deadline - 5:
                break
            points = _generate_candidate(w, h)
            if not _inbounds(points):
                continue
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = list(points)
    
    # Try lattice patterns with various parameters
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2)]:
        if time.time() > deadline - 5:
            break
        for dx in [1, 2]:
            for dy in [1, 2]:
                for skew in [0, 1, -1]:
                    if time.time() > deadline - 5:
                        break
                    x0 = random.randint(-25, 25)
                    y0 = random.randint(-25, 25)
                    points = _build_lattice(w, h, x0, y0, dx, dy, skew)
                    if not _inbounds(points):
                        continue
                    # Check distinctness
                    if len(set(points)) != 30:
                        continue
                    score = _compute_score(points)
                    if score > best_score:
                        best_score = score
                        best_points = list(points)
    
    # Local improvement from best structured config
    if best_points is not None:
        if time.time() < deadline - 15:
            improved_points, improved_score = _local_improve(
                best_points, max_iters=500, timeout=10
            )
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
        
        if time.time() < deadline - 8:
            improved_points, improved_score = _local_improve_global(
                best_points, max_iters=300, timeout=7
            )
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Try random starts with local improvement
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            points = []
            used = set()
            while len(points) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    points.append((x, y))
                    used.add((x, y))
            
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = list(points)
            
            if time.time() < deadline - 5:
                improved_points, improved_score = _local_improve(
                    points, max_iters=300, timeout=8
                )
                if improved_score > best_score:
                    best_score = improved_score
                    best_points = improved_points
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points

def _generate_candidate(w, h, x0=None, y0=None):
    """Generate a grid candidate."""
    if x0 is None:
        x0 = random.randint(-30, 30 - w + 1)
    if y0 is None:
        y0 = random.randint(-30, 30 - h + 1)
    return _build_grid(w, h, x0, y0)
# EVOLVE-BLOCK-END
