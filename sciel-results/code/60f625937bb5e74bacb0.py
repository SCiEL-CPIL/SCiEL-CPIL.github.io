"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute doubled areas for all triples
    # For points i, j, k: area2 = |x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)|
    # = |(x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i)|
    
    # Use vectorized approach with bincount for speed
    # We'll iterate over pairs and compute cross products
    # For efficiency, use the fact that area2(i,j,k) = |(p_j - p_i) x (p_k - p_i)|
    
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            for k in range(j+1, n):
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                a = abs(dx * dy2 - dy * dx2)
                if a > 0:
                    areas.append(a)
    
    if not areas:
        return 0
    
    # Find max multiplicity
    area_arr = np.array(areas)
    unique, counts = np.unique(area_arr, return_counts=True)
    return int(counts.max())

def _score_fast(points):
    """Faster scoring using numpy vectorization."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    
    # Collect all doubled areas
    areas_list = []
    for i in range(n):
        pi = pts[i]
        for j in range(i+1, n):
            v1 = pts[j] - pi
            for k in range(j+1, n):
                v2 = pts[k] - pi
                a = abs(v1[0]*v2[1] - v1[1]*v2[0])
                if a > 0:
                    areas_list.append(a)
    
    if not areas_list:
        return 0
    
    area_arr = np.array(areas_list)
    unique, counts = np.unique(area_arr, return_counts=True)
    return int(counts.max())

def _score_vectorized(points):
    """Vectorized scoring for speed."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    
    # Compute all pairwise differences
    # For each pair (i,j), compute cross product with all k > j
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            if dx == 0 and dy == 0:
                continue
            # For all k > j
            if j + 1 < n:
                remaining = pts[j+1:]
                dx2 = remaining[:, 0] - pts[i, 0]
                dy2 = remaining[:, 1] - pts[i, 1]
                cross = np.abs(dx * dy2 - dy * dx2)
                nonzero = cross[cross > 0]
                if len(nonzero) > 0:
                    areas.append(nonzero)
    
    if not areas:
        return 0
    
    all_areas = np.concatenate(areas)
    unique, counts = np.unique(all_areas, return_counts=True)
    return int(counts.max())

def _local_search(points, max_iters=500, time_limit=None):
    """Local search to improve the score."""
    start_time = time.time()
    current_points = list(points)
    current_score = _score_vectorized(current_points)
    
    # Try moving each point to nearby positions
    best_points = list(current_points)
    best_score = current_score
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        improved = False
        for idx in range(len(current_points)):
            x, y = current_points[idx]
            # Try small moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    # Check if point already exists
                    if (nx, ny) in current_points:
                        continue
                    new_points = list(current_points)
                    new_points[idx] = (nx, ny)
                    new_score = _score_vectorized(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_score

def _generate_lattice_base(basis_a, basis_b, origin, count=30):
    """Generate points on a lattice defined by basis vectors."""
    points = set()
    a, b = basis_a, basis_b
    ox, oy = origin
    
    # Generate grid of lattice points
    max_coord = 30
    for i in range(-30, 31):
        for j in range(-30, 31):
            x = ox + i * a[0] + j * b[0]
            y = oy + i * a[1] + j * b[1]
            if -30 <= x <= 30 and -30 <= y <= 30:
                points.add((x, y))
                if len(points) >= count * 3:
                    break
        if len(points) >= count * 3:
            break
    
    return list(points)

def _random_points(count=30):
    """Generate random points."""
    points = set()
    while len(points) < count:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        points.add((x, y))
    return list(points)

def _structured_points():
    """Generate structured points that might have good area multiplicity."""
    # Try a grid-like structure
    points = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            points.append((x*2, y*2))
            if len(points) >= 30:
                return points
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Try lattice-based constructions
    # For equal-area triangles, we want many triples to have the same area.
    # A regular grid can produce many equal areas.
    
    # Try different lattice bases
    lattice_bases = [
        ((1, 0), (0, 1)),      # Standard grid
        ((1, 0), (0, 2)),      # Stretched vertically
        ((2, 0), (0, 1)),      # Stretched horizontally
        ((1, 1), (1, -1)),     # Diagonal
        ((1, 2), (2, 1)),      # Another diagonal
        ((2, 1), (1, 2)),      # Transposed
        ((1, 0), (1, 1)),      # Sheared
        ((2, 0), (0, 2)),      # 2x scaled
        ((3, 0), (0, 1)),      # 3x horizontal
        ((1, 0), (0, 3)),      # 3x vertical
        ((2, 1), (1, -1)),     # Mixed
        ((3, 1), (1, 2)),      # Mixed 2
        ((2, 2), (1, -1)),     # Mixed 3
        ((1, 3), (3, 1)),      # Mixed 4
        ((4, 1), (1, 2)),      # Mixed 5
    ]
    
    for basis_a, basis_b in lattice_bases:
        if time.time() > deadline:
            break
        
        # Try different origins
        for ox in range(-5, 6, 2):
            for oy in range(-5, 6, 2):
                if time.time() > deadline:
                    break
                
                candidates = _generate_lattice_base(basis_a, basis_b, (ox, oy))
                if len(candidates) < 30:
                    continue
                
                # Select 30 points - try different subsets
                # Use the first 30, or try to maximize score
                np.random.shuffle(candidates)
                for attempt in range(min(10, len(candidates))):
                    subset = candidates[attempt*30:(attempt+1)*30] if (attempt+1)*30 <= len(candidates) else candidates[:30]
                    if len(subset) < 30:
                        continue
                    
                    score = _score_vectorized(subset)
                    if score > best_score:
                        best_score = score
                        best_points = subset[:]
                    
                    # Local search
                    if time.time() < deadline - 10:
                        improved_points, improved_score = _local_search(
                            subset, max_iters=100, 
                            time_limit=min(5, deadline - time.time())
                        )
                        if improved_score > best_score:
                            best_score = improved_score
                            best_points = improved_points[:]
    
    # Strategy 2: Random starts with local search
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        points = _random_points(30)
        score = _score_vectorized(points)
        if score > best_score:
            best_score = score
            best_points = points[:]
        
        if time.time() < deadline - 10:
            improved_points, improved_score = _local_search(
                points, max_iters=200,
                time_limit=min(5, deadline - time.time())
            )
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points[:]
    
    # Strategy 3: Structured patterns
    for pattern in [
        _structured_points,
        lambda: [(x, y) for x in range(-4, 4) for y in range(-4, 4) if abs(x)+abs(y) <= 5][:30],
        lambda: [(x*3, y) for x in range(-3, 4) for y in range(-3, 4)][:30],
        lambda: [(x, y*3) for x in range(-3, 4) for y in range(-3, 4)][:30],
        lambda: [(x+x*y, y) for x in range(-3, 4) for y in range(-3, 4) if -30 <= x+x*y <= 30][:30],
    ]:
        if time.time() > deadline - 5:
            break
        
        try:
            points = pattern()
            if len(points) == 30 and len(set(points)) == 30:
                score = _score_vectorized(points)
                if score > best_score:
                    best_score = score
                    best_points = points[:]
                
                if time.time() < deadline - 10:
                    improved_points, improved_score = _local_search(
                        points, max_iters=150,
                        time_limit=min(5, deadline - time.time())
                    )
                    if improved_score > best_score:
                        best_score = improved_score
                        best_points = improved_points[:]
        except Exception:
            continue
    
    # Fallback: return the initial grid if nothing better found
    if best_points is None or len(best_points) != 30:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    # Ensure exactly 30 distinct points in range
    seen = set()
    final_points = []
    for p in best_points:
        if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
            seen.add(p)
            final_points.append(p)
    
    while len(final_points) < 30:
        # Add random points to fill
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        if (x, y) not in seen:
            seen.add((x, y))
            final_points.append((x, y))
    
    return final_points[:30]

# EVOLVE-BLOCK-END
