"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(points):
    """Compute the max multiplicity of a single positive doubled area."""
    n = len(points)
    area_counts = {}
    for i in range(n - 2):
        x1, y1 = points[i]
        for j in range(i + 1, n - 1):
            x2, y2 = points[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs(dx12 * (y3 - y1) - dy12 * (x3 - x1))
                if det > 0:
                    area_counts[det] = area_counts.get(det, 0) + 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def _generate_grid():
    """Generate candidate grid points in [-30, 30]^2."""
    pts = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            pts.append((x, y))
    return pts

def _is_valid(points):
    """Check that we have 30 distinct points in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set(points)
    if len(seen) != 30:
        return False
    for x, y in points:
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin
    
    random.seed(42)
    all_pts = _generate_grid()
    n_all = len(all_pts)
    
    # Start with a grid-based initial configuration
    points = [(x, y) for x in range(-14, 16) for y in range(-14, 15)][:30]
    # Ensure we have 30 distinct points
    while len(points) < 30:
        p = random.choice(all_pts)
        if p not in points:
            points.append(p)
    points = points[:30]
    
    best_score = _score(points)
    best_points = list(points)
    
    # Local search: try swapping one point with another
    # and also try moving one point to a new location
    while time.time() < deadline:
        # Pick a random target area to focus on
        # First, let's see what areas are common
        area_counts = {}
        for i in range(30 - 2):
            x1, y1 = points[i]
            for j in range(i + 1, 30 - 1):
                x2, y2 = points[j]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for k in range(j + 1, 30):
                    x3, y3 = points[k]
                    det = abs(dx12 * (y3 - y1) - dy12 * (x3 - x1))
                    if det > 0:
                        area_counts[det] = area_counts.get(det, 0) + 1
        
        if not area_counts:
            break
        
        # Focus on the most common area
        target_area = max(area_counts, key=lambda a: area_counts[a])
        
        # Try improving by moving individual points
        improved = False
        for _ in range(200):  # limited inner iterations
            if time.time() > deadline:
                break
            idx = random.randint(0, 29)
            old_x, old_y = points[idx]
            
            # Try a random new position
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
            if (new_x, new_y) in points:
                continue
            
            # Evaluate the change
            # Count how many triples involving idx have area == target_area
            old_count = 0
            new_count = 0
            
            for i in range(30):
                if i == idx:
                    continue
                for j in range(i + 1, 30):
                    if j == idx:
                        continue
                    if i == j:
                        continue
                    # Triple (i, idx, j) - but we need to be careful with ordering
                    # Actually let's just count triples containing idx
                    pass
            
            # Simpler: just evaluate the full score for this move
            # But that's expensive. Let's do a partial evaluation.
            
            # Count triples containing idx with the old point
            for i in range(30):
                if i == idx:
                    continue
                for j in range(i + 1, 30):
                    if j == idx:
                        continue
                    if i == j:
                        continue
                    # The triple is {i, idx, j}
                    # We need to compute the determinant
                    # Points are points[i], points[idx], points[j]
                    # But if we move idx, the triple changes
                    pass
            
            # Let's just do full re-evaluation for simplicity, but limit it
            # Actually, let's be smarter: only evaluate triples involving idx
            # There are C(29,2) = 406 such triples
            old_idx_count = 0
            new_idx_count = 0
            pi = points[idx]
            for i in range(30):
                if i == idx:
                    continue
                for j in range(i + 1, 30):
                    if j == idx:
                        continue
                    if i == j:
                        continue
                    # Triple (i, idx, j)
                    x_a, y_a = points[i]
                    x_c, y_c = points[j]
                    # Old: determinant with pi
                    det_old = abs((x_a - pi[0]) * (y_c - pi[1]) - (y_a - pi[1]) * (x_c - pi[0]))
                    # Wait, this isn't quite right. The determinant for triangle (a, b, c) is
                    # |ax(by - cy) + bx(cy - ay) + cx(ay - by)|
                    # Let me recompute properly
                    pass
            
            # This is getting complex. Let's just do a simpler local search.
            # Try replacing point idx with a nearby point
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
            
            if (new_x, new_y) in points:
                continue
            
            # Evaluate
            new_points = list(points)
            new_points[idx] = (new_x, new_y)
            new_score = _score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                points = new_points
                improved = True
            elif new_score == best_score and random.random() < 0.3:
                # Accept equal moves with some probability (simulated annealing-like)
                points = new_points
                improved = True
        
        if not improved:
            # Try a larger jump
            idx = random.randint(0, 29)
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            if (new_x, new_y) not in points:
                new_points = list(points)
                new_points[idx] = (new_x, new_y)
                new_score = _score(new_points)
                if new_score >= best_score:
                    points = new_points
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
    
    return best_points
# EVOLVE-BLOCK-END
