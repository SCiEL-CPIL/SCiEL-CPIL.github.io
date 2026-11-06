"""Combined: fast vectorized evaluation + number-theoretic lattice construction + local search."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    # Precompute triple indices for n=30
    triple_indices = np.array([(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)], dtype=np.int64)
    I, J, K = triple_indices[:, 0], triple_indices[:, 1], triple_indices[:, 2]

    def evaluate(points):
        """Vectorized evaluation of max multiplicity of positive doubled areas."""
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        det = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
        det = np.abs(det)
        det = det[det > 0]
        if len(det) == 0:
            return 0
        vals, cnts = np.unique(det, return_counts=True)
        return int(cnts.max())

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    def try_config(points, current_best):
        if not valid(points):
            return None, current_best
        score = evaluate(points)
        if score > current_best:
            return points, score
        return None, current_best

    # --- Phase 1: Number-theoretic lattice constructions ---
    # Key insight: points = a*u + b*v where u, v are primitive lattice vectors.
    # The doubled area of triangle (a1,b1),(a2,b2),(a3,b3) is |det(u,v)| * |det of (a1,b1),(a2,b2),(a3,b3) in parameter space|
    # So we want to maximize the multiplicity of a specific area in parameter space, scaled by |det(u,v)|.
    
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)

    # Try various lattice bases and parameter ranges
    # Points are (i*a1 + j*a2, i*b1 + j*b2) for (i,j) in a set S of 30 pairs
    # det(u,v) = a1*b2 - a2*b1
    # The area in parameter space for triangle (i1,j1),(i2,j2),(i3,j3) is the 2D cross product
    
    # Strategy: use a rectangular grid in parameter space (i in [0,w-1], j in [0,h-1])
    # with w*h = 30, and vary the lattice basis
    
    factors_30 = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    
    for w, h in factors_30:
        for a1 in range(1, 8):
            for a2 in range(-4, 5):
                for b1 in range(-4, 5):
                    for b2 in range(1, 8):
                        if time.time() > deadline:
                            return best_points
                        
                        det_uv = a1 * b2 - a2 * b1
                        if det_uv == 0:
                            continue
                        abs_det = abs(det_uv)
                        if abs_det < 1:
                            continue
                        
                        # Generate parameter grid
                        pts = []
                        for i in range(w):
                            for j in range(h):
                                px = i * a1 + j * a2
                                py = i * b1 + j * b2
                                pts.append((px, py))
                        
                        # Check bounds
                        if not valid(pts):
                            continue
                        
                        # Try centering
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        cx = (max(xs) + min(xs)) // 2
                        cy = (max(ys) + min(ys)) // 2
                        centered = [(px - cx, py - cy) for px, py in pts]
                        
                        if valid(centered):
                            result, new_score = try_config(centered, best_score)
                            if result is not None:
                                best_points = result
                                best_score = new_score

    # --- Phase 2: Two-parallel-lines with varying spacing ---
    for d in range(1, 8):
        for offset in range(-3, 4):
            pts = [(x + offset, 0) for x in range(15)] + [(x + offset, d) for x in range(15)]
            if valid(pts):
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # --- Phase 3: Three-line configurations ---
    for d in range(1, 5):
        for a in range(6, 25):
            for b in range(6, 25 - a + 6):
                c = 30 - a - b
                if c < 1 or c > 30:
                    continue
                pts = [(x, 0) for x in range(a)] + [(x, d) for x in range(b)] + [(x, 2*d) for x in range(c)]
                if valid(pts):
                    result, new_score = try_config(pts, best_score)
                    if result is not None:
                        best_points = result
                        best_score = new_score
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    # --- Phase 4: Sheared grids ---
    for w, h in [(6,5), (5,6), (10,3), (3,10)]:
        for sx in range(1, 5):
            for sy in range(1, 5):
                for sxy in range(-3, 4):
                    if time.time() > deadline:
                        return best_points
                    pts = [(i * sx + j * sxy, j * sy) for i in range(w) for j in range(h)]
                    if valid(pts):
                        result, new_score = try_config(pts, best_score)
                        if result is not None:
                            best_points = result
                            best_score = new_score

    # --- Phase 5: Local search with larger neighborhood ---
    import random
    random.seed(42)
    
    improved = True
    max_iterations = 30
    iteration = 0
    while improved and time.time() < deadline - 8 and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(30):
            if time.time() > deadline - 8:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # --- Phase 6: Random restarts with lattice-inspired perturbations ---
    if time.time() < deadline - 5:
        random.seed(123)
        for restart in range(200):
            if time.time() > deadline - 3:
                break
            
            if random.random() < 0.5:
                # Small perturbation of best
                new_points = list(best_points)
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 29)
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    new_x = new_points[idx][0] + dx
                    new_y = new_points[idx][1] + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        new_points[idx] = (new_x, new_y)
            else:
                # Random lattice-based structure
                w, h = random.choice([(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)])
                a1 = random.randint(1, 4)
                a2 = random.randint(-2, 2)
                b1 = random.randint(-2, 2)
                b2 = random.randint(1, 4)
                cx = random.randint(-5, 5)
                cy = random.randint(-5, 5)
                new_points = []
                for i in range(w):
                    for j in range(h):
                        px = cx + i * a1 + j * a2 + random.randint(-1, 1)
                        py = cy + i * b1 + j * b2 + random.randint(-1, 1)
                        new_points.append((px, py))
            
            result, new_score = try_config(new_points, best_score)
            if result is not None:
                best_points = result
                best_score = new_score

    # --- Phase 7: Final polish with tighter local search ---
    if time.time() < deadline - 2:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            improved = False
            for idx in range(30):
                if time.time() > deadline - 2:
                    break
                old_x, old_y = best_points[idx]
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                            continue
                        new_points = list(best_points)
                        new_points[idx] = (new_x, new_y)
                        if len(set(new_points)) != 30:
                            continue
                        score = evaluate(new_points)
                        if score > best_score:
                            best_score = score
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    return best_points
# EVOLVE-BLOCK-END
