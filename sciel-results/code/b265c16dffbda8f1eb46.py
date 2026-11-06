"""Optimized: Lattice section approach - select 30 points from dense sublattice, optimize for target area."""
import time
import numpy as np
import random
from itertools import combinations
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin

    # Precompute indices for all triples
    indices = list(combinations(range(30), 3))
    idx_arr = np.array(indices, dtype=np.int64)
    I, J, K = idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]

    def fast_evaluate(points):
        """Compute max multiplicity of any positive doubled area."""
        if len(points) != 30:
            return 0
        if len(set(points)) < 30:
            return 0
        
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        positive_areas = areas[areas > 0]
        
        if len(positive_areas) == 0:
            return 0
            
        max_area = int(positive_areas.max())
        if max_area > 20000:
            _, counts = np.unique(positive_areas, return_counts=True)
            return int(counts.max())
        counts = np.bincount(positive_areas.astype(np.int64))
        return int(counts.max())

    def evaluate_target_area(points, target_area):
        """Count how many triples have exactly the given doubled area."""
        if len(points) != 30:
            return 0
        if len(set(points)) < 30:
            return 0
        
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return int(np.sum(areas == target_area))

    def compute_all_areas(points):
        """Compute all 4060 doubled areas for a set of 30 points."""
        pts = np.array(points, dtype=np.int64)
        x, y = pts[:, 0], pts[:, 1]
        xi, yi = x[I], y[I]
        xj, yj = x[J], y[J]
        xk, yk = x[K], y[K]
        return np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))

    def delta_evaluate(points, idx, new_pt, current_areas):
        """
        Fast delta evaluation: only recompute areas involving point idx.
        Returns new max multiplicity without full recomputation.
        """
        # Areas involving point idx: those where I==idx or J==idx or K==idx
        mask = (I == idx) | (J == idx) | (K == idx)
        n_affected = int(mask.sum())
        
        if n_affected == 0:
            return current_areas.copy()
        
        new_areas = current_areas.copy()
        
        # Get old and new coordinates
        old_pt = points[idx]
        pts_arr = np.array(points, dtype=np.int64)
        x, y = pts_arr[:, 0], pts_arr[:, 1]
        
        # For each affected triple, recompute with new point
        affected_I = I[mask]
        affected_J = J[mask]
        affected_K = K[mask]
        
        # Build arrays of coordinates for affected triples
        xi_arr = x[affected_I]
        yi_arr = y[affected_I]
        xj_arr = x[affected_J]
        yj_arr = y[affected_J]
        xk_arr = x[affected_K]
        yk_arr = y[affected_K]
        
        # Replace idx with new point in each position
        new_xi = np.where(affected_I == idx, new_pt[0], xi_arr)
        new_yi = np.where(affected_I == idx, new_pt[1], yi_arr)
        new_xj = np.where(affected_J == idx, new_pt[0], xj_arr)
        new_yj = np.where(affected_J == idx, new_pt[1], yj_arr)
        new_xk = np.where(affected_K == idx, new_pt[0], xk_arr)
        new_yk = np.where(affected_K == idx, new_pt[1], yk_arr)
        
        new_vals = np.abs(new_xi * (new_yj - new_yk) + new_xj * (new_yk - new_yi) + new_xk * (new_yi - new_yj))
        new_areas[mask] = new_vals
        
        return new_areas

    # --- Phase 1: Generate dense sublattice candidates ---
    best_points = None
    best_score = 0
    best_target = 0
    
    lattice_configs = []
    
    for spacing in [1, 2, 3]:
        max_dim = 60 // spacing + 1
        for side in range(5, min(8, max_dim + 1)):
            lattice_configs.append(('square', spacing, side))
    
    for spacing in [1, 2]:
        max_w = 60 // spacing + 1
        max_h = 60 // spacing + 1
        for w in range(4, min(8, max_w + 1)):
            for h in range(4, min(8, max_h + 1)):
                if w * h >= 30:
                    lattice_configs.append(('rect', spacing, (w, h)))
    
    for spacing in [1, 2]:
        for shear in [1, 2]:
            for w in range(4, 8):
                for h in range(4, 8):
                    if w * h >= 30:
                        lattice_configs.append(('shear', spacing, (w, h, shear)))
    
    random.seed(42)
    random.shuffle(lattice_configs)
    
    for config in lattice_configs:
        if time.time() > deadline - 20:
            break
        
        kind = config[0]
        
        if kind == 'square':
            _, spacing, side = config
            pts = []
            for i in range(side):
                for j in range(side):
                    pts.append((i * spacing, j * spacing))
        elif kind == 'rect':
            _, spacing, (w, h) = config
            pts = []
            for i in range(w):
                for j in range(h):
                    pts.append((i * spacing, j * spacing))
        else:
            _, spacing, (w, h, shear) = config
            pts = []
            for i in range(w):
                for j in range(h):
                    px = i * spacing + j * shear
                    py = j * spacing
                    pts.append((px, py))
        
        if len(pts) < 30:
            continue
        
        if len(set(pts)) < len(pts):
            continue
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        
        centered_pts = [(p[0] - cx, p[1] - cy) for p in pts]
        
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in centered_pts):
            continue
        
        if len(centered_pts) == 30:
            selected = centered_pts
        else:
            random.shuffle(centered_pts)
            selected = centered_pts[:30]
        
        score = fast_evaluate(selected)
        if score > best_score:
            best_score = score
            best_points = selected
            areas = compute_all_areas(selected)
            positive_areas = areas[areas > 0]
            if len(positive_areas) > 0:
                _, counts = np.unique(positive_areas, return_counts=True)
                best_target = int(positive_areas[np.argmax(counts)])
    
    if best_points is None:
        pts = [(i, j) for i in range(5) for j in range(6)]
        cx, cy = 2, 2
        best_points = [(p[0]-cx, p[1]-cy) for p in pts]
        best_score = fast_evaluate(best_points)
        best_target = 2

    # --- Phase 2: Local search with incremental delta evaluation ---
    current_points = list(best_points)
    current_score = best_score
    current_target = best_target
    best_overall_points = list(current_points)
    best_overall_score = current_score
    
    # Maintain running areas array for fast delta updates
    current_areas = compute_all_areas(current_points)
    
    # Precompute masks for each point index (which triples it's in)
    point_masks = []
    for idx in range(30):
        mask = (I == idx) | (J == idx) | (K == idx)
        point_masks.append(mask)
    
    small_moves = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]
    medium_moves = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    large_moves = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
    
    no_improve = 0
    max_no_improve = 1500
    restart_count = 0
    max_restarts = 5
    
    while time.time() < deadline - 2:
        if no_improve > max_no_improve:
            # Restart: perturb current solution
            restart_count += 1
            if restart_count > max_restarts:
                break
            
            # Perturb a few random points with larger moves
            for _ in range(3):
                idx = random.randint(0, 29)
                dx, dy = random.choice(large_moves)
                nx, ny = current_points[idx][0] + dx, current_points[idx][1] + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    if new_pt not in current_points:
                        # Apply move
                        current_areas = delta_evaluate(current_points, idx, new_pt, current_areas)
                        current_points[idx] = new_pt
            
            # Recompute score
            current_score = fast_evaluate(current_points)
            positive_areas = current_areas[current_areas > 0]
            if len(positive_areas) > 0:
                _, counts = np.unique(positive_areas, return_counts=True)
                current_target = int(positive_areas[np.argmax(counts)])
            
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_points = list(current_points)
            
            no_improve = 0
            continue
        
        # Pick random point and move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        if no_improve > 600:
            moves = medium_moves
        elif no_improve > 1200:
            moves = large_moves
        else:
            moves = small_moves
        
        dx, dy = random.choice(moves)
        new_x = old_x + dx
        new_y = old_y + dy
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            no_improve += 1
            continue
        
        new_point = (new_x, new_y)
        
        # Check collision
        if new_point in current_points:
            no_improve += 1
            continue
        
        # Fast delta evaluation
        new_areas = delta_evaluate(current_points, idx, new_point, current_areas)
        
        # Compute new score from new_areas
        positive_new = new_areas[new_areas > 0]
        if len(positive_new) == 0:
            new_score = 0
            new_target = current_target
        else:
            max_area = int(positive_new.max())
            if max_area > 20000:
                _, counts = np.unique(positive_new, return_counts=True)
                new_score = int(counts.max())
                new_target = int(positive_new[np.argmax(counts)])
            else:
                counts = np.bincount(positive_new.astype(np.int64))
                new_score = int(counts.max())
                new_target = int(np.argmax(counts))
        
        # Also compute target-specific count for the old target
        target_count_new = int(np.sum(new_areas == current_target))
        target_count_old = int(np.sum(current_areas == current_target))
        
        improved = False
        if new_score > current_score:
            improved = True
            current_score = new_score
            current_target = new_target
            current_points[idx] = new_point
            current_areas = new_areas
            no_improve = 0
            
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_points = list(current_points)
        elif target_count_new > target_count_old and random.random() < 0.4:
            # Accept moves that improve target count (even if overall score same)
            improved = True
            current_points[idx] = new_point
            current_areas = new_areas
            no_improve = 0
        else:
            no_improve += 1
    
    return best_overall_points
