# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import itertools

N = 13
# Precompute all 286 triangle index triples
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas) / A_hull

def compute_all_areas(pts):
    """Vectorized areas of all 286 triangles."""
    p1 = pts[TRI_IDX[:, 0]]
    p2 = pts[TRI_IDX[:, 1]]
    p3 = pts[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def delaunay_repair(pts, deadline, max_iters=2000):
    """
    Constrained Delaunay triangulation repair strategy:
    1. Compute Delaunay triangulation
    2. Find minimum-area Delaunay triangles
    3. Move vertices of those triangles toward centroid of adjacent triangles
    4. Repeat until convergence or deadline
    """
    p = pts.copy()
    best_p = p.copy()
    best_score = compute_score(p)
    
    # Precompute: for each point, which triangles contain it
    tri_masks = []
    for pi in range(N):
        mask = (TRI_IDX[:, 0] == pi) | (TRI_IDX[:, 1] == pi) | (TRI_IDX[:, 2] == pi)
        tri_masks.append(mask)
    
    step_size = 0.08
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current areas
        areas = compute_all_areas(p)
        min_area = np.min(areas)
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(p)
        except Exception:
            break
        
        # Find the minimum-area Delaunay triangles
        # For each Delaunay triangle, compute its area
        delaunay_areas = np.zeros(len(tri.simplices))
        for idx, simplex in enumerate(tri.simplices):
            i, j, k = simplex
            v1 = p[j] - p[i]
            v2 = p[k] - p[i]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            delaunay_areas[idx] = abs(cross) * 0.5
        
        # Get the worst (smallest area) Delaunay triangles
        num_worst = min(3, len(delaunay_areas))
        worst_indices = np.argsort(delaunay_areas)[:num_worst]
        
        improved = False
        
        for wi in worst_indices:
            simplex = tri.simplices[wi]
            i, j, k = simplex
            
            # Find adjacent triangles (sharing an edge)
            # For each vertex in the worst triangle, try moving it
            for vertex_idx in [i, j, k]:
                # Compute the centroid of the worst triangle
                centroid = (p[i] + p[j] + p[k]) / 3.0
                
                # Compute the centroid of the "neighbor" region
                # Use the circumcenter of the worst triangle as a target direction
                # Actually, let's move toward the average of adjacent triangle centroids
                
                # Find triangles adjacent to this one (sharing 2 vertices)
                adjacent_centroids = []
                for other_simplex in tri.simplices:
                    if np.array_equal(sorted(other_simplex), sorted(simplex)):
                        continue
                    shared = set(other_simplex) & set(simplex)
                    if len(shared) == 2:
                        other_centroid = (p[other_simplex[0]] + p[other_simplex[1]] + p[other_simplex[2]]) / 3.0
                        adjacent_centroids.append(other_centroid)
                
                if adjacent_centroids:
                    neighbor_centroid = np.mean(adjacent_centroids, axis=0)
                    # Move vertex toward the midpoint between its current position and neighbor centroid
                    target = 0.5 * p[vertex_idx] + 0.5 * neighbor_centroid
                    move = target - p[vertex_idx]
                    move_norm = np.linalg.norm(move)
                    if move_norm > 1e-10:
                        move = move / move_norm * step_size
                else:
                    # No adjacent triangles, move toward the overall centroid
                    overall_centroid = np.mean(p, axis=0)
                    move = overall_centroid - p[vertex_idx]
                    move_norm = np.linalg.norm(move)
                    if move_norm > 1e-10:
                        move = move / move_norm * step_size
                    else:
                        continue
                
                # Try the move
                old_pos = p[vertex_idx].copy()
                p[vertex_idx] += move
                
                # Check if hull is still valid
                try:
                    hull = ConvexHull(p)
                    A_hull = hull.volume
                    if A_hull <= 1e-12:
                        p[vertex_idx] = old_pos
                        continue
                except Exception:
                    p[vertex_idx] = old_pos
                    continue
                
                new_areas = compute_all_areas(p)
                new_min = np.min(new_areas)
                new_score = new_min / A_hull
                
                if new_score > best_score:
                    best_score = new_score
                    best_p = p.copy()
                    improved = True
                else:
                    p[vertex_idx] = old_pos
        
        if not improved:
            step_size *= 0.95
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
            step_size = min(step_size * 1.02, 0.1)
    
    return best_p, best_score

def gradient_ascent(pts, deadline, beta_start=8.0, beta_end=600.0,
                    step_start=0.03, step_end=0.0015, n_steps=500):
    """
    Annealed gradient ascent on the soft-min surrogate.
    """
    p = pts.copy()
    best_p = p.copy()
    best_val = -1e10
    
    for s in range(n_steps):
        if time.time() > deadline:
            break
        prog = s / max(n_steps - 1, 1)
        beta = beta_start + (beta_end - beta_start) * prog
        temperature = 1.0 / beta
        step = step_start + (step_end - step_start) * prog
        
        # Compute soft-min gradient
        try:
            hull = ConvexHull(p)
            A_hull = hull.volume
        except Exception:
            break
        if A_hull <= 1e-12:
            break
        
        areas = compute_all_areas(p)
        if not np.all(np.isfinite(areas)):
            break
        
        # Soft-min weights
        scaled = -areas / temperature
        m = np.max(scaled)
        w = np.exp(scaled - m)
        s_sum = np.sum(w)
        if s_sum <= 0 or not np.isfinite(s_sum):
            break
        w = w / s_sum
        
        # Compute gradient
        p1 = p[TRI_IDX[:, 0]]
        p2 = p[TRI_IDX[:, 1]]
        p3 = p[TRI_IDX[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        sign = np.sign(cross)
        
        coef = w * sign / A_hull
        i = TRI_IDX[:, 0]
        j = TRI_IDX[:, 1]
        k = TRI_IDX[:, 2]
        
        pi = p[i]
        pj = p[j]
        pk = p[k]
        
        dx_pj = pj[:, 0] - pi[:, 0]
        dy_pj = pj[:, 1] - pi[:, 1]
        dx_pk = pk[:, 0] - pi[:, 0]
        dy_pk = pk[:, 1] - pi[:, 1]
        
        g = np.zeros_like(p)
        np.add.at(g, i, coef[:, None] * np.column_stack([dy_pj - dy_pk, dx_pk - dx_pj]))
        np.add.at(g, j, coef[:, None] * np.column_stack([dy_pk, -dx_pk]))
        np.add.at(g, k, coef[:, None] * np.column_stack([-dy_pj, dx_pj]))
        
        nrm = np.linalg.norm(g)
        if nrm < 1e-14:
            break
        
        p = p + step * g
        if not np.all(np.isfinite(p)):
            break
        
        # Track best
        if s % 20 == 0 or s == n_steps - 1:
            cv = compute_score(p)
            if cv > best_val:
                best_val = cv
                best_p = p.copy()
    
    return best_p

def local_search(points, time_limit, rng):
    """Fast coordinate ascent: only recompute the 26 triangles affected by a moved point."""
    pts = points.copy()
    current_score = compute_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    step_size = 0.12
    
    # Precompute: for each point, which triangle indices contain it
    tri_masks = []
    for p in range(N):
        mask = (TRI_IDX[:, 0] == p) | (TRI_IDX[:, 1] == p) | (TRI_IDX[:, 2] == p)
        tri_masks.append(mask)
    
    areas = compute_all_areas(pts)
    try:
        A_hull = ConvexHull(pts).volume
    except Exception:
        return best_pts, best_score
    
    def fast_min_area(p, changed_pt, prev_areas):
        """Recompute only triangles involving changed_pt, return new min area."""
        mask = tri_masks[changed_pt]
        idxs = np.where(mask)[0]
        p1 = p[TRI_IDX[idxs, 0]]
        p2 = p[TRI_IDX[idxs, 1]]
        p3 = p[TRI_IDX[idxs, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        new_areas = np.abs(cross) * 0.5
        prev_areas[idxs] = new_areas
        return np.min(prev_areas)
    
    while time.time() < time_limit and step_size > 1e-8:
        improved = False
        order = rng.permutation(N)
        for idx in order:
            for _ in range(3):
                theta = rng.uniform(0, 2 * np.pi)
                mag = step_size * rng.uniform(0.4, 1.0)
                dx = mag * np.cos(theta)
                dy = mag * np.sin(theta)
                
                old_x, old_y = pts[idx, 0], pts[idx, 1]
                pts[idx, 0] += dx
                pts[idx, 1] += dy
                
                try:
                    A_hull_new = ConvexHull(pts).volume
                except Exception:
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
                    continue
                
                new_min = fast_min_area(pts, idx, areas)
                new_score = new_min / A_hull_new
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    A_hull = A_hull_new
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                else:
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
                    mask = tri_masks[idx]
                    idxs = np.where(mask)[0]
                    p1 = pts[TRI_IDX[idxs, 0]]
                    p2 = pts[TRI_IDX[idxs, 1]]
                    p3 = pts[TRI_IDX[idxs, 2]]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                    areas[idxs] = np.abs(cross) * 0.5
        
        if not improved:
            step_size *= 0.75
        else:
            step_size *= 0.97
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1.0
    
    candidates = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    # 3. 7 outer, 6 inner (offset)
    ang6b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 12
    p6b = np.column_stack([0.6 * np.cos(ang6b), 0.6 * np.sin(ang6b)])
    candidates.append(np.vstack([p7, p6b]))
    # 4. 5/5/3
    ang5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5o = np.column_stack([np.cos(ang5), np.sin(ang5)])
    ang5m = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5m = np.column_stack([0.6 * np.cos(ang5m), 0.6 * np.sin(ang5m)])
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
    p3i = np.column_stack([0.3 * np.cos(ang3), 0.3 * np.sin(ang3)])
    candidates.append(np.vstack([p5o, p5m, p3i]))
    # 5. 4/5/4
    ang4o = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4o = np.column_stack([np.cos(ang4o), np.sin(ang4o)])
    ang5m2 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5m2 = np.column_stack([0.65 * np.cos(ang5m2), 0.65 * np.sin(ang5m2)])
    ang4i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i = np.column_stack([0.35 * np.cos(ang4i), 0.35 * np.sin(ang4i)])
    candidates.append(np.vstack([p4o, p5m2, p4i]))
    # 6. 6 outer, 7 inner
    ang6o = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p6o = np.column_stack([np.cos(ang6o), np.sin(ang6o)])
    ang7i = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    p7i = np.column_stack([0.5 * np.cos(ang7i), 0.5 * np.sin(ang7i)])
    candidates.append(np.vstack([p6o, p7i]))
    # 7. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    # 8. 4/5/4 variant
    ang4a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4a = np.column_stack([np.cos(ang4a), np.sin(ang4a)])
    ang5b = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5b = np.column_stack([0.65 * np.cos(ang5b), 0.65 * np.sin(ang5b)])
    ang4c = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4c = np.column_stack([0.3 * np.cos(ang4c), 0.3 * np.sin(ang4c)])
    candidates.append(np.vstack([p4a, p5b, p4c]))
    # 9. 5/4/4
    ang5o2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 10
    p5o2 = np.column_stack([np.cos(ang5o2), np.sin(ang5o2)])
    ang4m2 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4m2 = np.column_stack([0.6 * np.cos(ang4m2), 0.6 * np.sin(ang4m2)])
    ang4i2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i2 = np.column_stack([0.3 * np.cos(ang4i2), 0.3 * np.sin(ang4i2)])
    candidates.append(np.vstack([p5o2, p4m2, p4i2]))
    # 10. Random
    for _ in range(5):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    # Perturbed structured
    for base_idx in [0, 1, 3, 4]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))
    
    n_cand = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()
        
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        remaining_cands = n_cand - i
        time_per_cand = min(remaining / max(remaining_cands, 1), 90.0)
        cand_deadline = time.time() + time_per_cand
        
        cur = init_pts.copy()
        
        # Phase 1: Delaunay repair (NEW - structural change)
        if time.time() < cand_deadline - 10:
            dr_pts, dr_score = delaunay_repair(cur, cand_deadline - 10)
            if dr_score > best_score:
                best_score = dr_score
                best_points = dr_pts.copy()
            cur = dr_pts
        
        # Phase 2: short coordinate warm-up
        if time.time() < cand_deadline - 5:
            ls_pts, ls_score = local_search(cur, cand_deadline - 5, rng)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_pts.copy()
            cur = ls_pts
        
        # Phase 3: gradient-ascent anneal
        if time.time() < deadline - 10:
            ga_deadline = min(cand_deadline, deadline - 10)
            ga_pts = gradient_ascent(cur, ga_deadline)
            ga_score = compute_score(ga_pts)
            if ga_score > best_score:
                best_score = ga_score
                best_points = ga_pts.copy()
                cur = ga_pts
        
        # Phase 4: fine coordinate polish
        if time.time() < deadline - 8:
            fine_deadline = min(cand_deadline, deadline - 8)
            fs_pts, fs_score = local_search(cur, fine_deadline, rng)
            if fs_score > best_score:
                best_score = fs_score
                best_points = fs_pts.copy()
                cur = fs_pts
        
        # Kick restart
        if time.time() < deadline - 20 and i < n_cand - 1:
            kick_pts = best_points + rng.normal(0, 0.15, size=(N, 2))
            kick_score = compute_score(kick_pts)
            if kick_score > best_score * 0.85:
                kick_deadline = time.time() + min(remaining * 0.25, 25.0)
                if time.time() < kick_deadline - 3:
                    # Delaunay repair on kicked points
                    dr_pts, dr_score = delaunay_repair(kick_pts, kick_deadline - 10)
                    if dr_score > best_score:
                        best_score = dr_score
                        best_points = dr_pts.copy()
                    if time.time() < kick_deadline - 3:
                        ls_pts, ls_score = local_search(dr_pts, kick_deadline - 8, rng)
                        if ls_score > best_score:
                            best_score = ls_score
                            best_points = ls_pts.copy()
                        if time.time() < kick_deadline - 3:
                            ga_pts = gradient_ascent(ls_pts, kick_deadline - 3,
                                                     beta_start=10.0, beta_end=400.0,
                                                     step_start=0.02, step_end=0.002, n_steps=300)
                            ga_score = compute_score(ga_pts)
                            if ga_score > best_score:
                                best_score = ga_score
                                best_points = ga_pts.copy()
    
    # Final polish if time remains
    if best_points is not None and time.time() < deadline - 5:
        # Round 1: Delaunay repair
        dr_pts, dr_score = delaunay_repair(best_points, deadline - 15)
        if dr_score > best_score:
            best_score = dr_score
            best_points = dr_pts.copy()
        
        # Round 2: gradient ascent
        if time.time() < deadline - 10:
            ga_pts = gradient_ascent(best_points, deadline - 10,
                                     beta_start=15.0, beta_end=500.0,
                                     step_start=0.015, step_end=0.001, n_steps=400)
            ga_score = compute_score(ga_pts)
            if ga_score > best_score:
                best_score = ga_score
                best_points = ga_pts.copy()
        
        # Round 3: fast local search polish
        if time.time() < deadline - 5:
            fs_pts, fs_score = local_search(best_points, deadline - 5, rng)
            if fs_score > best_score:
                best_score = fs_score
                best_points = fs_pts.copy()
        
        # Round 4: another gradient pass
        if time.time() < deadline - 3:
            ga_pts2 = gradient_ascent(best_points, deadline - 3,
                                      beta_start=50.0, beta_end=1500.0,
                                      step_start=0.008, step_end=0.0005, n_steps=200)
            ga_score2 = compute_score(ga_pts2)
            if ga_score2 > best_score:
                best_score = ga_score2
                best_points = ga_pts2.copy()
        
        # Round 5: final micro local search
        if time.time() < deadline - 1:
            fs_pts2, fs_score2 = local_search(best_points, deadline - 1, rng)
            if fs_score2 > best_score:
                best_score = fs_score2
                best_points = fs_pts2.copy()
    
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
