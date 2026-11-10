# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
import math

N = 13
# Precompute triangle indices for efficiency
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

# Precompute which triangles involve each point (for fast incremental updates)
# tri_list[i] = list of triangle indices that include point i
TRI_LIST = [[] for _ in range(N)]
for t_idx, (a, b, c) in enumerate(TRI_IDX):
    TRI_LIST[a].append(t_idx)
    TRI_LIST[b].append(t_idx)
    TRI_LIST[c].append(t_idx)

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    Returns 0.0 if invalid
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Vectorized area calculation
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def compute_all_areas(points):
    """Compute all 286 triangle areas."""
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def fast_local_search(points, step_size, max_rounds, deadline):
    """
    Fast incremental local search: only recompute 26 triangles per coordinate move.
    Returns (improved_points, improved_score).
    """
    pts = points.copy()
    areas = compute_all_areas(pts)
    
    try:
        hull = ConvexHull(pts)
        A_hull = hull.volume
    except Exception:
        return pts, 0.0
    
    if A_hull <= 0:
        return pts, 0.0
    
    current_min = np.min(areas)
    current_score = current_min / A_hull
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            # Triangles involving point idx
            tri_indices = TRI_LIST[idx]
            
            # Try 8 directions (4 cardinal + 4 diagonal)
            for dx, dy in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size),
                           (-step_size, -step_size), (step_size, -step_size),
                           (-step_size, step_size), (step_size, step_size)]:
                # Save old positions
                old_x, old_y = pts[idx, 0], pts[idx, 1]
                pts[idx, 0] = old_x + dx
                pts[idx, 1] = old_y + dy
                
                # Recompute only affected triangles
                new_min = np.inf
                for t_idx in tri_indices:
                    a, b, c = TRI_IDX[t_idx]
                    pa = pts[a]
                    pb = pts[b]
                    pc = pts[c]
                    v1 = pb - pa
                    v2 = pc - pa
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    area = abs(cross) * 0.5
                    areas[t_idx] = area
                    if area < new_min:
                        new_min = area
                
                # The global min is min of (unchanged areas min, new_min)
                # Unchanged areas: all except tri_indices
                if len(tri_indices) < 286:
                    mask = np.ones(286, dtype=bool)
                    mask[tri_indices] = False
                    unchanged_min = np.min(areas[mask])
                    global_min = min(new_min, unchanged_min)
                else:
                    global_min = new_min
                
                new_score = global_min / A_hull
                
                if new_score > current_score + 1e-12:
                    current_score = new_score
                    current_min = global_min
                    improved = True
                else:
                    # Revert
                    pts[idx, 0] = old_x
                    pts[idx, 1] = old_y
                    # Recompute affected triangles back
                    for t_idx in tri_indices:
                        a, b, c = TRI_IDX[t_idx]
                        pa = pts[a]
                        pb = pts[b]
                        pc = pts[c]
                        v1 = pb - pa
                        v2 = pc - pa
                        cross = v1[0] * v2[1] - v1[1] * v2[0]
                        areas[t_idx] = abs(cross) * 0.5
        
        if not improved:
            break
    
    return pts, current_score

def generate_initial_candidates(rng):
    """
    Generate diverse initial candidates based on geometric templates.
    """
    candidates = []
    
    # Template 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # Template 2: 1 center + 12 outer (regular 12-gon)
    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([center, outer])
    candidates.append(pts)
    
    # Template 3: 1 center + 6 inner + 6 outer
    center = np.array([0.0, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([center, inner, outer])
    candidates.append(pts)
    
    # Template 4: 2 center + 5 inner + 6 outer
    c1 = np.array([-0.1, 0.0])
    c2 = np.array([0.1, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([c1, c2, inner, outer])
    candidates.append(pts)
    
    # Template 5: Grid-like
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    pts = grid[:13]
    candidates.append(pts)
    
    # Template 6: Random
    pts = rng.uniform(-1, 1, size=(13, 2))
    candidates.append(pts)
    
    # Template 7: Fibonacci spiral
    golden_angle = 2.39996
    radii = np.sqrt(np.arange(13) / 13.0)
    angles = np.arange(13) * golden_angle
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    # Template 8: 3 concentric rings (4+4+5)
    angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    ring1 = 0.3 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    ring2 = 0.6 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    angles3 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ring3 = 1.0 * np.column_stack([np.cos(angles3), np.sin(angles3)])
    pts = np.vstack([ring1, ring2, ring3])
    candidates.append(pts)
    
    # Template 9: Center + 2 rings (1+5+7)
    center = np.array([0.0, 0.0])
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ring1 = 0.4 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    ring2 = 1.0 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    pts = np.vstack([center, ring1, ring2])
    candidates.append(pts)
    
    # Template 10: Perturbed regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    radii = 1.0 + 0.15 * rng.normal(0, 1, 13)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    return candidates

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses fast incremental local search with multi-scale refinement.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 15.0  # 15s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial candidates
    candidates = generate_initial_candidates(rng)
    
    # Add perturbed versions
    for base in candidates[:5]:
        for scale in [0.05, 0.1, 0.2]:
            perturbed = base + rng.normal(0, scale, size=base.shape)
            candidates.append(perturbed)
    
    # Shuffle
    rng.shuffle(candidates)
    
    n_candidates = len(candidates)
    
    # Phase 1: Quick evaluation + fast local search on all candidates
    # Time per candidate for initial pass
    phase1_deadline = start_time + (deadline - start_time) * 0.4
    time_per_candidate = max(5.0, (phase1_deadline - start_time) / n_candidates)
    
    for i, init_pts in enumerate(candidates):
        current_time = time.time()
        if current_time > phase1_deadline:
            break
        
        # Normalize
        init_pts_centered = init_pts - np.mean(init_pts, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        
        # Quick score
        score = compute_score(init_pts_scaled)
        
        # Fast local search with moderate step
        repair_deadline = min(phase1_deadline, current_time + time_per_candidate)
        pts_refined, score_refined = fast_local_search(
            init_pts_scaled,
            step_size=0.02,
            max_rounds=50,
            deadline=repair_deadline
        )
        
        if score_refined > best_score:
            best_score = score_refined
            best_points = pts_refined.copy()
    
    # Phase 2: Multi-scale refinement on the best candidate
    if best_points is not None:
        phase2_deadline = deadline - 10.0
        current_time = time.time()
        
        if current_time < phase2_deadline:
            # Multi-scale refinement: progressively smaller steps
            step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step_size in step_sizes:
                if time.time() > phase2_deadline:
                    break
                
                # Allocate time proportional to remaining budget
                remaining = phase2_deadline - time.time()
                # Each scale gets roughly equal time share of remaining
                n_remaining_scales = len(step_sizes) - step_sizes.index(step_size)
                time_this_scale = remaining / n_remaining_scales * 0.9
                
                pts_refined, score_refined = fast_local_search(
                    best_points,
                    step_size=step_size,
                    max_rounds=200,
                    deadline=time.time() + time_this_scale
                )
                
                if score_refined > best_score:
                    best_score = score_refined
                    best_points = pts_refined.copy()
    
    # Phase 3: Final ultra-fine polish
    if best_points is not None:
        current_time = time.time()
        final_deadline = deadline - 5.0
        
        if current_time < final_deadline:
            # Very fine steps with many rounds
            for step_size in [0.0001, 0.00005, 0.00002]:
                if time.time() > final_deadline:
                    break
                
                pts_refined, score_refined = fast_local_search(
                    best_points,
                    step_size=step_size,
                    max_rounds=500,
                    deadline=final_deadline
                )
                
                if score_refined > best_score:
                    best_score = score_refined
                    best_points = pts_refined.copy()
    
    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))])
        
    return best_points

# EVOLVE-BLOCK-END
