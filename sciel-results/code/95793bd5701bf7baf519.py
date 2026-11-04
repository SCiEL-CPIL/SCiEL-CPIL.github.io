"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _score(pts):
    """Compute min triangle area / convex hull area."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    n = len(pts)
    min_area = float('inf')
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area / hull_area

def _score_and_worst(pts):
    """Return (score, worst_triple_indices)."""
    if len(pts) < 3:
        return 0.0, (0, 1, 2)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0, (0, 1, 2)
    if hull_area < 1e-12:
        return 0.0, (0, 1, 2)
    n = len(pts)
    min_area = float('inf')
    worst = (0, 1, 2)
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                worst_k = j + 1 + int(np.argmin(areas))
                worst = (i, j, worst_k)
    return min_area / hull_area, worst

def _generate_ring_config(n_outer, n_inner, r_inner, theta_offset):
    """Generate points on two concentric rings."""
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + theta_offset
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _generate_3ring_config(n1, n2, n3, r2, r3, off2, off3):
    """Generate points on three concentric rings."""
    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    p1 = np.column_stack([np.cos(a1), np.sin(a1)])
    a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + off2
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + off3
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    return np.vstack([p1, p2, p3])

def _repair_worst(pts, worst, strength=0.02, iters=50):
    """Push points in the worst triangle apart to increase its area."""
    pts = pts.copy()
    i, j, k = worst
    for _ in range(iters):
        pi, pj, pk = pts[i], pts[j], pts[k]
        # Compute centroid of the triangle
        centroid = (pi + pj + pk) / 3.0
        # Push each point away from centroid
        for idx in [i, j, k]:
            direction = pts[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction /= norm
            pts[idx] += strength * direction
    return pts

def _local_refine(pts, deadline, max_rounds=200):
    """Iteratively repair the worst triangle."""
    best_pts = pts.copy()
    best_score = _score(pts)
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        score, worst = _score_and_worst(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        # Try different repair strengths
        improved = False
        for strength in [0.005, 0.01, 0.02, 0.05]:
            candidate = _repair_worst(pts, worst, strength, iters=10)
            c_score = _score(candidate)
            if c_score > score:
                pts = candidate
                improved = True
                break
        if not improved:
            # Try small random perturbation of worst points
            for _ in range(5):
                candidate = pts.copy()
                for idx in worst:
                    candidate[idx] += np.random.normal(0, 0.01, 2)
                c_score = _score(candidate)
                if c_score > score:
                    pts = candidate
                    improved = True
                    break
        if not improved:
            break
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_pts = None
    best_score = -np.inf
    
    # Phase 1: Search over structured ring configurations
    # 2-ring: 7 outer, 7 inner
    # 3-ring: various splits
    
    candidates = []
    
    # 2-ring configurations
    for r_inner in np.linspace(0.3, 0.7, 9):
        for theta_offset in np.linspace(0, np.pi/7, 5):
            pts = _generate_ring_config(7, 7, r_inner, theta_offset)
            candidates.append(pts)
    
    # 3-ring configurations: 5-4-5, 4-5-5, 5-5-4 splits
    for r2 in [0.4, 0.5, 0.6]:
        for r3 in [0.2, 0.3, 0.4]:
            for off2 in [0, np.pi/5, np.pi/4]:
                for off3 in [0, np.pi/5, np.pi/4]:
                    pts = _generate_3ring_config(5, 4, 5, r2, r3, off2, off3)
                    candidates.append(pts)
                    pts2 = _generate_3ring_config(4, 5, 5, r2, r3, off2, off3)
                    candidates.append(pts2)
    
    # Evaluate all candidates
    for pts in candidates:
        if time.time() > deadline - 60:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Phase 2: Local refinement via worst-triangle repair
    if best_pts is not None:
        refined = _local_refine(best_pts, deadline - 10)
        rs = _score(refined)
        if rs > best_score:
            best_score = rs
            best_pts = refined
    
    # Phase 3: Multi-start repair from perturbed best
    if best_pts is not None and time.time() < deadline - 30:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_pts + np.random.normal(0, 0.03, best_pts.shape)
            refined = _local_refine(perturbed, deadline - 5, max_rounds=100)
            rs = _score(refined)
            if rs > best_score:
                best_score = rs
                best_pts = refined
    
    # Phase 4: Try a few more structured families if time permits
    if time.time() < deadline - 20:
        # 2-ring with different split: 8 outer, 6 inner
        for r_inner in np.linspace(0.35, 0.65, 7):
            for theta_offset in np.linspace(0, np.pi/6, 4):
                pts = _generate_ring_config(8, 6, r_inner, theta_offset)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        
        # 2-ring: 6 outer, 8 inner
        for r_inner in np.linspace(0.5, 0.8, 7):
            for theta_offset in np.linspace(0, np.pi/8, 4):
                pts = _generate_ring_config(6, 8, r_inner, theta_offset)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
    
    # Final refinement
    if best_pts is not None and time.time() < deadline - 5:
        refined = _local_refine(best_pts, deadline - 1, max_rounds=100)
        rs = _score(refined)
        if rs > best_score:
            best_pts = refined
    
    if best_pts is None:
        # Fallback
        best_pts = _generate_ring_config(7, 7, 0.5, np.pi/7)
    
    return best_pts
# EVOLVE-BLOCK-END
