"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Sobol-Grid Initialization: Use a quasi-random low-discrepancy sequence (Sobol-like)
   to initialize points within the triangle, ensuring broad global coverage and
   avoiding the clustering issues of pure random initialization.
2. Local Nudge Refinement: 
   - Identify the set of "critical" points that form the current minimum area triangle(s).
   - Perform a local search by nudging these critical points in small steps.
   - This avoids the high-dimensional gradient noise of global optimizers and focuses
     computational effort on breaking the specific degeneracies limiting the score.
3. Multi-start with diverse Sobol seeds and refinement depths.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _sobol_init(rng, n_points):
    """
    Generate points using a Sobol-like quasi-random sequence.
    We use a simple halving method or a precomputed set of low-discrepancy points
    mapped to the triangle.
    For simplicity and robustness without scipy.stats.qmc (which might be slow to import or set up),
    we can use a deterministic grid with jitter or a simple recursive subdivision.
    Here we use a recursive subdivision of the triangle into 4 sub-triangles,
    picking the centroid of each, ensuring good spread.
    """
    pts = []
    # Vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, S3H])
    
    # Add vertices and midpoints first for a strong base
    mid_AB = (A + B) / 2
    mid_BC = (B + C) / 2
    mid_CA = (C + A) / 2
    centroid = (A + B + C) / 3
    
    base_pts = [A, B, C, mid_AB, mid_BC, mid_CA, centroid]
    
    # Add more points using a spiral or grid in barycentric coords
    # Let's use a simple deterministic spiral in barycentric space
    # Or just random with rejection, but seeded for reproducibility
    # To be "Sobol-like", let's use a low-discrepancy sequence for barycentric coords
    
    # Simple Sobol-like: use the van der Corput sequence for barycentric u, v
    # u = 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
    # v = 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ... (shifted)
    
    def van_der_corput(n, base=2):
        """Generate the first n terms of the van der Corput sequence."""
        seq = []
        for i in range(1, n + 1):
            f = 1.0 / base
            result = 0.0
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            seq.append(result)
        return seq
    
    n_seq = n_points + 10
    u_seq = van_der_corput(n_seq)
    v_seq = van_der_corput(n_seq, base=3) # Different base for low correlation
    
    for i in range(n_points):
        u = u_seq[i]
        v = v_seq[i]
        if u + v > 1:
            # Reflect into the triangle
            u = 1 - u
            v = 1 - v
            if u + v > 1:
                u = 1 - u - v
                v = 1 - u - v
                if u < 0 or v < 0:
                    continue
        # Barycentric: (1-u-v, u, v)
        # x = u*1 + v*0.5
        # y = v*S3H
        x = u + 0.5 * v
        y = S3H * v
        pts.append([x, y])
        
    # Combine base and sobol, then select best n_points by min distance
    all_candidates = np.array(base_pts + pts)
    # Remove duplicates
    all_candidates = np.unique(all_candidates, axis=0)
    
    if len(all_candidates) < n_points:
        # Fill with random
        while len(all_candidates) < n_points:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                x = u + 0.5 * v
                y = S3H * v
                all_candidates = np.vstack([all_candidates, [[x, y]]])
    
    # Select n_points from candidates to maximize min distance
    # Greedy selection
    selected = []
    # Start with centroid
    selected.append(all_candidates[np.argmin(np.linalg.norm(all_candidates - centroid, axis=1))])
    
    for _ in range(n_points - 1):
        best_pos = None
        best_min_dist = -1
        for cand in all_candidates:
            if cand in selected:
                continue
            dists = np.linalg.norm(np.array(selected) - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        if best_pos is None:
            break
        selected.append(best_pos)
        
    return np.array(selected)

def _local_nudge_refinement(pts, max_iter=100, step_sizes=None):
    """
    Refine points by nudging those involved in the minimum area triangle.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.001]
        
    current_pts = np.array(pts, dtype=float)
    current_score = _min_triangle_area(current_pts)
    
    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            if time.time() > DEADLINE - 5:
                return current_pts, current_score
            
            # Find the minimum area triangle
            p_i = current_pts[_i_idx]
            p_j = current_pts[_j_idx]
            p_k = current_pts[_k_idx]
            dx1 = p_j - p_i
            dx2 = p_k - p_i
            cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_area = np.min(areas)
            min_idx = np.argmin(areas)
            
            # Get the indices of the points in the min area triangle
            i, j, k = _triplets[min_idx]
            critical_indices = {i, j, k}
            
            # Try nudging each critical point in random directions
            for idx in critical_indices:
                # Try a few random nudges
                for _ in range(5):
                    angle = random.uniform(0, 2 * np.pi)
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    
                    new_pts = current_pts.copy()
                    new_pts[idx][0] += dx
                    new_pts[idx][1] += dy
                    new_pts = _project_to_triangle(new_pts)
                    
                    if not _check_inside(new_pts):
                        continue
                        
                    new_score = _min_triangle_area(new_pts)
                    if new_score > current_score + 1e-9:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        break # Restart inner loop with new min area
            
        if not improved:
            break
            
    return current_pts, current_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Sobol-Grid initialization and Local Nudge refinement.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start with different seeds for Sobol and refinement
    for seed in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Initialize
        init_pts = _sobol_init(rng, N)
        
        # Refine
        refined_pts, score = _local_nudge_refinement(init_pts)
        
        if score > best_score:
            best_score = score
            best_pts = refined_pts.copy()
            
        # Also try a pure random init with refinement
        init_pts_rand = np.array([
            [random.uniform(0, 1), random.uniform(0, S3H)]
            for _ in range(N)
        ])
        init_pts_rand = _project_to_triangle(init_pts_rand)
        refined_pts_rand, score_rand = _local_nudge_refinement(init_pts_rand)
        
        if score_rand > best_score:
            best_score = score_rand
            best_pts = refined_pts_rand.copy()
            
    if best_pts is None:
        # Fallback
        best_pts = _sobol_init(rng, N)
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
