# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist

def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning should be +-2*pi
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True

def _vertex_score_fast(vertex, polygon):
    """Compute the score for a single vertex (optimized)."""
    n = len(polygon)
    diffs = polygon - vertex
    dists_sq = np.einsum('ij,ij->i', diffs, diffs)
    dists = np.sqrt(dists_sq)
    idx = np.argmin(dists)
    dists[idx] = np.inf
    dists_sorted = np.sort(dists)
    
    if dists_sorted[0] < 0.05:
        return 0.0
    
    gaps = dists_sorted[1:] - dists_sorted[:-1]
    k = np.argmin(gaps)
    D = (dists_sorted[k] + dists_sorted[k + 1]) / 2
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        four_dists = dists_sorted[0:4]
    elif k == 17:
        four_dists = dists_sorted[15:19]
    else:
        four_dists = dists_sorted[k - 1:k + 3]
    
    ratios_a = four_dists / D
    ratios_b = D / four_dists
    raw = np.mean(np.maximum(ratios_a, ratios_b))
    
    return min(1.0, 1.0 / raw)

def _polygon_score_fast(polygon):
    """Compute the minimum vertex score over all vertices (vectorized)."""
    n = len(polygon)
    # Compute all pairwise distances at once
    diffs = polygon[:, np.newaxis, :] - polygon[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diffs, diffs)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    scores = np.empty(n)
    for i in range(n):
        dists_i = np.sort(dists[i])
        
        if dists_i[0] < 0.05:
            scores[i] = 0.0
            continue
        
        gaps = dists_i[1:] - dists_i[:-1]
        k = np.argmin(gaps)
        D = (dists_i[k] + dists_i[k + 1]) / 2
        
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        if k == 0:
            four_dists = dists_i[0:4]
        elif k == 17:
            four_dists = dists_i[15:19]
        else:
            four_dists = dists_i[k - 1:k + 3]
        
        ratios_a = four_dists / D
        ratios_b = D / four_dists
        raw = np.mean(np.maximum(ratios_a, ratios_b))
        scores[i] = min(1.0, 1.0 / raw)
    
    return np.min(scores)

def _score_from_radii(radii):
    """Compute score given radii for fixed angles."""
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.maximum(radii, 0.1)
    polygon = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    polygon = _normalize(polygon)
    
    if not _is_valid_convex(polygon):
        return -1.0
    
    return _polygon_score_fast(polygon)

def _local_refinement(radii, step_size, n_steps, deadline):
    """Perform local refinement with adaptive step sizes."""
    n = len(radii)
    best_radii = radii.copy()
    best_score = _score_from_radii(best_radii)
    if best_score < 0:
        return radii, 0.0
    
    current_step = step_size
    for step in range(n_steps):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 10.0:
                break
            
            # Try positive and negative perturbations
            for direction in [1, -1]:
                trial = best_radii.copy()
                trial[i] += direction * current_step
                trial = np.maximum(trial, 0.1)
                
                score = _score_from_radii(trial)
                if score > best_score:
                    best_radii = trial
                    best_score = score
                    improved = True
        
        if not improved:
            current_step *= 0.5
            if current_step < 1e-4:
                break
        else:
            current_step *= 0.8
    
    return best_radii, best_score

def _broad_exploration(angles, deadline):
    """Phase 1: Broad exploration with diverse initial candidates."""
    n = len(angles)
    candidates = []
    
    # Regular polygon
    candidates.append(np.ones(n))
    
    # Symmetric wavy patterns
    for freq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for amp in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            radii = 1.0 + amp * np.cos(freq * angles)
            radii = np.maximum(radii, 0.3)
            candidates.append(radii)
    
    # Random symmetric patterns
    rng = np.random.default_rng(123)
    for _ in range(30):
        # Random low-frequency pattern
        coeffs = rng.normal(0, 0.15, 5)
        radii = 1.0
        for f in range(1, 5):
            radii += coeffs[f-1] * np.cos(f * angles)
        radii = np.maximum(radii, 0.3)
        candidates.append(radii)
    
    # Evaluate all candidates
    best_radii = None
    best_score = -1.0
    
    for radii in candidates:
        if time.time() > deadline - 60.0:
            break
        score = _score_from_radii(radii)
        if score > best_score:
            best_score = score
            best_radii = radii.copy()
    
    if best_radii is None:
        best_radii = np.ones(n)
        best_score = _score_from_radii(best_radii)
    
    return best_radii, best_score

def _targeted_refinement(radii, deadline):
    """Phase 2: Targeted refinement using scipy optimizer."""
    n = len(radii)
    
    def objective(radii):
        if time.time() > deadline - 10.0:
            return 1e9
        score = _score_from_radii(radii)
        if score < 0:
            return 1e9
        return -score
    
    try:
        res = minimize(objective, radii, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < 1e8:  # Valid result
            return res.x, -res.fun
    except Exception:
        pass
    
    return radii, _score_from_radii(radii)

def _diverse_candidates(radii, best_score, deadline):
    """Phase 3: Generate diverse candidates from current best."""
    n = len(radii)
    rng = np.random.default_rng(456)
    
    best_radii = radii.copy()
    best_s = best_score
    
    for _ in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Different perturbation strategies
        strategy = rng.integers(0, 3)
        
        if strategy == 0:
            # Small random noise
            noise = rng.normal(0, 0.03, n)
            trial = best_radii + noise
        elif strategy == 1:
            # Perturb a few random vertices
            idx = rng.choice(n, size=3, replace=False)
            trial = best_radii.copy()
            trial[idx] += rng.normal(0, 0.1, 3)
        else:
            # Frequency-domain perturbation
            freqs = rng.choice([1, 2, 3, 4, 5], size=2)
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            pert = np.zeros(n)
            for f in freqs:
                pert += rng.normal(0, 0.05) * np.cos(f * angles)
            trial = best_radii + pert
        
        trial = np.maximum(trial, 0.3)
        score = _score_from_radii(trial)
        
        if score > best_s:
            best_s = score
            best_radii = trial.copy()
    
    return best_radii, best_s

def _final_polish(radii, deadline):
    """Phase 4: Final polish with very fine steps."""
    return _local_refinement(radii, 0.005, 30, deadline)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses multi-phase structured optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Phase 1: Broad exploration
    best_radii, best_score = _broad_exploration(angles, deadline)
    
    # Phase 2: Targeted refinement
    if time.time() < deadline - 60.0:
        refined_radii, refined_score = _targeted_refinement(best_radii, deadline)
        if refined_score > best_score:
            best_radii = refined_radii
            best_score = refined_score
    
    # Phase 3: Diverse candidates
    if time.time() < deadline - 40.0:
        diverse_radii, diverse_score = _diverse_candidates(best_radii, best_score, deadline)
        if diverse_score > best_score:
            best_radii = diverse_radii
            best_score = diverse_score
    
    # Phase 4: Final polish
    if time.time() < deadline - 20.0:
        polished_radii, polished_score = _final_polish(best_radii, deadline)
        if polished_score > best_score:
            best_radii = polished_radii
            best_score = polished_score
    
    # Reconstruct polygon
    best_radii = np.maximum(best_radii, 0.1)
    p = np.stack([best_radii * np.cos(angles), best_radii * np.sin(angles)], axis=1)
    best_polygon = _normalize(p)
    
    # Final validity check
    if not _is_valid_convex(best_polygon):
        # Fallback to regular polygon
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(p)
    
    return best_polygon

# EVOLVE-BLOCK-END
