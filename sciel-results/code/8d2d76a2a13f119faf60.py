import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2
DEADLINE = 145.0  # Safety margin

def _pairwise_dists_sq(pts):
    """Compute squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]

def _soft_objective(params, temp=0.1):
    """
    Soft objective using log-sum-exp for smooth min/max.
    Centers the points to remove translational invariance.
    """
    pts = params.reshape(N, D)
    # Center points
    pts = pts - np.mean(pts, axis=0)
    
    d2 = _pairwise_dists_sq(pts)
    d2 = np.maximum(d2, 1e-12)
    
    m_max = np.max(d2)
    # Soft max
    soft_max = m_max + temp * np.log(np.sum(np.exp((d2 - m_max) / temp)))
    
    m_min = np.min(d2)
    # Soft min
    soft_min = m_min - temp * np.log(np.sum(np.exp(-(d2 - m_min) / temp)))
    
    return soft_max / soft_min

def _exact_score(pts):
    """Compute exact score: (max_dist / min_dist)^2 = max_d2 / min_d2."""
    pts = pts - np.mean(pts, axis=0)
    d2 = _pairwise_dists_sq(pts)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2

def _generate_grid_pts():
    """Generate 4x4 grid points."""
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    return np.column_stack([X.flatten(), Y.flatten()])

def _generate_compressed_grid_pts():
    """Generate 4x4 grid with corners pulled in."""
    pts = _generate_grid_pts()
    center = np.array([1.5, 1.5])
    # Corners indices: 0, 3, 12, 15
    for idx in [0, 3, 12, 15]:
        pts[idx] = pts[idx] + 0.5 * (center - pts[idx])
    return pts

def _generate_hex_pts():
    """Generate 16 points from hexagonal lattice."""
    pts = []
    pts.append([0.0, 0.0])
    for k in range(6):
        angle = np.pi / 3 * k
        pts.append([np.cos(angle), np.sin(angle)])
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            if dist == 2.0:
                ring2.append((q, r))
    for q, r in ring2[:9]:
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        pts.append([x, y])
    return np.array(pts)

def _generate_triangular_pts():
    """Generate 4x4 triangular lattice points."""
    pts = []
    for i in range(4):
        for j in range(4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            pts.append([x, y])
    return np.array(pts)

def _generate_ring_pts(r1, r2, n1, n2, offset=0):
    """Generate points on two concentric rings."""
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + offset + np.pi / n2
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _refine_candidate(init_pts, t0):
    """
    Refine a single candidate using annealed L-BFGS-B.
    Returns (best_pts, best_score).
    """
    best_pts = init_pts.copy()
    best_score = _exact_score(init_pts)
    
    # Annealing schedule
    temps = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    current_params = init_pts.flatten()
    
    for temp in temps:
        if time.time() - t0 > DEADLINE - 5:
            break
            
        try:
            res = minimize(
                lambda p: _soft_objective(p, temp=temp),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_pts = res.x.reshape(N, D)
            new_score = _exact_score(new_pts)
            
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_params = res.x
        except Exception:
            continue
            
    return best_pts, best_score

def construct():
    t0 = time.time()
    
    # 1. Generate Static Candidates
    candidates = []
    candidates.append(_generate_grid_pts())
    candidates.append(_generate_compressed_grid_pts())
    candidates.append(_generate_hex_pts())
    candidates.append(_generate_triangular_pts())
    
    # Add ring-based candidates from inspiration 1
    candidates.append(_generate_ring_pts(0.4, 0.9, 8, 8))
    candidates.append(_generate_ring_pts(0.3, 0.85, 8, 8))
    candidates.append(_generate_ring_pts(0.5, 0.95, 8, 8))
    candidates.append(_generate_ring_pts(0.35, 0.8, 8, 8))
    candidates.append(_generate_ring_pts(0.4, 0.9, 6, 10))
    candidates.append(_generate_ring_pts(0.4, 0.9, 10, 6))
    
    # Add some random perturbations of the best static ones
    rng = np.random.RandomState(42)
    for base in [_generate_grid_pts(), _generate_hex_pts(), _generate_ring_pts(0.4, 0.9, 8, 8)]:
        for _ in range(3):
            perturbed = base + rng.randn(N, D) * 0.2
            candidates.append(perturbed)
            
    # 2. Global Search with Differential Evolution
    # Use a higher temperature for DE to make the landscape smoother
    de_candidates = []
    if time.time() - t0 < DEADLINE - 30:
        try:
            bounds = [(-10, 10)] * (N * D)
            de_res = differential_evolution(
                _soft_objective,
                bounds,
                args=(0.1,),
                maxiter=100,
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=False # We will polish later
            )
            if de_res.success or de_res.x is not None:
                de_candidates.append(de_res.x.reshape(N, D))
                # Also take a few random points from the final population if possible?
                # DE doesn't easily expose population history in simple call, 
                # but we can run a few more random restarts or just use the result.
                # Let's add a few random perturbations of the DE result
                for _ in range(3):
                    perturbed = de_res.x.reshape(N, D) + rng.randn(N, D) * 0.1
                    de_candidates.append(perturbed)
        except Exception:
            pass

    candidates.extend(de_candidates)
    
    best_pts = None
    best_score = np.inf
    
    # 3. Local Refinement on all candidates
    # Sort candidates by initial exact score to prioritize good ones?
    # Or just iterate.
    # To save time, we can limit the number of full refinements if time is tight.
    # But L-BFGS-B is fast.
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        try:
            pts, score = _refine_candidate(init_pts, t0)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # 4. Final Polish
    if best_pts is not None and time.time() - t0 < DEADLINE - 2:
        try:
            # Very fine refinement
            res_final = minimize(
                lambda p: _soft_objective(p, temp=0.0001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _exact_score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _generate_grid_pts()
        
    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
