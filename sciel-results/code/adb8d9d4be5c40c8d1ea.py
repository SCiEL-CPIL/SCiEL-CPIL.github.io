"""Combined packer: diverse init + force refinement + soft-min L-BFGS-B + PSO."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # seconds; leaves 15s margin under 150s kill


def _score(pts):
    """Compute the packing radius r for centres pts (N,2)."""
    n = len(pts)
    if n == 0:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms > 1.0):
        return 0.0
    # distance to boundary
    d = 1.0 - norms
    r_bound = np.min(d)
    # pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_bound, r_pair)


def _force_refine(pts, max_iter=20000, tol=1e-12):
    """
    Vectorized force-based local refinement.
    Points repel each other when closer than 2r and are pulled inward when too close to boundary.
    """
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    r = _score(pts)
    k = 5.0
    best_r = r
    best_pts = pts.copy()

    for it in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break

        forces = np.zeros_like(pts)

        # Vectorized pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1.0)  # avoid div by zero

        # Only apply force where dist < 2r
        mask = dist < 2.0 * r
        # Force magnitude: k * (2r - dist) / dist, applied along direction
        force_mag = np.zeros_like(dist)
        force_mag[mask] = k * (2.0 * r - dist[mask]) / dist[mask]
        # Sum forces: for each i, sum over j of force_mag[i,j] * diff[i,j]
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)

        # Boundary constraint: if |p| > 1 - r, push INWARD (toward origin)
        norms = np.linalg.norm(pts, axis=1)
        boundary_mask = norms > (1.0 - r)
        if np.any(boundary_mask):
            excess = norms[boundary_mask] - (1.0 - r)
            safe_norms = np.where(norms[boundary_mask] > 1e-10, norms[boundary_mask], 1.0)
            forces[boundary_mask] -= k * 2.0 * excess / safe_norms * pts[boundary_mask]

        # Adaptive step size with line search
        step = 0.02
        improved = False
        new_r = r
        for _ in range(15):
            pts_new = pts + step * forces

            # Clip to ensure inside container
            norms_new = np.linalg.norm(pts_new, axis=1)
            max_norm = 1.0 - r * 0.9
            mask_out = norms_new > max_norm
            if np.any(mask_out):
                safe_norms_new = np.where(norms_new[mask_out] > 1e-10, norms_new[mask_out], 1.0)
                pts_new[mask_out] *= max_norm / safe_norms_new

            new_r = _score(pts_new)
            if new_r >= r + 1e-12:
                improved = True
                break
            elif new_r >= r - 1e-10:
                # Accept small decrease to escape plateaus
                improved = True
                break
            step *= 0.5

        if not improved:
            break

        if abs(new_r - r) < tol:
            break

        pts = pts_new
        r = new_r
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts


def _soft_min_lbfgs(pts, betas=None):
    """
    Use L-BFGS-B with a soft-min (log-sum-exp) formulation to maximize the packing radius.
    This provides a smooth objective with gradient, enabling faster convergence than
    the non-smooth min-based objective.
    """
    if betas is None:
        betas = [20, 50, 100, 200, 400]

    def make_obj_func(beta_val):
        """Create objective function with given beta for soft-min."""
        def obj_func(flat_x):
            x = flat_x.reshape(N, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(d, np.inf)
            dist_boundary = 1.0 - np.linalg.norm(x, axis=1)

            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu] / 2.0  # half-distance
            d_bnds = dist_boundary

            all_d = np.concatenate([d_pairs, d_bnds])

            exponents = -beta_val * all_d
            max_exp = np.max(exponents)

            if not np.isfinite(max_exp):
                return 100.0, np.zeros_like(flat_x)

            sum_exp = np.sum(np.exp(exponents - max_exp))
            log_sum = max_exp + np.log(sum_exp)
            obj = log_sum / beta_val

            weights = np.exp(exponents - max_exp) / sum_exp
            grad = np.zeros_like(x)

            n_pairs = len(d_pairs)
            w_pairs = weights[:n_pairs]
            w_bnds = weights[n_pairs:]

            i_idx, j_idx = iu
            d_safe = np.where(d_pairs > 1e-8, d_pairs, 1e-8)
            vecs = x[i_idx] - x[j_idx]
            norm_vecs = vecs / d_safe[:, np.newaxis]

            # Gradient contribution from pairwise distances
            # d = ||x_i - x_j|| / 2, so grad_{x_i} d = (x_i - x_j) / (2 * ||x_i - x_j||)
            # But we're using d_pairs = d[iu] / 2.0 already, so:
            # grad_{x_i} d_pairs = (x_i - x_j) / (2 * ||x_i - x_j||)
            # Let's recompute: d_pairs = ||x_i - x_j|| / 2
            # grad_{x_i} ||x_i - x_j|| = (x_i - x_j) / ||x_i - x_j||
            # So grad_{x_i} d_pairs = (x_i - x_j) / (2 * ||x_i - x_j||)
            
            # Actually, let's be more careful:
            # all_d includes d_pairs (= dist/2) and d_bnds (= 1 - ||x_i||)
            # For pairwise: f_i = d_pairs[i,j] = ||x_i - x_j|| / 2
            # grad_{x_i} f_i = (x_i - x_j) / (2 * ||x_i - x_j||)
            # grad_{x_j} f_i = (x_j - x_i) / (2 * ||x_i - x_j||)
            
            # For boundary: f_i = d_bnds[i] = 1 - ||x_i||
            # grad_{x_i} f_i = -x_i / ||x_i||
            
            # The soft-min objective is: obj = (1/beta) * log(sum(exp(-beta * all_d)))
            # grad obj = -weights * grad(all_d)
            # where weights are the softmax weights
            
            # For pairwise terms:
            # grad_{x_i} obj += -w_pairs * (x_i - x_j) / (2 * d_full)
            # where d_full = ||x_i - x_j|| = 2 * d_pairs
            
            d_full = 2.0 * d_pairs
            d_full_safe = np.where(d_full > 1e-8, d_full, 1e-8)
            vecs_full = x[i_idx] - x[j_idx]
            norm_vecs_full = vecs_full / d_full_safe[:, np.newaxis]

            np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * norm_vecs_full)
            np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * norm_vecs_full)

            # For boundary terms:
            norms = np.linalg.norm(x, axis=1)
            norms_safe = np.where(norms > 1e-8, norms, 1e-8)
            norm_x = x / norms_safe[:, np.newaxis]

            # grad_{x_i} d_bnds[i] = -x_i / ||x_i||
            # So grad obj += -w_bnds * (-x_i / ||x_i||) = w_bnds * x_i / ||x_i||
            np.add.at(grad, np.arange(N), w_bnds[:, np.newaxis] * norm_x)

            return obj, grad.flatten()
        return obj_func

    best_pts = pts.copy()
    best_r = _score(pts)

    for beta in betas:
        if time.time() > DEADLINE - 5.0:
            break

        try:
            obj_func = make_obj_func(beta)
            result = minimize(obj_func, best_pts.flatten(), method='L-BFGS-B', jac=True,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.x is not None:
                new_pts = result.x.reshape(N, 2)
                new_r = _score(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    return best_pts


def _psO_optimize(pts, n_particles=15, max_iter=40):
    """Particle Swarm Optimization to escape local minima."""
    if time.time() > DEADLINE - 20.0:
        return pts
    
    pts = pts.copy().astype(np.float64)
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Initialize particles
    particles = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    pbests = np.zeros((n_particles, N, 2))
    pbest_scores = np.zeros(n_particles)
    
    for i in range(n_particles):
        particles[i] = pts + 0.01 * np.random.randn(N, 2)
        # Ensure inside container
        norms = np.linalg.norm(particles[i], axis=1)
        mask = norms > 0.99
        if np.any(mask):
            particles[i][mask] *= 0.99 / norms[mask]
        pbests[i] = particles[i].copy()
        pbest_scores[i] = _score(particles[i])
    
    gbest = best_pts.copy()
    gbest_score = best_r
    
    w = 0.9  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10.0:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(N, 2)
            r2 = np.random.rand(N, 2)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            
            # Clamp velocity
            v_max = 0.05
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # Update position
            particles[i] += velocities[i]
            
            # Ensure inside container
            norms = np.linalg.norm(particles[i], axis=1)
            mask = norms > 0.99
            if np.any(mask):
                particles[i][mask] *= 0.99 / norms[mask]
            
            # Update personal best
            score_i = _score(particles[i])
            if score_i > pbest_scores[i]:
                pbest_scores[i] = score_i
                pbests[i] = particles[i].copy()
                if score_i > gbest_score:
                    gbest_score = score_i
                    gbest = particles[i].copy()
        
        # Update global best
        if gbest_score > best_r:
            best_r = gbest_score
            best_pts = gbest.copy()
    
    return best_pts


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, pick best N."""
    best_pts = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.25, 30):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            idx = np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:N]
            pts = points[idx]
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    return best_pts, best_r


def _ring_init(n_rings, r_start, r_end):
    """Initialize with concentric rings."""
    pts = [[0.0, 0.0]]
    for ring in range(1, n_rings + 1):
        r = r_start + (r_end - r_start) * (ring - 1) / max(n_rings - 1, 1)
        if ring == 1:
            n_pts = 6
        else:
            n_pts = ring * 6
        n_pts = min(n_pts, N - len(pts))
        if n_pts <= 0:
            break
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts[:N])
    if len(pts) < N:
        return None
    return pts


def _diverse_init():
    """Generate diverse initializations and return the best few."""
    candidates = []

    # Hexagonal grids with various spacings
    for s in np.linspace(0.10, 0.25, 25):
        k = int(1.0 / s) + 3
        rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - s / 2.0]
        if len(points) >= N:
            idx = np.argsort(np.hypot(points[:, 0], points[:, 1]), kind="stable")[:N]
            pts = points[idx]
            candidates.append(pts)

    # Ring-based inits
    for n_rings in [5, 6, 7, 8, 9]:
        for r_start in [0.05, 0.1, 0.15, 0.2]:
            pts = _ring_init(n_rings, r_start, 0.9)
            if pts is not None:
                candidates.append(pts)

    # Random inits
    for _ in range(10):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None] * 0.95
        candidates.append(pts)

    # Evaluate all candidates and pick the best few
    scored_candidates = []
    for c in candidates:
        if len(c) == N:
            r = _score(c)
            scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: -x[0])

    top_candidates = [c for r, c in scored_candidates[:5]]
    if not top_candidates:
        top_candidates = [np.random.rand(N, 2) * 0.8]

    return top_candidates


def construct():
    t0 = time.time()
    
    # Generate diverse initializations
    top_candidates = _diverse_init()
    
    best_pts = None
    best_r = 0.0

    for cand in top_candidates:
        if time.time() > DEADLINE - 30.0:
            break

        # Force-based refinement
        if time.time() < DEADLINE - 25.0:
            pts = _force_refine(cand, max_iter=15000, tol=1e-13)
        else:
            pts = cand

        # Soft-min L-BFGS-B refinement
        if time.time() < DEADLINE - 15.0:
            pts = _soft_min_lbfgs(pts)

        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = top_candidates[0]
        best_r = _score(best_pts)

    # PSO to escape local minima
    if time.time() < DEADLINE - 20.0:
        pts_pso = _psO_optimize(best_pts, n_particles=15, max_iter=40)
        if _score(pts_pso) > best_r:
            best_r = _score(pts_pso)
            best_pts = pts_pso

    # Final force refinement pass
    if time.time() < DEADLINE - 10.0:
        best_pts = _force_refine(best_pts, max_iter=10000, tol=1e-14)

    # Final soft-min L-BFGS polish
    if time.time() < DEADLINE - 5.0:
        best_pts = _soft_min_lbfgs(best_pts, betas=[100, 200, 400])

    # Ensure validity: clip any point outside
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        safe_norms = np.where(norms[mask] > 1e-10, norms[mask], 1.0)
        best_pts[mask] /= safe_norms[:, np.newaxis] * 1.0001

    return best_pts
