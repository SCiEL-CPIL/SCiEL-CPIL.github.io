# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D maximizing (d_min/d_max)^2.
    Uses a multi-stage approach:
    1. Generate diverse initializations (Fibonacci, random, icosahedron-based).
    2. Gradient descent on a smooth proxy (minimizing sum of inverse powers of distances)
       on the unit sphere. This is efficient and helps escape shallow local minima.
    3. Simulated Annealing (SA) on the true objective to escape local minima 
       that gradient methods might get stuck in.
    4. Nelder-Mead refinement on the top candidates for final polishing.
    """
    n = 14
    d = 3
    t0 = time.time()
    deadline = t0 + 580.0  # Safety margin before 630s limit

    def remaining():
        return deadline - time.time()

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmax <= 1e-15 or dmin <= 1e-15:
            return 0.0
        return (dmin / dmax) ** 2

    def fib_sphere(n_pts):
        phi = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(n_pts):
            y = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(max(0.0, 1.0 - y * y))
            th = phi * i
            pts.append([np.cos(th) * r, y, np.sin(th) * r])
        return np.array(pts)

    def generate_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        pts = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        return np.array(pts)

    def grad_desc_sphere(pts, iters=2000, lr=0.008, mom=0.9, k=4):
        """
        Gradient descent on the sphere to minimize sum_{i!=j} 1/d_ij^k.
        """
        p = pts.copy()
        v = np.zeros_like(p)
        
        for it in range(iters):
            if remaining() < 5.0:
                break
            
            diff = p[:, None, :] - p[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, 1e-12)
            
            # w_ij = 1/d_ij^(k+2)
            if k % 2 == 0:
                w = 1.0 / (d2 ** ((k // 2) + 1))
            else:
                w = 1.0 / (d2 ** (k // 2 + 1)) * (1.0 / np.sqrt(d2))
            
            np.fill_diagonal(w, 0.0)
            
            sum_w = w.sum(axis=1, keepdims=True)
            sum_wp = (w[:, :, None] * p).sum(axis=1)
            
            grad = -k * (p * sum_w - sum_wp)
            
            v = mom * v - lr * grad
            p = p + v
            p = normalize(p)
            
        return p

    def simulated_annealing(pts, max_iters=5000, T_start=0.1, T_end=0.001):
        """
        Simulated Annealing on the true objective (min/max ratio squared).
        Moves are random perturbations of points, followed by re-normalization.
        """
        p = pts.copy()
        current_score = score(p)
        best_p = p.copy()
        best_score = current_score
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if remaining() < 5.0:
                break
            
            # Generate a move: perturb a random point or all points slightly
            # Randomly select a subset of points to move to allow local adjustments
            # Here we just move all points slightly for simplicity, or a random subset.
            # Let's try moving a random subset (1-3 points) with random step size.
            num_moved = np.random.randint(1, 4)
            idx = np.random.choice(n, num_moved, replace=False)
            
            # Step size decreases with temperature
            step_size = 0.05 * T / T_start
            
            p_new = p.copy()
            p_new[idx] += np.random.randn(num_moved, d) * step_size
            p_new = normalize(p_new)
            
            new_score = score(p_new)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_score = new_score
                p = p_new
                if current_score > best_score:
                    best_score = current_score
                    best_p = p.copy()
            
            T *= cooling
            
        return best_p, best_score

    def nm_refine(pts, maxfev=2000):
        """
        Nelder-Mead refinement on the true objective.
        """
        def obj(x):
            if remaining() < 2.0:
                return 1e6
            pp = x.reshape(n, d)
            pp = normalize(pp)
            s = score(pp)
            return -s if s > 0 else 1e6

        x0 = pts.flatten()
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxfev': maxfev, 'xatol': 1e-7, 'fatol': 1e-7})
            pp = normalize(res.x.reshape(n, d))
            return pp
        except Exception:
            return pts

    # --- Generate diverse initializations ---
    inits = []
    inits.append(fib_sphere(n))
    
    # Icosahedron + 2 random points on sphere
    ico = generate_icosahedron()
    for _ in range(5):
        p1 = np.random.randn(3)
        p2 = np.random.randn(3)
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)
        cand = np.vstack([ico, [p1, p2]])
        inits.append(normalize(cand))
        
    # Random points on sphere
    for _ in range(10):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))
        
    # Perturbed Fibonacci
    for eps in [0.05, 0.1, 0.2]:
        pts = fib_sphere(n) + np.random.randn(n, d) * eps
        inits.append(normalize(pts))

    best_score = -1.0
    best_points = inits[0]

    # --- Phase 1: Gradient Descent ---
    # Use GD to quickly improve initializations
    gd_results = []
    for init in inits:
        if remaining() < 60.0:
            break
        
        # Try a few different k and lr values
        for k in [2, 4, 6]:
            for lr in [0.005, 0.01, 0.02]:
                if remaining() < 30.0:
                    break
                pts = grad_desc_sphere(init, iters=1000, lr=lr, mom=0.9, k=k)
                s = score(pts)
                gd_results.append((s, pts))
                if s > best_score:
                    best_score = s
                    best_points = pts

    # Sort by score and keep top distinct candidates
    gd_results.sort(key=lambda x: x[0], reverse=True)
    top = []
    for s, pts in gd_results:
        dup = False
        for _, p2 in top:
            if np.max(np.abs(pts - p2)) < 1e-3:
                dup = True
                break
        if not dup:
            top.append((s, pts))
        if len(top) >= 10:
            break

    # --- Phase 2: Simulated Annealing ---
    # Apply SA to the top GD results to escape local minima
    sa_results = []
    for s, pts in top:
        if remaining() < 30.0:
            break
        
        # Run SA for a few iterations
        sa_pts, sa_score = simulated_annealing(pts, max_iters=2000, T_start=0.05, T_end=0.001)
        sa_results.append((sa_score, sa_pts))
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts

    # Sort SA results
    sa_results.sort(key=lambda x: x[0], reverse=True)
    top_sa = []
    for s, pts in sa_results:
        dup = False
        for _, p2 in top_sa:
            if np.max(np.abs(pts - p2)) < 1e-3:
                dup = True
                break
        if not dup:
            top_sa.append((s, pts))
        if len(top_sa) >= 5:
            break

    # --- Phase 3: Nelder-Mead Refinement ---
    # Refine the best candidates from SA (and maybe GD if SA failed)
    candidates = top_sa
    if not candidates:
        candidates = top[:5]

    for s, pts in candidates:
        if remaining() < 20.0:
            break
        
        refined = nm_refine(pts, maxfev=1500)
        rs = score(refined)
        if rs > best_score:
            best_score = rs
            best_points = refined

        # Try a second NM pass from a slightly perturbed version
        if remaining() > 30.0:
            perturbed = normalize(pts + np.random.randn(n, d) * 0.02)
            refined2 = nm_refine(perturbed, maxfev=1000)
            rs2 = score(refined2)
            if rs2 > best_score:
                best_score = rs2
                best_points = refined2

    # --- Phase 4: Final Polish ---
    if remaining() > 40.0:
        for _ in range(3):
            if remaining() < 20.0:
                break
            perturbed = normalize(best_points + np.random.randn(n, d) * 0.01)
            refined = nm_refine(perturbed, maxfev=1500)
            rs = score(refined)
            if rs > best_score:
                best_score = rs
                best_points = refined

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
