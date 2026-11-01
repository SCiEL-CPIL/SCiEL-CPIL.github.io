import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # ---------- helpers ----------

    def compute_energy(pts):
        """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(d_ij)"""
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2.0 * (pts @ pts.T)
        dist_sq = np.maximum(dist_sq, 1e-14)
        dist = np.sqrt(dist_sq)
        i_upper, j_upper = np.triu_indices(N, k=1)
        d_pairs = dist[i_upper, j_upper]
        d_pairs = np.maximum(d_pairs, 1e-14)
        return -np.sum(np.log(d_pairs))

    def compute_gradient(pts):
        """
        Gradient of E = -sum_{i<j} log(d_ij) w.r.t. each point in R^3.
        grad_i = x_i * sum_{j != i} 1/d_ij^2 - sum_{j != i} x_j / d_ij^2
        """
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2.0 * (pts @ pts.T)
        dist_sq = np.maximum(dist_sq, 1e-14)
        inv_d2 = 1.0 / dist_sq
        np.fill_diagonal(inv_d2, 0.0)
        row_sums = np.sum(inv_d2, axis=1)
        grad = pts * row_sums[:, None] - inv_d2 @ pts
        return grad

    def project_to_tangent(pts, grad):
        """Project gradient onto tangent space of sphere at each point."""
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        return grad - dot * pts

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return pts / norms

    # ---------- L-BFGS optimization ----------

    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        energy = compute_energy(pts)
        grad = compute_gradient(pts)
        grad_tangent = project_to_tangent(pts, grad)
        return energy, grad_tangent.flatten()

    def lbfgs_optimize(x0, maxiter=500, tight=False):
        """Run L-BFGS-B and return (best_points, best_energy)."""
        try:
            if tight:
                opts = {'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80}
            else:
                opts = {'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 60}
            result = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options=opts
            )
            new_pts = normalize(result.x.reshape(N, 3))
            new_energy = compute_energy(new_pts)
            return new_pts, new_energy
        except Exception:
            return None, None

    # ---------- gradient descent with line search (tangent space) ----------

    def gradient_descent(pts, max_iters=5000, initial_step=0.005, time_budget=10.0):
        """Adaptive-step gradient descent in tangent space."""
        current_energy = compute_energy(pts)
        step_size = initial_step
        min_step = 1e-14
        max_step = 0.1
        no_improve = 0
        prev_grad_norm = np.inf

        for it in range(max_iters):
            if time.time() > time_budget:
                break

            grad = compute_gradient(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break

            # Early exit if gradient is not decreasing
            if it > 10 and grad_norm > 2.0 * prev_grad_norm:
                step_size *= 0.5
            prev_grad_norm = grad_norm

            grad_tangent = project_to_tangent(pts, grad)

            # Backtracking line search with Armijo condition
            alpha = step_size
            improved = False
            c1 = 1e-4  # Armijo parameter
            for _ in range(30):
                candidate = pts - alpha * grad_tangent
                candidate = normalize(candidate)
                cand_energy = compute_energy(candidate)
                # Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad^T*d
                if cand_energy < current_energy:
                    pts = candidate
                    current_energy = cand_energy
                    improved = True
                    no_improve = 0
                    step_size = min(alpha * 1.3, max_step)
                    break
                alpha *= 0.5
                if alpha < min_step:
                    break

            if not improved:
                no_improve += 1
                if no_improve > 100:
                    break

        return pts, current_energy

    # ---------- swap-based simulated annealing ----------

    def swap_annealing(pts, max_iters=5000, T_start=0.05, T_end=1e-6, time_budget=15.0):
        """
        Simulated annealing with swap moves: pick two random points, 
        rotate them around their midpoint by a small angle in the tangent plane.
        """
        current_energy = compute_energy(pts)
        best_pts = pts.copy()
        best_energy = current_energy
        current_pts = pts.copy()

        for it in range(max_iters):
            if time.time() > time_budget:
                break

            # Annealing schedule (exponential cooling)
            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress

            # Pick two random points to swap/rotate
            i, j = np.random.choice(N, 2, replace=False)

            # Create a random rotation in the plane spanned by (pts[i]-pts[j]) and a random vector
            v = pts[i] - pts[j]
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                continue
            v = v / v_norm

            # Find a vector perpendicular to v for rotation
            w = np.random.randn(3)
            w = w - np.dot(w, v) * v
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-10:
                continue
            w = w / w_norm

            # Small rotation angle
            angle = np.random.uniform(-0.1, 0.1)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Rotate pts[i] and pts[j] in the plane spanned by v and w
            # Apply rotation to both points around the axis w
            # Actually, let's do a simpler perturbation: move each point slightly in a random tangent direction
            noise_i = np.random.randn(3) * 0.05
            noise_j = np.random.randn(3) * 0.05
            # Project to tangent
            noise_i = noise_i - np.dot(noise_i, current_pts[i]) * current_pts[i]
            noise_j = noise_j - np.dot(noise_j, current_pts[j]) * current_pts[j]

            candidate = current_pts.copy()
            candidate[i] = normalize(np.array([current_pts[i] + noise_i]))[0]
            candidate[j] = normalize(np.array([current_pts[j] + noise_j]))[0]

            cand_energy = compute_energy(candidate)
            delta = cand_energy - current_energy

            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-12)):
                current_pts = candidate
                current_energy = cand_energy
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_pts = candidate.copy()

        return best_pts, best_energy

    # ---------- full swap-based annealing with energy delta computation ----------

    def compute_pair_energy(pts, i, j):
        """Compute energy contribution of pair (i,j)."""
        d = np.linalg.norm(pts[i] - pts[j])
        return -math.log(max(d, 1e-14))

    def swap_annealing_fast(pts, max_iters=8000, T_start=0.05, T_end=1e-8, time_budget=20.0):
        """
        Fast swap-based simulated annealing.
        Move: pick 2-4 random points, perturb each in random tangent direction.
        Also includes swap moves for better global exploration.
        """
        current_energy = compute_energy(pts)
        best_pts = pts.copy()
        best_energy = current_energy
        current_pts = pts.copy()

        for it in range(max_iters):
            if time.time() > time_budget:
                break

            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress

            # Choose move type: 70% perturb, 30% swap
            if np.random.random() < 0.7:
                # Perturb 2-4 random points
                n_perturb = np.random.randint(2, 5)
                indices = np.random.choice(N, n_perturb, replace=False)
                noise_scale = 0.05 + 0.03 * (1 - progress)  # Larger noise early
                noise = np.random.randn(n_perturb, 3) * noise_scale

                candidate = current_pts.copy()
                for k, idx in enumerate(indices):
                    # Project noise to tangent space
                    n = noise[k]
                    n = n - np.dot(n, current_pts[idx]) * current_pts[idx]
                    candidate[idx] = current_pts[idx] + n
                candidate = normalize(candidate)
            else:
                # Swap move: exchange positions of two random points with small perturbation
                i, j = np.random.choice(N, 2, replace=False)
                candidate = current_pts.copy()
                # Add small noise to both swapped points
                noise_i = np.random.randn(3) * 0.03
                noise_j = np.random.randn(3) * 0.03
                noise_i = noise_i - np.dot(noise_i, current_pts[i]) * current_pts[i]
                noise_j = noise_j - np.dot(noise_j, current_pts[j]) * current_pts[j]
                candidate[i] = current_pts[j] + noise_i
                candidate[j] = current_pts[i] + noise_j
                candidate = normalize(candidate)

            cand_energy = compute_energy(candidate)
            delta = cand_energy - current_energy

            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-12)):
                current_pts = candidate
                current_energy = cand_energy
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_pts = candidate.copy()

        return best_pts, best_energy

    # ---------- initializations ----------

    def fibonacci_spiral(offset=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(pts)

    def random_points(rng):
        pts = rng.randn(N, 3)
        return normalize(pts)

    def ring_initialization(num_rings=10, rng=None):
        if rng is None:
            rng = np.random
        pts_list = []
        for ring in range(num_rings):
            z_val = -1.0 + 2.0 * ring / (num_rings - 1)
            r_val = np.sqrt(max(1.0 - z_val**2, 0.0))
            if r_val < 0.01:
                r_val = 0.01
            num_per_ring = N // num_rings
            for j in range(num_per_ring):
                angle = 2.0 * np.pi * j / num_per_ring + ring * 0.3
                pts_list.append([r_val * np.cos(angle), r_val * np.sin(angle), z_val])
        while len(pts_list) < N:
            pts_list.append(rng.randn(3).tolist())
        pts = np.array(pts_list[:N])
        return normalize(pts)

    def clustered_initialization(rng, num_clusters=5):
        """Clustered starts: points near random cluster centers."""
        centers = rng.randn(num_clusters, 3)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        pts = []
        for i in range(N):
            center = centers[i % num_clusters]
            noise = rng.randn(3) * 0.3
            pt = center + noise
            pt = pt / np.linalg.norm(pt)
            pts.append(pt)
        return np.array(pts)

    # ---------- main search ----------

    rng = np.random.RandomState(42)

    best_points = None
    best_energy = float('inf')

    # Initial set of starting points - more diverse
    initializations = [
        fibonacci_spiral(0.0),
        fibonacci_spiral(0.2),
        fibonacci_spiral(0.5),
        fibonacci_spiral(0.8),
        ring_initialization(10, rng),
        ring_initialization(14, rng),
        ring_initialization(20, rng),
        random_points(rng),
        random_points(rng),
        random_points(rng),
        clustered_initialization(rng, 4),
        clustered_initialization(rng, 6),
        clustered_initialization(rng, 8),
    ]

    # Phase 1: L-BFGS from multiple starts
    for init_pts in initializations:
        if time.time() > deadline - 50:
            break

        pts, energy = lbfgs_optimize(init_pts.flatten(), maxiter=400)
        if pts is not None and energy < best_energy:
            best_energy = energy
            best_points = pts.copy()

    # Phase 2: Gradient descent polish on best
    if best_points is not None:
        if time.time() < deadline - 40:
            pts, energy = gradient_descent(best_points, max_iters=5000, initial_step=0.005, time_budget=deadline - time.time() - 35)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 3: Swap-based annealing for global search
    if best_points is not None:
        if time.time() < deadline - 30:
            pts, energy = swap_annealing_fast(best_points, max_iters=8000, T_start=0.05, T_end=1e-8, time_budget=deadline - time.time() - 25)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 4: More L-BFGS restarts from perturbed best
    if best_points is not None:
        for restart in range(8):
            if time.time() > deadline - 20:
                break

            noise_scale = 0.015 + 0.008 * restart
            noise = rng.randn(N, 3) * noise_scale
            perturbed = normalize(best_points + noise)
            pts, energy = lbfgs_optimize(perturbed.flatten(), maxiter=300)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 5: Final gradient descent polish with tight tolerance
    if best_points is not None:
        if time.time() < deadline - 10:
            pts, energy = gradient_descent(best_points, max_iters=5000, initial_step=0.002, time_budget=deadline - time.time() - 8)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 6: One more L-BFGS polish with tight tolerance
    if best_points is not None:
        if time.time() < deadline - 4:
            pts, energy = lbfgs_optimize(best_points.flatten(), maxiter=300, tight=True)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 7: Final ultra-tight polish
    if best_points is not None:
        if time.time() < deadline - 1:
            pts, energy = gradient_descent(best_points, max_iters=3000, initial_step=0.001, time_budget=deadline - time.time() - 0.5)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Final normalization and validity check
    if best_points is None:
        best_points = fibonacci_spiral(0.0)

    best_points = normalize(best_points)

    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_spiral(0.0)
        best_points = normalize(best_points)

    # Ensure no two points are closer than 1e-9
    dist_sq = np.sum(best_points**2, axis=1)[:, None] + np.sum(best_points**2, axis=1)[None, :] - 2.0 * (best_points @ best_points.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)
    i_upper, j_upper = np.triu_indices(N, k=1)
    d_pairs = dist[i_upper, j_upper]
    if np.any(d_pairs < 1e-9):
        # Separate close points
        for _ in range(100):
            idx_i = np.argmin(d_pairs)
            i_idx = i_upper[idx_i]
            j_idx = j_upper[idx_i]
            # Push them apart
            direction = best_points[i_idx] - best_points[j_idx]
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                direction = rng.randn(3)
                norm_dir = np.linalg.norm(direction)
            direction = direction / norm_dir
            best_points[i_idx] = best_points[i_idx] + 1e-8 * direction
            best_points[j_idx] = best_points[j_idx] - 1e-8 * direction
            best_points = normalize(best_points)
            # Recompute
            dist_sq = np.sum(best_points**2, axis=1)[:, None] + np.sum(best_points**2, axis=1)[None, :] - 2.0 * (best_points @ best_points.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            dist = np.sqrt(dist_sq)
            d_pairs = dist[i_upper, j_upper]
            if np.all(d_pairs >= 1e-9):
                break

    return best_points

# EVOLVE-BLOCK-END
