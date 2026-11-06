import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    def _raw_score(centers):
        """
        Calculate the maximum radius r such that all circles fit.
        r = min( min_pair_dist/2, min_edge_dist )
        """
        if not np.all(np.isfinite(centers)):
            return 0.0
        centers = centers.reshape(N, 2)
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms > 1.0 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_edge_dist = np.min(1.0 - norms)
        
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(0.0, r)

    def _soft_min_objective(centers, beta=100.0):
        """
        Objective function for L-BFGS-B: minimize -soft_min(constraints)
        """
        centers = centers.reshape(N, 2)
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta
        
        return -soft_min_val

    def _soft_min_gradient(centers, beta=100.0):
        """
        Gradient of the soft-min objective.
        """
        centers = centers.reshape(N, 2)
        grad = np.zeros_like(centers)
        
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Edge gradient contribution
        edge_w = weights[:N]
        grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
        
        # Pair gradient contribution
        n_pairs = len(pair_constraints)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+n_pairs]
        
        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
        
        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
        
        return grad.reshape(-1)

    def _refine(centers, beta=300.0, maxiter=200):
        """
        Local refinement using L-BFGS-B with soft-min objective.
        """
        try:
            result = minimize(
                _soft_min_objective,
                centers.flatten(),
                args=(beta,),
                jac=lambda x, b: _soft_min_gradient(x, b),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            refined = result.x.reshape(N, 2)
            # Ensure points stay inside
            norms = np.sqrt(np.sum(refined**2, axis=1))
            mask = norms > 1.0
            if np.any(mask):
                refined[mask] *= (1.0 - 1e-6) / norms[mask, np.newaxis]
            return refined
        except Exception:
            return centers

    def _simulated_annealing(centers, steps=1500, T_init=0.015, T_final=0.0003):
        """
        Simulated Annealing for global exploration.
        """
        current_centers = centers.copy()
        current_score = _raw_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        
        iu = np.triu_indices(N, 1)
        
        for i in range(steps):
            if time.time() > deadline - 10.0:
                break
                
            T = T_init * (T_final / T_init) ** (i / steps)
            
            # Move 1-3 random points
            num_moves = np.random.randint(1, 4)
            indices = np.random.choice(N, num_moves, replace=False)
            
            new_centers = current_centers.copy()
            step_size = max(T * 0.5, 0.003)
            moves = np.random.normal(0, step_size, size=(num_moves, 2))
            new_centers[indices] += moves
            
            # Keep inside unit circle
            norms = np.sqrt(np.sum(new_centers**2, axis=1))
            out_mask = norms > 0.9995
            if np.any(out_mask):
                new_centers[out_mask] *= 0.9995 / norms[out_mask, np.newaxis]
                
            new_score = _raw_score(new_centers)
            delta = new_score - current_score
            
            if delta > 0:
                current_centers = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
            else:
                clamped_delta = max(delta, -10.0 * T)
                if np.random.rand() < np.exp(clamped_delta / T):
                    current_centers = new_centers
                    current_score = new_score
        
        return best_centers, best_score

    # --- New Ring-Based Optimization ---
    
    def _generate_ring_centers(params, counts):
        """
        Generate centers from ring parameters (radii, offsets).
        params: array of length sum(counts) + num_rings
        First num_rings values are radii? No, better to separate.
        Let's say params structure: [r_0, r_1, ..., r_k, theta_0, theta_1, ..., theta_k]
        where k is number of rings.
        counts: list of integers summing to N.
        """
        num_rings = len(counts)
        radii = params[:num_rings]
        offsets = params[num_rings:]
        
        centers = []
        idx = 0
        for i, count in enumerate(counts):
            r = radii[i]
            theta_off = offsets[i]
            for k in range(count):
                theta = 2 * np.pi * k / count + theta_off
                centers.append([r * np.cos(theta), r * np.sin(theta)])
            idx += count
            
        return np.array(centers[:N])

    def _ring_objective(params, counts, beta=100.0):
        centers = _generate_ring_centers(params, counts)
        return _soft_min_objective(centers, beta)

    def _ring_gradient(params, counts, beta=100.0):
        centers = _generate_ring_centers(params, counts)
        grad_centers = _soft_min_gradient(centers, beta).reshape(N, 2)
        
        num_rings = len(counts)
        grad_params = np.zeros_like(params)
        
        idx = 0
        for i, count in enumerate(counts):
            r = params[i]
            theta_off = params[num_rings + i]
            
            # Gradient w.r.t radius
            # d center / d r = [cos(theta), sin(theta)]
            # grad_params[i] += sum( grad_centers[j] . d center[j] / d r )
            
            # Gradient w.r.t offset
            # d center / d theta = [-r sin(theta), r cos(theta)]
            # grad_params[num_rings+i] += sum( grad_centers[j] . d center[j] / d theta )
            
            grad_r = 0.0
            grad_theta = 0.0
            
            for k in range(count):
                theta = 2 * np.pi * k / count + theta_off
                c_j = centers[idx + k]
                g_j = grad_centers[idx + k]
                
                # Vector from origin to center
                # d center / d r = c_j / r (if r > 0)
                if r > 1e-9:
                    dc_dr = c_j / r
                    grad_r += np.dot(g_j, dc_dr)
                
                # d center / d theta = [-r sin(theta), r cos(theta)]
                dc_dtheta = np.array([-r * np.sin(theta), r * np.cos(theta)])
                grad_theta += np.dot(g_j, dc_dtheta)
                
            grad_params[i] = grad_r
            grad_params[num_rings + i] = grad_theta
            idx += count
            
        return grad_params

    def _optimize_ring_config(counts, init_params=None, beta=200.0, maxiter=300):
        """
        Optimize a specific ring configuration.
        """
        num_rings = len(counts)
        if init_params is None:
            # Initialize radii evenly spaced, offsets random
            total_rings = num_rings
            radii = np.linspace(0.1, 0.9, total_rings)
            offsets = np.random.uniform(0, 2*np.pi, total_rings)
            params = np.concatenate([radii, offsets])
        else:
            params = init_params.copy()
            
        try:
            result = minimize(
                _ring_objective,
                params,
                args=(counts, beta),
                jac=lambda x, c, b: _ring_gradient(x, c, b),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            best_params = result.x
            centers = _generate_ring_centers(best_params, counts)
            # Clip radii to be non-negative and within bounds
            # Note: The objective handles the "inside" constraint via soft-min, 
            # but we should ensure validity for the final score
            return centers
        except Exception:
            return _generate_ring_centers(params, counts)

    # --- Generate diverse initializations ---
    
    initializations = []
    
    # 1. Ring-based initializations with various configurations
    def make_rings(radii, counts, offsets=None):
        centers = []
        centers.append([0.0, 0.0])
        start_idx = 1
        for r, c in zip(radii, counts):
            base_offset = offsets[start_idx-1] if offsets else 0.0
            for k in range(c):
                theta = 2 * np.pi * k / c + base_offset
                centers.append([r * np.cos(theta), r * np.sin(theta)])
            start_idx += 1
        return np.array(centers[:N])

    # Standard ring configuration
    r1 = [0.0, 0.24, 0.46, 0.68, 0.88]
    c1 = [1, 6, 12, 18, 13]
    initializations.append(make_rings(r1, c1))
    
    # Perturbed rings
    for seed in range(10):
        np.random.seed(seed)
        perturbed_radii = [r1[0]] + [r + np.random.uniform(-0.02, 0.02) for r in r1[1:]]
        offsets = [0.0] * len(r1)
        for s in range(1, len(r1)):
            offsets[s] = np.random.uniform(-0.15, 0.15)
        initializations.append(make_rings(perturbed_radii, c1, offsets))
    
    # Different ring count distributions
    ring_configs = [
        ([0.0, 0.25, 0.5, 0.75, 0.9], [1, 6, 12, 18, 13]),
        ([0.0, 0.22, 0.44, 0.66, 0.85], [1, 7, 14, 14, 14]),
        ([0.0, 0.2, 0.4, 0.6, 0.8, 0.95], [1, 5, 10, 15, 10, 9]),
        ([0.0, 0.18, 0.38, 0.58, 0.78, 0.92], [1, 6, 11, 16, 10, 6]),
        ([0.0, 0.2, 0.45, 0.7, 0.9], [1, 8, 16, 14, 11]),
    ]
    for radii, counts in ring_configs:
        total = sum(counts)
        if total < N:
            counts[-1] += N - total
        elif total > N:
            counts[-1] -= total - N
            if counts[-1] <= 0:
                continue
        cfg = make_rings(radii, counts)
        if cfg is not None and len(cfg) == N:
            initializations.append(cfg)
    
    # 2. Hexagonal lattice initializations
    def make_hex_lattice(spacing):
        points = []
        for j in range(-15, 16):
            for i in range(-15, 16):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x*x + y*y) <= 1.0 - spacing/2 - 1e-9:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            return None
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]
    
    for spacing in [0.23, 0.24, 0.25, 0.26, 0.27]:
        hex_cfg = make_hex_lattice(spacing)
        if hex_cfg is not None:
            initializations.append(hex_cfg)
    
    # 3. Random initializations
    np.random.seed(42)
    for _ in range(8):
        cfg = np.random.uniform(-0.95, 0.95, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.95
        if np.any(mask):
            cfg[mask] *= 0.95 / norms[mask, np.newaxis]
        initializations.append(cfg)

    # --- Phase 1: Refine all initializations ---
    best_score = 0.0
    best_centers = initializations[0]
    top_solutions = []
    
    for idx, x0 in enumerate(initializations):
        if time.time() > deadline - 80.0:
            break
        
        refined = _refine(x0, beta=200.0, maxiter=150)
        score = _raw_score(refined)
        
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
        
        top_solutions.append((score, refined.copy()))
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        top_solutions = top_solutions[:5]

    # --- Phase 2: Ring-Structure Optimization ---
    # Try various ring count distributions
    ring_count_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 17, 14],
        [1, 6, 12, 16, 15],
        [1, 7, 12, 15, 15],
        [1, 6, 13, 17, 13],
        [1, 5, 12, 18, 14],
        [1, 6, 11, 18, 14],
        [1, 8, 14, 14, 13],
        [1, 6, 12, 18, 12, 1], # 6 rings
        [1, 6, 12, 16, 15],
        [1, 6, 10, 16, 17],
    ]
    
    for counts in ring_count_configs:
        if time.time() > deadline - 40.0:
            break
        if sum(counts) != N:
            continue
            
        # Initialize parameters based on current best or random
        # Use random init for diversity
        num_rings = len(counts)
        radii = np.linspace(0.1, 0.9, num_rings)
        offsets = np.random.uniform(0, 2*np.pi, num_rings)
        init_params = np.concatenate([radii, offsets])
        
        ring_centers = _optimize_ring_config(counts, init_params, beta=300.0, maxiter=400)
        
        # Refine with full 100D optimizer to break symmetry
        refined_ring = _refine(ring_centers, beta=500.0, maxiter=300)
        score = _raw_score(refined_ring)
        
        if score > best_score:
            best_score = score
            best_centers = refined_ring.copy()
            
        top_solutions.append((score, refined_ring.copy()))
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        top_solutions = top_solutions[:5]

    # --- Phase 3: Simulated Annealing from top solutions ---
    for sa_round in range(min(4, len(top_solutions))):
        if time.time() > deadline - 20.0:
            break
        
        start_sol = top_solutions[sa_round][1]
        sa_centers, sa_score = _simulated_annealing(
            start_sol, 
            steps=1000, 
            T_init=0.012, 
            T_final=0.0002
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            top_solutions.append((sa_score, sa_centers.copy()))
            top_solutions.sort(key=lambda x: x[0], reverse=True)
            top_solutions = top_solutions[:5]
        
        # Refine SA result
        if time.time() < deadline - 10.0:
            refined_sa = _refine(best_centers, beta=600.0, maxiter=250)
            score = _raw_score(refined_sa)
            if score > best_score:
                best_score = score
                best_centers = refined_sa.copy()
                top_solutions.append((score, refined_sa.copy()))
                top_solutions.sort(key=lambda x: x[0], reverse=True)
                top_solutions = top_solutions[:5]

    # --- Phase 4: Final high-precision polish ---
    if time.time() < deadline - 5.0:
        final_refined = _refine(best_centers, beta=1000.0, maxiter=400)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined

    # Ensure valid return
    if not np.all(np.isfinite(best_centers)):
        return initializations[0]
    
    # Final validity check
    norms = np.sqrt(np.sum(best_centers**2, axis=1))
    if np.any(norms > 1.0):
        # Scale down if any point is outside
        max_norm = np.max(norms)
        best_centers *= (1.0 - 1e-6) / max_norm

    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(f"Resulting shape: {centers.shape}")
