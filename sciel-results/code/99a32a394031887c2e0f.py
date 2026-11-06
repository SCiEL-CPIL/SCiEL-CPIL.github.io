"""
Fifty equal circles in a circle - direct structural approach.
Uses a ring-based configuration with known optimal ring sizes,
then refines with gradient-based optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit, 20s buffer)
    
    best_score = 0.0
    best_centers = None
    
    def _score_raw(centers):
        """Compute the actual score (minimum radius) for a set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms >= 1.0):
            return 0.0
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_edge_dist = np.min(1.0 - norms)
        
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(r, 0.0)
    
    def _objective(centers, beta=300.0):
        """Smooth approximation of negative score."""
        centers = centers.reshape(-1, 2)
        
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta
        
        return -soft_min_val
    
    def _gradient(centers, beta=300.0):
        """Compute gradient of the objective - vectorized."""
        centers = centers.reshape(-1, 2)
        grad = np.zeros_like(centers)
        
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-15
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-15
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Edge gradient (vectorized)
        edge_grad = -centers / norms[:, np.newaxis]
        grad += -weights[:N, np.newaxis] * beta * edge_grad
        
        # Pair gradients (vectorized using scatter)
        n_pairs = len(pair_constraints)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+n_pairs]
        
        # Direction vectors for each pair
        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
        
        # Accumulate gradients: grad[i] += -w * beta * dir, grad[j] += w * beta * dir
        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * beta * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * beta * dir_vecs)
        
        return grad.reshape(-1)
    
    # Direct structural approach: use ring-based configuration
    # Known good configuration: 1 center, 6, 12, 18, 13 in rings
    def _make_ring_config(ring_radii, ring_counts, angles_offset=0.0):
        centers = []
        
        # Center
        if ring_counts[0] > 0:
            centers.append([0.0, 0.0])
        
        for ring_idx in range(1, len(ring_radii)):
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            for k in range(count):
                theta = 2 * np.pi * k / count + angles_offset * ring_idx
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        return centers
    
    # Try multiple ring configurations
    ring_configs = []
    
    # Configuration 1: Standard rings
    radii1 = [0.0, 0.25, 0.5, 0.75, 0.95]
    counts1 = [1, 6, 12, 18, 13]
    ring_configs.append(_make_ring_config(radii1, counts1))
    
    # Configuration 2: Slightly different radii
    radii2 = [0.0, 0.22, 0.48, 0.72, 0.94]
    counts2 = [1, 6, 12, 18, 13]
    ring_configs.append(_make_ring_config(radii2, counts2))
    
    # Configuration 3: Adjusted
    radii3 = [0.0, 0.28, 0.52, 0.74, 0.93]
    counts3 = [1, 6, 12, 18, 13]
    ring_configs.append(_make_ring_config(radii3, counts3))
    
    # Configuration 4: Different angle offset
    ring_configs.append(_make_ring_config(radii1, counts1, angles_offset=0.2))
    
    # Configuration 5: Shifted angles
    ring_configs.append(_make_ring_config(radii1, counts1, angles_offset=0.5))
    
    # Hexagonal lattice-based
    def _hexagonal_config():
        points = []
        spacing = 0.26  # Approximate spacing for r~0.13
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y)
                if r <= 1.0 - 0.13 - 1e-9:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            return None
        
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]
    
    hex_cfg = _hexagonal_config()
    if hex_cfg is not None:
        ring_configs.append(hex_cfg)
    
    # Random configurations
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(42 + seed * 100)
        cfg = np.random.uniform(-0.85, 0.85, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        ring_configs.append(cfg)
    
    # Optimize each configuration with increased iterations (vectorized gradient is ~10x faster)
    for idx, x0 in enumerate(ring_configs):
        if time.time() > deadline:
            break
        
        # Multiple beta values for different stages - more iterations now feasible
        for beta in [100.0, 300.0, 500.0]:
            if time.time() > deadline - 10.0:
                break
            
            try:
                result = minimize(
                    _objective,
                    x0.flatten(),
                    args=(beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers
            except Exception:
                continue
    
    # Final refinement with higher beta - more passes possible with faster gradient
    if best_centers is not None and time.time() < deadline - 15.0:
        np.random.seed(123)
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb best solution with slightly larger perturbation for better exploration
            perturbation_scale = 0.005
            perturbed = best_centers + np.random.normal(0, perturbation_scale, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                perturbed[mask] *= 0.98 / norms[mask, np.newaxis]
            
            try:
                result = minimize(
                    _objective,
                    perturbed.flatten(),
                    args=(500.0,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
                )
                
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers
            except Exception:
                continue
    
    # Final high-precision polish pass
    if best_centers is not None and time.time() < deadline - 10.0:
        # Try a few more polishes with very tight tolerances
        for polish_beta in [800.0, 1200.0, 1500.0]:
            if time.time() > deadline - 5.0:
                break
            try:
                result = minimize(
                    _objective,
                    best_centers.flatten(),
                    args=(polish_beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                )
                centers = result.x.reshape(N, 2)
                score = _score_raw(centers)
                if score > best_score:
                    best_score = score
                    best_centers = centers
            except Exception:
                continue
    
    # Final validation and return
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers
    
    # Fallback
    return _make_ring_config([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13])
