import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Helper functions
    def _raw_score(centers):
        """
        Calculate the maximum radius r such that all circles fit.
        r = min( min_pair_dist/2, min_edge_dist )
        """
        if not np.all(np.isfinite(centers)):
            return 0.0
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms > 1.0):
            return 0.0
        
        # Pairwise distances
        # Using broadcasting for simplicity, N=50 is small enough (50x50 = 2500 ops)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_edge_dist = np.min(1.0 - norms)
        
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(0.0, r)

    def _objective(centers, beta=100.0):
        """
        Soft-min approximation of the negative radius.
        We want to maximize r. Minimizing -r is equivalent.
        r = min(c_i) where c_i are constraints (dist/2 or edge_dist).
        soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
        We minimize -soft_min(c) -> maximize soft_min(c)
        """
        centers = centers.reshape(-1, 2)
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        # Numerical stability for log-sum-exp
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta
        
        # We want to maximize soft_min_val, so minimize its negative
        return -soft_min_val

    def _gradient(centers, beta=100.0):
        """
        Gradient of the objective.
        """
        centers = centers.reshape(-1, 2)
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
        
        # Edge gradient: d/dx (1 - ||x||) = -x/||x||
        # Objective is -soft_min. 
        # d/dx (-soft_min) = - sum( w_i * d(c_i)/dx )
        # Wait, let's derive carefully.
        # f(x) = -soft_min(c(x)) = 1/beta * log(sum(exp(-beta*c(x))))
        # df/dx = 1/beta * 1/sum(exp) * sum( exp(-beta*c_i) * (-beta * dc_i/dx) )
        # df/dx = - sum( (exp(-beta*c_i)/sum(exp)) * dc_i/dx )
        # df/dx = - sum( w_i * dc_i/dx )
        
        # Edge part: c_edge = 1 - ||x||. d(c_edge)/dx = -x/||x||
        # Contribution to grad: - sum( w_edge * (-x/||x||) ) = sum( w_edge * x/||x|| )
        # Note: weights are for all constraints. First N are edges.
        edge_w = weights[:N]
        grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
        
        # Pair part: c_pair(i,j) = ||x_i - x_j|| / 2
        # d(c_pair)/dx_i = (x_i - x_j) / ||x_i - x_j||
        # d(c_pair)/dx_j = (x_j - x_i) / ||x_i - x_j|| = -d(c_pair)/dx_i
        # Contribution to grad[i]: - w_pair * (x_i - x_j)/||x_i - x_j||
        # Contribution to grad[j]: - w_pair * (x_j - x_i)/||x_i - x_j||
        
        n_pairs = len(pair_constraints)
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:N+n_pairs]
        
        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
        
        # grad[i] -= w * dir
        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
        # grad[j] += w * dir
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
        
        return grad.reshape(-1)

    def _refine_l_bfgs_b(centers, beta=300.0, maxiter=200):
        """
        Local refinement using L-BFGS-B.
        """
        try:
            result = minimize(
                _objective,
                centers.flatten(),
                args=(beta,),
                jac=lambda x, b: _gradient(x, b),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            return result.x.reshape(N, 2)
        except Exception:
            return centers

    def _simulated_annealing(centers, steps=2000, T_init=0.02, T_final=0.0005):
        """
        Simulated Annealing to escape local minima.
        Uses adaptive step size based on current score.
        """
        current_centers = centers.copy()
        current_score = _raw_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        
        # Pre-calculate indices for speed
        iu = np.triu_indices(N, 1)
        
        for i in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            # Temperature decay (exponential)
            T = T_init * (T_final / T_init) ** (i / steps)
            
            # Perturb a random subset of points
            num_moves = np.random.randint(1, 4)
            indices = np.random.choice(N, num_moves, replace=False)
            
            new_centers = current_centers.copy()
            # Adaptive step size: larger when score is poor, smaller when good
            step_size = max(T * 0.5, 0.005)
            moves = np.random.normal(0, step_size, size=(num_moves, 2))
            new_centers[indices] += moves
            
            # Keep inside unit circle
            norms = np.sqrt(np.sum(new_centers**2, axis=1))
            out_mask = norms > 0.999
            if np.any(out_mask):
                new_centers[out_mask] *= 0.999 / norms[out_mask, np.newaxis]
                
            new_score = _raw_score(new_centers)
            
            delta = new_score - current_score
            
            if delta > 0:
                current_centers = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = new_centers.copy()
            else:
                # Accept worse solution with probability exp(delta / T)
                # Clamp delta to avoid underflow
                clamped_delta = max(delta, -10.0 * T)
                if np.random.rand() < np.exp(clamped_delta / T):
                    current_centers = new_centers
                    current_score = new_score
        
        return best_centers, best_score

    # --- Initialization Strategies ---
    
    initializations = []
    
    # 1. Ring Configuration: 1, 6, 12, 18, 13
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

    # Known good radii approximations - tuned for better packing
    r1 = [0.0, 0.24, 0.46, 0.68, 0.88]
    c1 = [1, 6, 12, 18, 13]
    initializations.append(make_rings(r1, c1))
    
    # Perturbed rings with smaller perturbations
    for seed in range(8):
        np.random.seed(seed)
        perturbed_radii = [r1[0]] + [r + np.random.uniform(-0.015, 0.015) for r in r1[1:]]
        # Also perturb angles slightly
        offsets = [0.0] * len(r1)
        for s in range(1, len(r1)):
            offsets[s] = np.random.uniform(-0.1, 0.1)
        initializations.append(make_rings(perturbed_radii, c1, offsets))
        
    # New: Smart Ring Initialization
    # Generate rings with radii spaced to maximize minimum distance
    # Try different counts per ring
    def make_smart_rings(counts):
        # counts[0] must be 1 (center)
        if counts[0] != 1:
            return None
        num_rings = len(counts)
        # Radii should be roughly 1/num_rings apart, but optimized for packing
        # Simple heuristic: uniform spacing in radius
        radii = [0.0] + [1.0 * i / (num_rings - 1) for i in range(1, num_rings)]
        # Adjust outermost to be slightly less than 1 to allow for circle radius
        # Actually, the optimizer will fix the radii. We just need a good start.
        # Let's try to space them so that the "effective" ring width is uniform.
        # r_i = i / (num_rings - 1) * (1 - r_circle_approx)
        # r_circle_approx ~ 0.12
        # So max radius ~ 0.88
        max_r = 0.88
        radii = [0.0] + [max_r * i / (num_rings - 1) for i in range(1, num_rings)]
        
        offsets = [0.0] * num_rings
        # Stagger angles for adjacent rings to improve packing
        for i in range(1, num_rings):
            offsets[i] = np.pi / counts[i]
            
        return make_rings(radii, counts, offsets)

    # Try a few different ring count distributions
    smart_counts_list = [
        [1, 6, 12, 18, 13],
        [1, 5, 10, 15, 10, 9], # 6 rings
        [1, 7, 14, 14, 14], # 5 rings, more balanced
        [1, 6, 11, 16, 16], # 5 rings
        [1, 4, 8, 12, 16, 9, 4] # 7 rings
    ]
    
    for counts in smart_counts_list:
        if sum(counts) < N:
            # Pad with extra points on the outer ring or add a ring?
            # Just skip if sum < N, or distribute remainder
            pass
        # Ensure sum is exactly N for valid initialization
        # The make_rings function takes first N points. 
        # If sum(counts) > N, it truncates. If < N, it's invalid.
        # Let's adjust counts to sum to N
        total = sum(counts)
        if total < N:
            # Add difference to last ring
            counts[-1] += N - total
        elif total > N:
            # Remove difference from last ring
            counts[-1] -= total - N
            if counts[-1] <= 0:
                continue
                
        cfg = make_smart_rings(counts)
        if cfg is not None and len(cfg) == N:
            initializations.append(cfg)

    # 2. Hexagonal Lattice - try multiple spacings
    def make_hex_lattice(spacing):
        points = []
        for j in range(-12, 13):
            for i in range(-12, 13):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x*x + y*y) <= 1.0 - spacing/2 - 1e-9:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            return None
        # Pick the N closest to center
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]
    
    for spacing in [0.24, 0.25, 0.26]:
        hex_cfg = make_hex_lattice(spacing)
        if hex_cfg is not None:
            initializations.append(hex_cfg)
    
    # 3. Random Points (Jittered) - more diverse
    np.random.seed(42)
    for _ in range(6):
        cfg = np.random.uniform(-0.92, 0.92, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.92
        if np.any(mask):
            cfg[mask] *= 0.92 / norms[mask, np.newaxis]
        initializations.append(cfg)

    best_score = 0.0
    best_centers = initializations[0]
    top_solutions = []  # Keep track of top solutions for SA
    
    # Phase 1: Refine all initializations with L-BFGS-B
    for idx, x0 in enumerate(initializations):
        if time.time() > deadline - 70.0: # Leave time for SA
            break
        
        refined = _refine_l_bfgs_b(x0, beta=200.0, maxiter=150)
        score = _raw_score(refined)
        
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
        
        # Track top 3 solutions
        top_solutions.append((score, refined.copy()))
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        top_solutions = top_solutions[:3]

    # Phase 2: Simulated Annealing from top solutions
    for sa_round in range(min(3, len(top_solutions))):
        if time.time() > deadline - 25.0:
            break
        
        start_sol = top_solutions[sa_round][1]
        sa_centers, sa_score = _simulated_annealing(
            start_sol, 
            steps=2500, 
            T_init=0.015, 
            T_final=0.0003
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            # Update top solutions
            top_solutions.append((sa_score, sa_centers.copy()))
            top_solutions.sort(key=lambda x: x[0], reverse=True)
            top_solutions = top_solutions[:3]
        
        # Refine the SA result
        if time.time() < deadline - 10.0:
            refined_sa = _refine_l_bfgs_b(best_centers, beta=400.0, maxiter=250)
            score = _raw_score(refined_sa)
            if score > best_score:
                best_score = score
                best_centers = refined_sa.copy()
                top_solutions.append((score, refined_sa.copy()))
                top_solutions.sort(key=lambda x: x[0], reverse=True)
                top_solutions = top_solutions[:3]

    # Final Polish - two stages for better convergence
    if time.time() < deadline - 5.0:
        # Stage 1: Medium precision
        final_refined = _refine_l_bfgs_b(best_centers, beta=500.0, maxiter=300)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined
    
    if time.time() < deadline - 2.0:
        # Stage 2: High precision
        final_refined = _refine_l_bfgs_b(best_centers, beta=800.0, maxiter=400)
        final_score = _raw_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_refined

    # Ensure valid return
    if not np.all(np.isfinite(best_centers)):
        return initializations[0]
        
    return best_centers

if __name__ == "__main__":
    # Simple test to ensure it runs
    centers = construct()
    print(f"Resulting shape: {centers.shape}")
    # print(_raw_score(centers)) # Cannot access internal directly if not exposed, but it's fine.
