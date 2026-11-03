"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_soft_radii(centers, beta=100.0):
        """
        Compute soft-min radii using log-sum-exp approximation.
        """
        centers = np.asarray(centers)
        n = len(centers)
        
        # Boundary distances
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        boundary_dists = np.min(np.column_stack([d_left, d_right, d_bottom, d_top]), axis=1)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        radii = np.zeros(n)
        for i in range(n):
            c_bound = boundary_dists[i]
            c_neighbors = dist[i, :] / 2.0
            all_constraints = np.concatenate(([c_bound], c_neighbors))
            
            # Soft min: -1/beta * log(sum(exp(-beta * x)))
            # Shift for numerical stability
            m = np.min(all_constraints)
            if m < 0:
                m = 0.0
            log_sum = np.log(np.sum(np.exp(-beta * (all_constraints - m))))
            radii[i] = m - (1.0/beta) * log_sum
            if radii[i] < 0:
                radii[i] = 0.0
        return radii

    def objective_smooth(centers_flat, beta=100.0):
        centers = centers_flat.reshape(-1, 2)
        radii = compute_soft_radii(centers, beta)
        return -np.sum(radii)

    def compute_hard_radii(centers):
        """Compute exact maximum radii for given centers."""
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)
        
        for i in range(n):
            radii[i] = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = dist / 2.0
                if max_r < radii[i]:
                    radii[i] = max_r
                if max_r < radii[j]:
                    radii[j] = max_r
        return radii

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    best_result = None
    best_score = np.inf
    best_centers = None
    best_smooth_val = np.inf

    # Initializations
    inits = [initial_grid_centers()]
    for seed in [1, 7, 13, 23, 37, 53, 71, 89]:
        inits.append(random_centers(seed))

    # Annealing schedule: start smooth (low beta) to escape local minima, end sharp (high beta) for precision
    beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15:
            break
        
        # Use current best as primary start, plus a few randoms
        starts = []
        if best_centers is not None:
            starts.append(best_centers)
        else:
            starts.extend(inits)
            
        # Add random restarts if time allows
        if time.time() < deadline - 20:
            for s in [42, 99]:
                starts.append(random_centers(s))
        
        for start_c in starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    objective_smooth,
                    start_c.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success:
                    val = res.fun
                    if val < best_smooth_val:
                        best_smooth_val = val
                        best_centers = res.x.reshape(-1, 2)
                        
                        # Evaluate hard radii for this configuration
                        hard_radii = compute_hard_radii(best_centers)
                        final_x = np.column_stack([best_centers, hard_radii])
                        
                        if is_valid(final_x, tol=1e-6):
                            score = np.sum(hard_radii)
                            if -score < best_score:
                                best_score = -score
                                best_result = final_x
            except Exception:
                continue

    # Final Polish: High beta optimization from best found
    if best_centers is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            perturbed = best_centers + np.random.normal(0, 0.005, size=best_centers.shape)
            perturbed = np.clip(perturbed, 0.0, 1.0)
            try:
                res = minimize(
                    objective_smooth,
                    perturbed.flatten(),
                    args=(1000.0,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success:
                    val = res.fun
                    if val < best_smooth_val:
                        best_smooth_val = val
                        best_centers = res.x.reshape(-1, 2)
                        hard_radii = compute_hard_radii(best_centers)
                        final_x = np.column_stack([best_centers, hard_radii])
                        if is_valid(final_x, tol=1e-6):
                            score = np.sum(hard_radii)
                            if -score < best_score:
                                best_score = -score
                                best_result = final_x
            except Exception:
                continue

    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_hard_radii(centers)
        best_result = np.column_stack([centers, radii])
        
    return best_result

# EVOLVE-BLOCK-END
