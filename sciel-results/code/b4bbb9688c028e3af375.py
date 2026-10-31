import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    def get_pairs(X):
        # X is (N, D)
        # Returns array of pairwise squared distances (n_pairs,)
        # and gradients for each point
        xi = X[i_idx]
        xj = X[j_idx]
        diff = xi - xj
        u = np.sum(diff**2, axis=1)
        return u, diff
        
    def objective_and_grad(X_flat, beta, mu):
        X = X_flat.reshape(N, D)
        u, diff = get_pairs(X)
        
        # Log-sum-exp for max
        # U_beta = 1/beta * log(sum(exp(beta * u)))
        # To avoid overflow, subtract max
        u_max = np.max(u)
        exp_terms = np.exp(beta * (u - u_max))
        sum_exp = np.sum(exp_terms)
        U_beta = u_max + (1.0 / beta) * np.log(sum_exp)
        
        # Weights for gradient of U_beta
        weights_u = exp_terms / sum_exp
        
        # Gradient of U_beta w.r.t X
        # dU_beta/dX_i = sum_{j} weights_{ij} * 2 * (x_i - x_j)
        grad_U = np.zeros_like(X)
        # Accumulate gradients for i_idx
        # diff = xi - xj
        # grad for xi is 2 * diff * weight
        # grad for xj is -2 * diff * weight
        np.add.at(grad_U, i_idx, 2.0 * weights_u[:, None] * diff)
        np.add.at(grad_U, j_idx, -2.0 * weights_u[:, None] * diff)
        
        # Penalty P(X) = sum max(0, 1 - u)^2
        # If u < 1, penalty is (1-u)^2
        # Grad is 2(1-u)(-1) * grad_u = -2(1-u) * grad_u
        # grad_u for xi is 2(xi-xj), for xj is -2(xi-xj)
        # So grad P for xi is -2(1-u) * 2(xi-xj) = -4(1-u)(xi-xj)
        # grad P for xj is -2(1-u) * (-2(xi-xj)) = 4(1-u)(xi-xj)
        
        mask = u < 1.0
        P_val = 0.0
        if np.any(mask):
            u_masked = u[mask]
            diff_masked = diff[mask]
            # Coefficient is -4 * (1 - u)
            coeff = -4.0 * (1.0 - u_masked)
            
            # Add to grad
            # For i_idx
            np.add.at(grad_U, i_idx[mask], coeff[:, None] * diff_masked)
            # For j_idx
            np.add.at(grad_U, j_idx[mask], -coeff[:, None] * diff_masked)
            
            P_val = np.sum((1.0 - u_masked)**2)
            
        val = U_beta + mu * P_val
        grad = grad_U
        
        return val, grad

    # --- Phase 1: Lattice-based Initialization & Local Search ---
    # Generate a triangular lattice
    R = 6
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            lattice_pts.append((x, y))
    
    lattice_pts = np.array(lattice_pts)
    num_lattice_pts = len(lattice_pts)
    
    # Initial guess: 16 closest points to origin
    dists_from_origin = np.sum(lattice_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    
    # Candidate pool: top 150 closest points
    pool_size = 150
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = lattice_pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    current_indices = initial_subset.copy()
    current_score = score_subset(current_indices)
    
    in_subset = set(current_indices)
    outside_candidates = [i for i in candidate_indices if i not in in_subset]
    
    improved = True
    while improved and time.time() < deadline - 20: # Leave 20s for optimization
        improved = False
        # Try swapping each point in subset with each outside candidate
        for i_idx, i in enumerate(current_indices):
            if time.time() > deadline - 20:
                break
            for j in outside_candidates:
                if time.time() > deadline - 20:
                    break
                # Swap i and j
                new_indices = current_indices.copy()
                new_indices[i_idx] = j
                new_score = score_subset(new_indices)
                if new_score < current_score:
                    current_score = new_score
                    current_indices = new_indices
                    in_subset = set(current_indices)
                    outside_candidates = [k for k in candidate_indices if k not in in_subset]
                    improved = True
                    break # Restart j loop
            if improved:
                break
                
    best_lattice_pts = lattice_pts[current_indices]
    
    # --- Phase 2: Continuous Optimization from Lattice Start ---
    best_score = current_score
    best_points = best_lattice_pts
    
    # Optimization parameters
    beta = 5.0
    mu = 10.0
    max_iter = 300
    
    # We will try to optimize from the lattice configuration
    # And also from a few random perturbations of it
    starts = [best_lattice_pts]
    # Add a few random perturbations
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        perturbed = best_lattice_pts + np.random.normal(0, 0.1, (N, D))
        starts.append(perturbed)
        
    for X0 in starts:
        if time.time() > deadline - 5:
            break
            
        res = minimize(
            objective_and_grad,
            X0.flatten(),
            args=(beta, mu),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'maxfun': 500}
        )
        
        if res.success:
            X_opt = res.x.reshape(N, D)
            
            # Calculate actual score
            u, _ = get_pairs(X_opt)
            d_min = np.sqrt(np.min(u))
            d_max = np.sqrt(np.max(u))
            if d_min > 1e-6:
                score = (d_max / d_min) ** 2
            else:
                score = np.inf
                
            if score < best_score:
                best_score = score
                best_points = X_opt
        else:
            # If optimization fails, still check the last point?
            # res.x might be better than start?
            X_opt = res.x.reshape(N, D)
            u, _ = get_pairs(X_opt)
            d_min = np.sqrt(np.min(u))
            d_max = np.sqrt(np.max(u))
            if d_min > 1e-6:
                score = (d_max / d_min) ** 2
                if score < best_score:
                    best_score = score
                    best_points = X_opt

    # --- Phase 3: Refinement with higher beta/mu ---
    if best_points is not None and time.time() < deadline - 5:
        X_refine = best_points
        beta = 50.0
        mu = 100.0
        res = minimize(
            objective_and_grad,
            X_refine.flatten(),
            args=(beta, mu),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'maxfun': 200}
        )
        
        X_opt = res.x.reshape(N, D)
        u, _ = get_pairs(X_opt)
        d_min = np.sqrt(np.min(u))
        d_max = np.sqrt(np.max(u))
        if d_min > 1e-6:
            score = (d_max / d_min) ** 2
            if score < best_score:
                best_score = score
                best_points = X_opt

    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points

BET: {mechanism: "Combining the combinatorial lattice search (which finds a structurally sound starting point with high symmetry and good packing) with the smooth log-sum-exp gradient optimization (which allows fine-tuning of coordinates to reduce the max/min ratio) avoids the local minima traps of pure random initialization and the sub-optimality of pure lattice swapping.", q: 0.6, delta: [0.0, 0.1]}
