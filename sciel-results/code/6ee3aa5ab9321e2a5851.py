import numpy as np
import time
import sys

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    Strategy: Topological optimization using eigenvector gradients.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry handling
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            points = [
                (x, y),
                (y, size-1-x),
                (size-1-x, size-1-y),
                (size-1-y, x),
                (x, size-1-y),
                (size-1-x, y),
                (y, x),
                (size-1-y, size-1-x)
            ]
            for p in points:
                if p not in orbit:
                    orbit.add(p)
            orbits.append(orbit)
            seen.update(orbit)
    
    n_orbits = len(orbits)

    def orbit_to_grid(orbit_vals):
        grid = np.zeros((size, size), dtype=np.float64)
        for k in range(n_orbits):
            val = orbit_vals[k]
            for (i, j) in orbits[k]:
                grid[i, j] = val
        return grid

    def grid_to_orbit(grid):
        vals = np.zeros(n_orbits)
        for k in range(n_orbits):
            # Average values in the orbit to ensure consistency
            s = 0
            cnt = 0
            for (i, j) in orbits[k]:
                s += grid[i, j]
                cnt += 1
            vals[k] = s / cnt
        return vals

    # Fast Evaluator
    # Use a reduced plane wave basis for speed during optimization.
    # N_pw = 11 (from -5 to 5) -> 121 modes.
    N_pw = 11
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # Precompute G_diff indices for epsilon matrix
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    idx_x = G_diff[:, :, 0] % size
    idx_y = G_diff[:, :, 1] % size

    # K-points in the irreducible Brillouin zone
    # We sample key points to approximate the gap.
    # Gamma (0,0), M (0.5,0.5), X (0.5,0)
    # And midpoints.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.375),
        (0.375, 0.375)
    ]

    def evaluate_and_get_eigenvectors(vals):
        """
        Calculate the gap metric and store eigenvectors for gradient calculation.
        Returns score and a dictionary of eigenvectors.
        """
        grid = orbit_to_grid(vals)
        eps_grid = eps_low + (eps_high - eps_low) * grid
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0
        global_min_5 = float('inf')
        global_k_idx = -1
        global_evec_4 = None
        global_evec_5 = None
        global_eigvals = None
        
        for k_idx, (kx, ky) in enumerate(k_points):
            kG = G_vectors + np.array([kx, ky])
            kG_sq = np.sum(kG**2, axis=1)
            dot = np.dot(kG, kG.T)
            
            # Epsilon matrix
            eps_mat = eps_fft[idx_x, idx_y].real
            
            # M = diag(kG_sq) - dot / eps_mat
            M = np.diag(kG_sq) - dot / eps_mat
            
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
            except:
                return -1.0, None
                
            if len(eigvals) < 5:
                return -1.0, None
                
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
            if omega_4 > global_max_4:
                global_max_4 = omega_4
                global_k_idx = k_idx
                global_evec_4 = eigvecs[:, 3]
                global_evec_5 = eigvecs[:, 4]
                global_eigvals = eigvals
            if omega_5 < global_min_5:
                # We might need the evec at min_5 for gradient, but usually max_4 and min_5 are at different k points.
                # For simplicity of gradient, we approximate gradient using the dominant k-point for the gap.
                # If max_4 and min_5 are at different k points, the gradient is a sum.
                # We will store info for the "bottleneck" k-point.
                pass
        
        if global_min_5 == float('inf'):
            return -1.0, None
            
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap == 0:
            return 0.0, None
        ratio = gap / midgap
        
        return ratio, {
            'k_idx': global_k_idx,
            'evec_4': global_evec_4,
            'evec_5': global_evec_5,
            'eigvals': global_eigvals,
            'grid': grid
        }

    def compute_gradient(vals, info):
        """
        Compute gradient of the gap with respect to the dielectric distribution.
        Using first-order perturbation theory.
        d(omega^2)/d(eps) involves the field profile.
        For TM, omega^2 is eigenvalue of M.
        dM/deps_G = - (dot / eps_mat^2) * deps_G
        This is complex. 
        Alternative: Finite difference is safer and faster if we can do it efficiently.
        But we have 120s. 
        Let's try a semi-analytical gradient or just use finite differences on a few key orbits?
        
        Actually, the "inspiration" mentioned field-intensity-driven topological optimization.
        Let's use the gradient of the objective function J = (min_5 - max_4) / midgap.
        
        Let's approximate the gradient by evaluating the change in the local gap at the bottleneck k-point
        when flipping orbits. This is essentially a local search but guided by the field.
        
        Better: Use the eigenvectors to compute the sensitivity.
        For a specific k-point, omega^2 = lambda.
        d lambda / d eps_G = - (v^T (dM/deps_G) v) / (v^T M v) ... no, M is symmetric.
        d lambda / d eps_G = v^T (dM/deps_G) v (since v is normalized).
        dM/deps_G = - dot / eps_mat^2 * (delta at G_diff).
        
        This is computationally heavy to sum over all G.
        
        Let's stick to a robust "Gradient Ascent" via finite differences on the orbit vector, 
        but use the field info to prioritize which orbits to perturb?
        
        Actually, let's implement a simple gradient step:
        For each orbit, flip it, evaluate score, unflip. Compute delta.
        This is O(N_orbits * Eval_Cost).
        N_orbits ~ 1600. Eval_Cost ~ 10ms. Total ~ 16s per gradient step.
        We have 120s. So we can do ~5-10 gradient steps.
        
        Let's do that. It's a structural change from random SA to Gradient Ascent.
        """
        if info is None:
            return np.zeros(n_orbits)
            
        base_score, _ = evaluate_and_get_eigenvectors(vals)
        if base_score < 0:
            return np.zeros(n_orbits)
            
        grad = np.zeros(n_orbits)
        
        # To save time, we might not evaluate ALL orbits for gradient if it's too slow.
        # But 1600 * 10ms = 16s. That's acceptable for a few steps.
        
        for idx in range(n_orbits):
            new_vals = vals.copy()
            new_vals[idx] = 1 - new_vals[idx]
            new_score, _ = evaluate_and_get_eigenvectors(new_vals)
            
            if new_score < 0:
                delta = -1.0
            else:
                delta = new_score - base_score
                
            # Gradient direction: if delta > 0, increasing the value (if current is 0) or decreasing (if current is 1) is good.
            # We want to move towards 1 if current is 0 and delta > 0.
            # We want to move towards 0 if current is 1 and delta > 0 (so decreasing value increases score).
            
            if vals[idx] == 0:
                # Flipping to 1 changed score by delta.
                # So increasing value by 1 changes score by delta.
                grad[idx] = delta
            else:
                # Flipping to 0 changed score by delta.
                # So decreasing value by 1 changes score by delta.
                # Increasing value by 1 would change score by -delta.
                grad[idx] = -delta
                
        return grad

    # Initial design: Rod
    def make_initial():
        grid = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    grid[i, j] = 1
        vals = grid_to_orbit(grid)
        return vals

    current_vals = make_initial()
    best_vals = current_vals.copy()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 20s margin
    
    current_score, current_info = evaluate_and_get_eigenvectors(current_vals)
    best_score = current_score
    best_vals = current_vals.copy()
    
    # Gradient Ascent Loop
    step_size = 0.5
    
    while time.time() < deadline:
        # Compute gradient
        grad = compute_gradient(current_vals, current_info)
        
        # Propose update: move towards higher score
        # new_vals = current_vals + step_size * grad
        # But we need to keep values in [0, 1] or binary?
        # The problem asks for 0/1 pixels.
        # So we should project or use a thresholding scheme.
        # Let's use a soft update and then threshold.
        
        proposed_vals = current_vals + step_size * grad
        
        # Threshold to 0/1
        proposed_binary = (proposed_vals > 0.5).astype(int)
        
        # Evaluate proposed
        new_score, new_info = evaluate_and_get_eigenvectors(proposed_binary)
        
        if new_score > current_score:
            current_vals = proposed_binary.astype(float)
            current_score = new_score
            current_info = new_info
            
            if current_score > best_score:
                best_score = current_score
                best_vals = current_vals.copy()
        else:
            # If it fails, try a smaller step or random restart from best?
            # Let's try a smaller step for a few iterations.
            step_size *= 0.8
            if step_size < 0.1:
                # Restart from best with a small perturbation
                current_vals = best_vals.copy()
                current_score, current_info = evaluate_and_get_eigenvectors(current_vals)
                step_size = 0.5
                # Perturb a few random orbits
                for _ in range(10):
                    idx = np.random.randint(n_orbits)
                    current_vals[idx] = 1 - current_vals[idx]
                current_score, current_info = evaluate_and_get_eigenvectors(current_vals)
                
    # Return the best grid found
    best_grid = orbit_to_grid(best_vals)
    # Convert to list of lists of ints
    return best_grid.astype(int).tolist()
