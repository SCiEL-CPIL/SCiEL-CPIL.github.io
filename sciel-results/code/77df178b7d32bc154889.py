"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_centers = None
    best_radius = 0.0

    def _radius_for_centers(centers):
        centers = np.asarray(centers)
        if centers.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        if min_wall < 0:
            return 0.0
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return min(min_wall, min_pair)

    def _make_bcc_initial():
        """
        Create a distorted BCC-like initialization.
        We use a 4x4x4 grid of BCC unit cells.
        Each cell has 2 points: corner and body center.
        Total 128 points. We select 50 by keeping the ones closest to the origin 
        or by taking the first 50 in a space-filling order, then optimize.
        Actually, simpler: Generate a 4x4x4 BCC lattice (128 pts) and pick 50 
        that are well-distributed (e.g., every other point or random selection 
        with repulsion).
        
        Let's generate a BCC lattice in the box [-0.5, 0.5]^3.
        Grid size 4x4x4.
        Points: (i/4 - 0.5, j/4 - 0.5, k/4 - 0.5) and (i/4 - 0.5 + 1/8, j/4 - 0.5 + 1/8, k/4 - 0.5 + 1/8)
        """
        m = 4
        # Corners
        x, y, z = np.meshgrid(
            np.linspace(-0.5, 0.5, m, endpoint=False) - 0.5/m,
            np.linspace(-0.5, 0.5, m, endpoint=False) - 0.5/m,
            np.linspace(-0.5, 0.5, m, endpoint=False) - 0.5/m,
            indexing='ij'
        )
        corners = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        
        # Body centers (shifted by half cell size)
        shift = 0.5 / m
        centers = np.column_stack((x.ravel() + shift, y.ravel() + shift, z.ravel() + shift))
        
        all_pts = np.vstack([corners, centers])
        
        # We have 128 points. We need 50.
        # To ensure good distribution, we can just take a random subset 
        # but ensure they are spread out. 
        # A simple heuristic: sort by distance to center, take the first 50? 
        # No, that clusters them in the center.
        # Better: take every ~2.56th point in a shuffled order?
        # Or just use the first 50 of a space-filling curve (Hilbert) over the BCC points.
        
        # Simple approach: Random selection with a seed, then optimize.
        # But random selection from BCC might have close pairs if unlucky.
        # Let's try to select points that maximize the minimum distance among themselves 
        # from the BCC set. This is a small k-center problem on 128 points.
        
        # Greedy selection for k-center on the BCC points
        selected = [0] # Start with point 0
        # Actually, let's start with the center of the box if available, or closest to it.
        dist_to_origin = np.sum(all_pts**2, axis=1)
        start_idx = np.argmin(dist_to_origin)
        selected = [start_idx]
        
        for _ in range(N - 1):
            # Calculate min distance from each candidate to selected points
            dists = np.inf * np.ones(len(all_pts))
            for idx in selected:
                d = np.linalg.norm(all_pts - all_pts[idx], axis=1)
                dists = np.minimum(dists, d)
            
            # Exclude already selected
            for idx in selected:
                dists[idx] = -1.0
            
            # Pick the point with the maximum min-distance
            next_idx = np.argmax(dists)
            selected.append(next_idx)
            
        return all_pts[selected]

    def _make_grid_initial():
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid_pts = []
        for z in ticks:
            for y in ticks:
                for x in ticks:
                    grid_pts.append([x, y, z])
        return np.array(grid_pts[:N])

    def _make_random_initial(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    initializations = [
        _make_bcc_initial(),
        _make_grid_initial(),
        _make_random_initial(42),
        _make_random_initial(123)
    ]

    # SLSQP Epigraph Formulation
    # Variables: x (N*3) and r (1). Total N*3 + 1.
    # Objective: minimize -r
    # Constraints:
    # 1. For all i<j: ||x_i - x_j|| - 2r >= 0
    # 2. For all i, k: 0.5 - |x_ik| - r >= 0
    
    def _objective(x):
        return -x[-1] # Minimize -r

    def _constraints(x):
        centers = x[:-1].reshape(N, 3)
        r = x[-1]
        
        cons = []
        
        # Pairwise constraints
        # To reduce the number of constraints for SLSQP, we can use a subset 
        # or just all. 50*49/2 = 1225 constraints. This might be slow for SLSQP.
        # SLSQP handles many constraints but it's quadratic.
        # Alternative: Use a penalty method or trust-constr.
        # Let's try SLSQP first. If too slow, switch.
        
        # Actually, 1225 constraints + 150 wall constraints = 1375 constraints.
        # This is heavy.
        
        # Let's use a "soft" constraint approach or just optimize the energy 
        # but with a better potential?
        # The prompt asks for a DIFFERENT strategy. Epigraph is different.
        # But if it's too slow, we fail.
        
        # Let's try `trust-constr` which handles constraints better?
        # Or stick to SLSQP but maybe reduce constraints?
        
        # For now, let's implement the full epigraph.
        
        # Generate all pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Constraint: dists - 2r >= 0
        pair_cons = dists - 2.0 * r
        
        # Wall constraints
        # For each coordinate
        walls = 0.5 - np.abs(centers)
        wall_cons = walls.ravel() - r
        
        # Combine
        all_cons = np.concatenate([pair_cons, wall_cons])
        
        # SLSQP requires a list of dicts or a single dict with 'fun' returning array.
        # We'll return a single dict with 'type': 'ineq' and 'fun' returning the array.
        return {'type': 'ineq', 'fun': lambda x: all_cons, 'jac': None} # Note: lambda captures x, but SLSQP calls fun(x). 
        # Wait, the lambda needs to take x and return the value.
        # The 'fun' in the dict is called with x.
        # So:
        def fun(x):
             c = x[:-1].reshape(N, 3)
             r = x[-1]
             i_idx, j_idx = np.triu_indices(N, k=1)
             diffs = c[i_idx] - c[j_idx]
             dists = np.linalg.norm(diffs, axis=1)
             pair_cons = dists - 2.0 * r
             walls = 0.5 - np.abs(c)
             wall_cons = walls.ravel() - r
             return np.concatenate([pair_cons, wall_cons])
        
        return {'type': 'ineq', 'fun': fun}

    # SLSQP might be too slow with 1300+ constraints.
    # Let's try a different approach: 
    # Minimize the "violation" of the constraints.
    # E = sum(max(0, 2r - d_ij)^2 + max(0, r - wall_i)^2)
    # This is smooth if we use softplus or just squared hinge.
    # But this is back to energy minimization.
    
    # Let's try `trust-constr` with the linearized constraints? No.
    
    # Let's try a simpler epigraph:
    # Maximize r
    # Subject to:
    # d_ij >= 2r
    # wall_i >= r
    
    # We can use `scipy.optimize.minimize` with method='SLSQP' and bounds on r.
    # Bounds: r in [0, 0.5]. Centers in [-0.5, 0.5].
    
    # If SLSQP is too slow, we fall back to the energy method but with BCC init.
    # Let's try to run SLSQP with a time limit per initialization.
    
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]

    for init_idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        x0 = np.concatenate([init_centers.flatten(), [0.01]])
        
        try:
            # Use SLSQP
            # maxiter is important. 1300 constraints is heavy.
            # Let's try 100 iterations.
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=_constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.fun < 0: # -r < 0 => r > 0
                new_centers = result.x[:-1].reshape(N, 3)
                r_val = -result.fun
                
                # Verify and calculate true radius
                if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5):
                    true_radius = _radius_for_centers(new_centers)
                    if true_radius > best_radius:
                        best_radius = true_radius
                        best_centers = new_centers.copy()
                        
        except Exception as e:
            # SLSQP might fail. Fall back to energy minimization for this init?
            # Or just skip.
            pass

    # If SLSQP didn't find anything good, or to polish, use the energy method on the best found so far
    # or on the BCC init if best_radius is low.
    
    if best_radius < 0.13:
        # Use the energy method as a fallback/polisher on the BCC init
        def _energy_and_grad(x_flat):
            centers = x_flat.reshape(N, 3)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            
            q = 4.0 # Stronger repulsion
            
            inv_dist_sq_q = dist_sq ** (-q)
            np.fill_diagonal(inv_dist_sq_q, 0.0)
            
            E_pair = 0.5 * np.sum(inv_dist_sq_q)
            
            W = dist_sq ** (-(q + 1))
            np.fill_diagonal(W, 0.0)
            
            S = np.sum(W, axis=1)
            Wx = W.dot(centers)
            grad_pair = -2.0 * q * (S[:, np.newaxis] * centers - Wx)
            
            eps = 1e-8
            smooth_abs = np.sqrt(centers**2 + eps)
            d_wall = 0.5 - smooth_abs
            d_wall = np.clip(d_wall, 1e-6, 1.0)
            
            E_wall = np.sum(d_wall ** (-q))
            
            d_wall_grad_factor = d_wall ** (-(q + 1))
            ratio = centers / smooth_abs
            grad_wall = q * d_wall_grad_factor * ratio
            
            E_total = E_pair + E_wall
            grad_total = grad_pair + grad_wall
            
            return E_total, grad_total.ravel()

        def _neg_energy(x_flat):
            val, _ = _energy_and_grad(x_flat)
            return val

        def _neg_energy_grad(x_flat):
            _, g = _energy_and_grad(x_flat)
            return g

        # Try polishing the BCC init with energy method
        bcc_init = _make_bcc_initial().flatten()
        try:
            result = minimize(
                _neg_energy,
                bcc_init,
                jac=_neg_energy_grad,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.success or result.fun < 1e10:
                new_centers = result.x.reshape(N, 3)
                if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5):
                    radius = _radius_for_centers(new_centers)
                    if radius > best_radius:
                        best_radius = radius
                        best_centers = new_centers.copy()
        except Exception:
            pass

    if best_centers is None:
        # Fallback to grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
