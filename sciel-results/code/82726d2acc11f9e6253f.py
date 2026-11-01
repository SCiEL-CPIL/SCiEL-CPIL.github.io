# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a trust-region constrained optimization approach (trust-constr) 
    which acts as a robust relaxation for the rigid-body constraints.
    It directly maximizes the radius r subject to non-overlap and boundary constraints.
    """
    N = 50
    
    # Variables: x[0:2N] are centers, x[2N] is radius r
    def objective(x):
        # We want to maximize r, so minimize -r
        return -x[2 * N]
    
    def constraints(x):
        c = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        
        cons = []
        
        # 1. Boundary constraints: |c_i| <= 1 - r
        # Inequality: 1 - r - |c_i| >= 0
        for i in range(N):
            dist = np.linalg.norm(c[i])
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: (1.0 - x[2*N]) - np.linalg.norm(x[2*i:2*i+2]),
                'jac': lambda x, i=i: _jac_boundary(x, i)
            })
            
        # 2. Pairwise constraints: |c_i - c_j| >= 2r
        # Inequality: |c_i - c_j| - 2r >= 0
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - 2*x[2*N],
                    'jac': lambda x, i=i, j=j: _jac_pair(x, i, j)
                })
        
        return cons

    def _jac_boundary(x, i):
        # Gradient of (1 - r - ||c_i||)
        # w.r.t c_i: -c_i / ||c_i||
        # w.r.t r: -1
        c_i = x[2*i:2*i+2]
        norm = np.linalg.norm(c_i)
        if norm < 1e-10:
            grad_c = np.array([0.0, 0.0])
        else:
            grad_c = -c_i / norm
        grad_r = -1.0
        return np.concatenate([grad_c, [grad_r]])
        
    def _jac_pair(x, i, j):
        # Gradient of (||c_i - c_j|| - 2r)
        # w.r.t c_i: (c_i - c_j) / ||c_i - c_j||
        # w.r.t c_j: (c_j - c_i) / ||c_i - c_j||
        # w.r.t r: -2
        c_i = x[2*i:2*i+2]
        c_j = x[2*j:2*j+2]
        diff = c_i - c_j
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            grad_i = np.array([0.0, 0.0])
            grad_j = np.array([0.0, 0.0])
        else:
            grad_i = diff / norm
            grad_j = -diff / norm
        grad_r = -2.0
        return np.concatenate([grad_i, grad_j, [grad_r]])

    def generate_initial_configurations():
        """
        Generate diverse initial configurations for the optimizer.
        """
        configs = []
        
        # 1. Hexagonal lattice
        scale = 0.15
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:N]
            configs.append(points[indices])
            
        # 2. Concentric rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        # 3. Random restarts
        np.random.seed(42)
        for _ in range(3):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        return configs

    best_centers = None
    best_r = -np.inf
    best_x = None

    configs = generate_initial_configurations()
    
    # Run optimization from each initial configuration
    for c_init in configs:
        x0 = np.concatenate([c_init.flatten(), [0.1]]) # Initial guess for r
        
        try:
            res = minimize(
                objective,
                x0,
                constraints=constraints(x0), # Note: passing constraints correctly
                method='trust-constr',
                options={'maxiter': 500, 'gtol': 1e-8, 'verbose': 0}
            )
            
            if res.success or (res.status == 0):
                r_opt = res.x[2 * N]
                if r_opt > best_r:
                    best_r = r_opt
                    best_x = res.x
                    best_centers = res.x[:2 * N].reshape(N, 2)
                    
        except Exception as e:
            # Fallback to penalty method if trust-constr fails
            pass
            
    # Fallback if trust-constr failed or didn't find a good solution
    if best_centers is None or best_r < 0.1:
         # Simple penalty method fallback
         def energy_and_grad(c, r):
             c = c.reshape(N, 2)
             energy = 0.0
             grad = np.zeros_like(c)
             diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
             dists = np.sqrt(np.sum(diff ** 2, axis=2))
             np.fill_diagonal(dists, np.inf)
             overlap = np.maximum(0, 2 * r - dists)
             pair_energy = 0.5 * np.sum(overlap ** 2)
             energy += pair_energy
             safe_dists = np.where(dists > 1e-10, dists, 1.0)
             np.fill_diagonal(overlap, 0)
             weights = -2.0 * overlap / safe_dists
             np.fill_diagonal(weights, 0)
             grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
             grad += grad_pairs
             radii = np.linalg.norm(c, axis=1)
             bound_violation = np.maximum(0, radii - (1.0 - r))
             energy += np.sum(bound_violation ** 2)
             safe_radii = np.where(radii > 1e-10, radii, 1.0)
             grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
             grad += grad_bound
             return energy, grad

         def solve_for_radius(r, c_init):
             c_flat = c_init.flatten()
             def obj(x):
                 e, _ = energy_and_grad(x, r)
                 return e
             def jac(x):
                 _, g = energy_and_grad(x, r)
                 return g.flatten()
             try:
                 res = minimize(obj, c_flat, jac=jac, method='L-BFGS-B', 
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                 return res.x.reshape(N, 2), res.fun
             except:
                 return c_init, np.inf

         # Binary search fallback
         r_low = 0.1
         r_high = 0.13
         best_c_fb = configs[0]
         best_r_fb = r_low
         
         for _ in range(15):
             r_mid = (r_low + r_high) / 2
             feasible = False
             best_e = np.inf
             best_c = None
             for c_init in configs:
                 c_opt, e_opt = solve_for_radius(r_mid, c_init)
                 if e_opt < best_e:
                     best_e = e_opt
                     best_c = c_opt
                 if e_opt < 1e-6:
                     feasible = True
             if feasible:
                 r_low = r_mid
                 best_r_fb = r_mid
                 best_c_fb = best_c
             else:
                 r_high = r_mid
                 
         if best_r_fb > best_r:
             best_r = best_r_fb
             best_centers = best_c_fb

    return best_centers
# EVOLVE-BLOCK-END
