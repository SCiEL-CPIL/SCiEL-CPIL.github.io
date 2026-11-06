import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Basin Hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Constants for Dzugutov potential
    A = 5.82
    B = 1.28
    a = 1.87
    b = 1.94
    c = 1.10
    d = 0.27
    m = 16
    
    # Time budget: 300s total / 6 instances = 50s per instance. 
    # We use 45s to be safe.
    time_limit_per_instance = 45.0
    start_time = time.time()
    
    def compute_energy_gradient(pos):
        """
        Vectorized computation of Dzugutov energy and gradient.
        pos: (n, 3) array
        Returns: energy (float), gradient (n, 3) array
        """
        n_atoms = pos.shape[0]
        # Pairwise differences: (n, n, 3)
        # rij[i, j] = pos[i] - pos[j]
        rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        # Distances: (n, n)
        r = np.linalg.norm(rij, axis=2)
        
        # Mask for i < j to avoid double counting and self-interaction
        i_idx, j_idx = np.triu_indices(n_atoms, k=1)
        
        # Extract pairwise vectors and distances for i < j
        rij_pairs = rij[i_idx, j_idx] # (N_pairs, 3)
        r_pairs = r[i_idx, j_idx]     # (N_pairs,)
        
        # Initialize energy and gradient
        energy = 0.0
        grad = np.zeros_like(pos)
        
        # 1. Pairs with r < a
        mask_a = r_pairs < a
        # 2. Pairs with a <= r < b
        mask_b = (r_pairs >= a) & (r_pairs < b)
        
        # Calculate for mask_a
        if np.any(mask_a):
            r_a = r_pairs[mask_a]
            rij_a = rij_pairs[mask_a]
            
            # Avoid r=0
            r_safe = np.maximum(r_a, 1e-10)
            
            # Term: A * (r^-m - B) * exp(c / (r - a))
            term_r = r_safe ** (-m)
            term_exp = np.exp(c / (r_safe - a))
            v_a = A * (term_r - B) * term_exp
            
            # Gradient of V wrt r:
            dv_dr_a = A * ( 
                (-m) * (r_safe ** (-m - 1)) * term_exp 
                + (term_r - B) * term_exp * (-c) / ((r_safe - a) ** 2)
            )
            
            # Gradient wrt pos:
            factor_a = dv_dr_a / r_safe
            grad_vec_a = factor_a[:, np.newaxis] * rij_a
            
            np.add.at(grad, i_idx[mask_a], grad_vec_a)
            np.add.at(grad, j_idx[mask_a], -grad_vec_a)
            
            energy += np.sum(v_a)
            
        if np.any(mask_b):
            r_b = r_pairs[mask_b]
            rij_b = rij_pairs[mask_b]
            
            r_safe = np.maximum(r_b, 1e-10)
            
            # Term: B * exp(d / (r - b))
            term_exp = np.exp(d / (r_safe - b))
            v_b = B * term_exp
            
            # Gradient:
            dv_dr_b = v_b * (-d) / ((r_safe - b) ** 2)
            
            factor_b = dv_dr_b / r_safe
            grad_vec_b = factor_b[:, np.newaxis] * rij_b
            
            np.add.at(grad, i_idx[mask_b], grad_vec_b)
            np.add.at(grad, j_idx[mask_b], -grad_vec_b)
            
            energy += np.sum(v_b)
            
        return energy, grad

    def local_minimize(pos):
        """
        Local minimization using L-BFGS-B.
        """
        def fun(x):
            p = x.reshape(-1, 3)
            e, g = compute_energy_gradient(p)
            return e, g.flatten()
        
        res = minimize(
            fun, 
            x0=pos.flatten(), 
            jac=True, 
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20}
        )
        return res.fun, res.x.reshape(-1, 3)

    def generate_initial(n, seed):
        """
        Generate a random initial configuration with minimum distance ~0.5.
        """
        rng = np.random.default_rng(seed)
        radius = 2.5 * n ** (1.0 / 3.0)
        
        points = []
        points.append(np.zeros(3))
        
        attempts = 0
        while len(points) < n and attempts < 50000:
            attempts += 1
            r = radius * rng.uniform(0.1, 1.0)
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            x = r * np.sqrt(1 - z**2) * np.cos(theta)
            y = r * np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z])
            
            if len(points) > 0:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 0.6:
                    continue
            
            points.append(p)
            
        while len(points) < n:
            points.append(rng.normal(0, radius, 3))
            
        points = np.array(points[:n])
        points = points - np.mean(points, axis=0)
        return points

    def basin_hopping(seed):
        """
        Run a single Basin Hopping trajectory.
        """
        rng = np.random.default_rng(seed)
        
        x = generate_initial(n, seed)
        
        try:
            e_curr, x_curr = local_minimize(x)
        except Exception:
            return float('inf'), x
            
        best_e = e_curr
        best_x = x_curr.copy()
        
        T = 10.0
        T_min = 0.01
        kick_size = 0.5
        n_iter = 200
        
        for i in range(n_iter):
            if time.time() - start_time > time_limit_per_instance - 2:
                break
                
            noise = rng.normal(0, kick_size, (n, 3))
            x_perturbed = x_curr + noise
            
            try:
                e_new, x_new = local_minimize(x_perturbed)
            except Exception:
                continue
                
            delta_e = e_new - e_curr
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                x_curr = x_new
                e_curr = e_new
                
                if e_curr < best_e:
                    best_e = e_curr
                    best_x = x_curr.copy()
            
            T *= 0.95
            if T < T_min:
                T = T_min
            
            kick_size = 0.4 + 0.1 * (T / 10.0)
            
        return best_e, best_x

    n_threads = 4
    best_energy = float('inf')
    best_positions = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            seed = 1000 + i * 100
            futures.append(executor.submit(basin_hopping, seed))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
    
    if best_positions is None:
        rng = np.random.default_rng(n)
        points = rng.normal(0, 1, (n, 3))
        points = points - np.mean(points, axis=0)
        _, best_positions = relax(points, potential)
        
    return best_positions

# EVOLVE-BLOCK-END
