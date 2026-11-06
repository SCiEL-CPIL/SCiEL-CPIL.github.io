"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def _exact_radius(C):
        """Compute the exact packing radius."""
        C = np.asarray(C).reshape(-1, 3)
        n = len(C)
        if n < 2:
            return np.inf
        
        # Pairwise distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distance
        min_bnd = np.min(0.5 - np.max(np.abs(C), axis=1))
        
        return min(min_pair, min_bnd)

    def _repulsive_energy(C):
        """
        Computes the energy to be minimized for SA.
        Minimizing sum(1/d_ij^2) encourages maximizing the minimum distance.
        Boundary penalty keeps spheres inside [-0.5, 0.5].
        """
        c_sq = np.sum(C * C, axis=1)
        d2 = c_sq[:, np.newaxis] + c_sq[np.newaxis, :] - 2.0 * (C @ C.T)
        np.maximum(d2, 1e-12, out=d2)
        np.fill_diagonal(d2, 1.0)
        
        e_pair = np.sum(1.0 / d2) - N
        
        abs_c = np.abs(C)
        violations = np.maximum(0.0, abs_c - 0.5)
        e_boundary = np.sum(violations * violations) * 100.0
        
        return e_pair + e_boundary

    def sa_run(init, time_budget):
        """
        Runs Simulated Annealing starting from init.
        Returns the best center found.
        """
        current_c = init.copy()
        current_e = _repulsive_energy(current_c)
        best_c = current_c.copy()
        best_e = current_e
        best_r = _exact_radius(best_c)
        
        T = 0.5
        cooling_rate = 0.99995
        min_T = 1e-6
        step_size = 0.05
        
        # Pre-allocate for speed if needed, but copy is fine for N=50
        while time.time() < deadline - 30.0 and T > min_T:
            # Propose move
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, size=3)
            new_c = current_c.copy()
            new_c[i] += delta
            
            # Quick boundary check to avoid expensive energy calc if way out
            if np.any(np.abs(new_c) > 0.51):
                T *= cooling_rate
                continue

            new_e = _repulsive_energy(new_c)
            
            delta_e = new_e - current_e
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_c = new_c
                current_e = new_e
                
                if current_e < best_e:
                    best_c = current_c.copy()
                    best_e = current_e
                    r = _exact_radius(best_c)
                    if r > best_r:
                        best_r = r
            
            T *= cooling_rate
            
        return best_c, best_r

    # Initialize multiple configurations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Random
    np.random.seed(42) # Fixed seed for reproducibility in evaluation if needed, or random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # 3. Another Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))

    best_overall_c = inits[0].copy()
    best_overall_r = _exact_radius(best_overall_c)
    
    # Run SA on each init
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        c, r = sa_run(init, 0)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_c = c.copy()

    # Final Polish with SLSQP Epigraph Formulation
    if time.time() < deadline - 5.0:
        def epigraph_obj(x):
            # x = [c_0..c_{N-1}, t]
            # Minimize -t
            return -x[-1]

        def epigraph_constraints(x):
            C = x[:-1].reshape(N, 3)
            t = x[-1]
            
            constraints = []
            
            # Pairwise constraints: 2*t - dist_ij <= 0
            # Vectorized calculation for values
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            triu_idx = np.triu_indices(N, k=1)
            pair_dists = dists[triu_idx]
            # Constraint value: 2*t - d_ij
            c_vals = 2.0 * t - pair_dists
            
            # Boundary constraints: t + |c_ik| - 0.5 <= 0
            # c_ik is x[i*3 + k]
            abs_C = np.abs(C)
            bnd_vals = t + abs_C - 0.5
            
            # SLSQP expects a list of dicts or a function returning array of values
            # We will use the 'type' inco in a list of dicts for clarity, 
            # but for speed, we can define the function to return the array of inequality values.
            # scipy.optimize.minimize with SLSQP and constraints=[{'type': 'ineq', 'fun': ...}]
            # The 'fun' must return an array of values >= 0 for ineq.
            # Our condition is: 2*t - d <= 0  => d - 2*t >= 0
            # Our condition is: t + |c| - 0.5 <= 0 => 0.5 - t - |c| >= 0
            
            return {
                'type': 'ineq',
                'fun': lambda x: np.concatenate([
                    pair_dists[x] - 2.0 * x[-1],  # This lambda is wrong, x changes
                ])
            }
            
        # The lambda above is incorrect because x is captured by reference or similar issues in closure.
        # Better to define the constraint function properly.

        def cons_fun(x):
            C = x[:-1].reshape(N, 3)
            t = x[-1]
            
            # Pairwise
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            triu_idx = np.triu_indices(N, k=1)
            pair_dists = dists[triu_idx]
            c_pair = pair_dists - 2.0 * t
            
            # Boundary
            abs_C = np.abs(C)
            c_bnd = 0.5 - t - abs_C
            
            return np.concatenate([c_pair, c_bnd])

        def cons_jac(x):
            C = x[:-1].reshape(N, 3)
            t = x[-1]
            
            # Jacobian shape: (num_constraints, 3*N + 1)
            # num_constraints = N*(N-1)/2 + N*3
            n_pairs = N * (N - 1) // 2
            n_bnds = N * 3
            J = np.zeros((n_pairs + n_bnds, 3 * N + 1))
            
            # Pairwise part
            # d(dist_ij - 2t)/dx
            # d(dist)/dt = -2
            # d(dist)/dc_i = (c_i - c_j) / dist
            # d(dist)/dc_j = (c_j - c_i) / dist
            
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            triu_i, triu_j = np.triu_indices(N, k=1)
            pair_dists = dists[triu_i, triu_j]
            
            # Avoid division by zero
            safe_dists = np.where(pair_dists < 1e-12, 1e-12, pair_dists)
            
            # Gradient wrt t
            J[:n_pairs, -1] = -2.0
            
            # Gradient wrt C
            # Vector from i to j: c_i - c_j
            vecs = C[triu_i] - C[triu_j]
            # Unit vectors
            unit_vecs = vecs / safe_dists[:, np.newaxis]
            
            # For each pair k (index in triu arrays):
            # d(dist_k)/dc_i = unit_vec_k
            # d(dist_k)/dc_j = -unit_vec_k
            
            # We need to scatter these into the J matrix.
            # J[k, i*3:(i+1)*3] += unit_vec_k
            # J[k, j*3:(j+1)*3] += -unit_vec_k
            
            # Efficient scattering using np.add.at
            # Flatten C indices
            i_flat = triu_i * 3
            j_flat = triu_j * 3
            
            # We need to add 3 components for each
            # Create a temporary array for the flat gradient
            grad_flat = np.zeros(3 * N)
            
            # This is tricky with add.at for multiple components. 
            # Let's just loop for the Jacobian as N=50 is small (1225 pairs).
            # 1225 iterations is very fast.
            for k in range(n_pairs):
                i = triu_i[k]
                j = triu_j[k]
                uv = unit_vecs[k]
                J[k, i*3:(i+1)*3] += uv
                J[k, j*3:(j+1)*3] += -uv

            # Boundary part
            # c_bnd = 0.5 - t - |c_ik|
            # d/dt = -1
            # d/dc_ik = -sign(c_ik)
            
            J[n_pairs:, -1] = -1.0
            
            # Scatter boundary gradients
            # c_bnd indices are n_pairs + i*3 + k
            for i in range(N):
                for k in range(3):
                    val = C[i, k]
                    sign = 1.0 if val >= 0 else -1.0
                    J[n_pairs + i*3 + k, i*3 + k] = -sign

            return J

        def obj_grad(x):
            # Minimize -t
            # grad = [0, ..., 0, -1]
            g = np.zeros(3 * N + 1)
            g[-1] = -1.0
            return g

        # Run SLSQP
        # Initial guess
        x0 = np.concatenate([best_overall_c.flatten(), [best_overall_r * 0.9]])
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (3 * N) + [(0.0, 0.5)]
        
        try:
            # Stage 1: Moderate
            res1 = minimize(
                epigraph_obj, x0, method='SLSQP', 
                jac=obj_grad,
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac}],
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )
            
            if res1.success or (res1.status == 0):
                x1 = res1.x
                # Stage 2: Tight
                if time.time() < deadline - 2.0:
                    res2 = minimize(
                        epigraph_obj, x1, method='SLSQP', 
                        jac=obj_grad,
                        bounds=bounds,
                        constraints=[{'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac}],
                        options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                    )
                    if res2.success or (res2.status == 0):
                        x_final = res2.x
                    else:
                        x_final = x1
                else:
                    x_final = x1
                
                c_final = x_final[:-1].reshape(N, 3)
                # Clip to ensure validity
                c_final = np.clip(c_final, -0.5, 0.5)
                r_final = _exact_radius(c_final)
                
                if r_final > best_overall_r:
                    best_overall_c = c_final
                    best_overall_r = r_final

        except Exception:
            pass

    return best_overall_c

# EVOLVE-BLOCK-END
