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

    def _softmin_radius_and_grad(C, beta):
        """
        Computes the negative soft-min radius and its gradient.
        Objective: minimize -softmin(radius_terms)
        softmin(x) = -1/beta * log(sum(exp(-beta * x)))
        """
        C = np.asarray(C).reshape(-1, 3)
        n = len(C)
        
        # Pairwise terms: d_ij / 2
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
        np.fill_diagonal(dists, np.inf)
        
        triu_idx = np.triu_indices(n, k=1)
        pair_dists = dists[triu_idx]
        pair_terms = pair_dists / 2.0
        
        # Boundary terms: 0.5 - max(|c_ik|)
        # We use max(|c_ik|) for the Chebyshev distance to the boundary
        max_abs = np.max(np.abs(C), axis=1)
        boundary_terms = 0.5 - max_abs
        
        all_terms = np.concatenate([pair_terms, boundary_terms])
        
        # Soft min calculation using log-sum-exp for numerical stability
        min_val = np.min(all_terms)
        shifted = -beta * (all_terms - min_val)
        shifted = np.clip(shifted, -700, 700)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0 or not np.isfinite(sum_exp):
            # Fallback to hard min if numerical issues
            weights = np.zeros_like(all_terms)
            min_indices = np.where(all_terms == min_val)[0]
            weights[min_indices] = 1.0 / len(min_indices)
            soft_min_val = min_val
        else:
            weights = exp_vals / sum_exp
            soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
        
        # We want to minimize -soft_min_val
        obj_val = -soft_min_val
        
        grad_C = np.zeros_like(C)
        
        # Pair gradients
        pair_weights = weights[:len(pair_terms)]
        i_idx = triu_idx[0]
        j_idx = triu_idx[1]
        
        d_vecs = C[i_idx] - C[j_idx]
        d_norms = np.linalg.norm(d_vecs, axis=1)
        d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
        
        # d(term_k)/d(c_i) for pair term (i,j): (c_i - c_j) / (2 * d_ij)
        # d(obj)/d(C) = - sum w_k * d(term_k)/d(C)
        # So grad = - sum w_k * d_vec / (2 * d_norm)
        grad_pair_i = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
        grad_pair_j = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
        
        np.add.at(grad_C, i_idx, grad_pair_i)
        np.add.at(grad_C, j_idx, grad_pair_j)
        
        # Boundary gradients
        # term = 0.5 - max(|c_ik|)
        # d(term)/d(c_ik) = -sign(c_ik) if |c_ik| == max, else 0
        bnd_weights = weights[len(pair_terms):].reshape(n, 3)
        
        abs_C = np.abs(C)
        max_idx = np.argmax(abs_C, axis=1)
        
        # Create a mask for the max coordinate
        rows = np.arange(n)
        signs = np.sign(C[rows, max_idx])
        signs[signs == 0] = 1.0
        
        # Gradient contribution for each point
        # grad_C[i] += -w_i * sign(c_i, max_idx) * e_max_idx
        # Note: bnd_weights is already the weight for the boundary term of point i
        # The term is scalar per point.
        # d(term_i)/d(c_i) is a vector with -sign at the max coord.
        
        grad_bnd = np.zeros_like(C)
        grad_bnd[rows, max_idx] = -signs * bnd_weights[:, 0] # bnd_weights is shape (n,3) but only 1 non-zero? 
        # Wait, weights for boundary terms are separate.
        # all_terms has n boundary terms.
        # bnd_weights is shape (n, 3) if we reshape? No, len(pair_terms) + n = total terms.
        # weights is 1D.
        # bnd_weights = weights[len(pair_terms):] is shape (n,)
        
        # Correction:
        bnd_weights_flat = weights[len(pair_terms):]
        
        grad_bnd[rows, max_idx] = -signs * bnd_weights_flat
        
        grad_C += grad_bnd
        
        return obj_val, grad_C

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
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 1e-6
        step_size = 0.05
        
        while time.time() < deadline - 5.0 and T > min_T:
            # Propose move
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, size=3)
            new_c = current_c.copy()
            new_c[i] += delta
            
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
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # 3. Another Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_overall_c = inits[0].copy()
    best_overall_r = _exact_radius(best_overall_c)
    
    # Run SA on each init
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        c, r = sa_run(init, 0)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_c = c.copy()
            
    # Final Polish with L-BFGS-B using Soft-Min objective
    if time.time() < deadline - 2.0:
        def obj_and_grad(x):
            c = x.reshape(N, 3)
            val, grad = _softmin_radius_and_grad(c, beta=50.0)
            return val, grad.flatten()
        
        x0 = best_overall_c.flatten()
        
        try:
            res = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                           jac=True,
                           options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12, 'disp': False})
            
            if res.success or time.time() < deadline - 1.0:
                c_polished = res.x.reshape(N, 3)
                # Clip to ensure validity
                c_polished = np.clip(c_polished, -0.5, 0.5)
                r_polished = _exact_radius(c_polished)
                if r_polished > best_overall_r:
                    best_overall_c = c_polished
                    best_overall_r = r_polished
        except Exception:
            pass

    return best_overall_c

# EVOLVE-BLOCK-END
