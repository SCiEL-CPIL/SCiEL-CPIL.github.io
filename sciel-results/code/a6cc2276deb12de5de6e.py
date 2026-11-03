import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    def compute_radius(centers):
        """Compute the radius for a given configuration."""
        if centers is None or len(centers) != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        center_dists = np.sqrt(np.sum(centers**2, axis=1))
        min_boundary_dist = np.min(1.0 - center_dists)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r if r > 0 else 0.0

    def smooth_objective_and_grad(x, beta=100.0):
        """
        Compute the smooth objective function and its gradient.
        We minimize g(x) = (1/beta) * ln(sum(exp(-beta * m_k)))
        where m_k are the margins (boundary and pairwise).
        This approximates -min(m_k), so minimizing it maximizes the min margin.
        """
        points = x.reshape(N, 2)
        
        # 1. Compute margins and their gradients
        # Boundary margins: m_i = 1 - ||c_i||
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        m_boundary = 1.0 - norms
        # Gradient of m_boundary w.r.t x: -c_i / ||c_i||
        grad_m_boundary = -points / safe_norms[:, np.newaxis]
        
        # Pairwise margins: m_ij = ||c_i - c_j|| / 2
        # We need to accumulate gradients for each point
        # Vectorized computation of pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0) # Placeholder, we will mask it
        
        # Margins are dists / 2
        m_pairwise = dists / 2.0
        
        # Gradients of m_pairwise
        # grad m_ij w.r.t c_i = (c_i - c_j) / (2 * dist_ij) = diff[i,j] / (2 * dists[i,j])
        # grad m_ij w.r.t c_j = (c_j - c_i) / (2 * dist_ij) = -diff[i,j] / (2 * dists[i,j])
        
        # We will compute weights w_k = exp(-beta * m_k) / sum(exp(-beta * m_k))
        # Then grad g = - sum_k w_k * grad m_k
        
        # To avoid overflow/underflow in exp, use log-sum-exp trick
        # Let A = -beta * m
        # sum(exp(A)) = exp(max(A)) * sum(exp(A - max(A)))
        
        # Combine all margins into a flat array for easier handling?
        # No, let's keep them separate for gradient accumulation.
        
        # Boundary part
        A_b = -beta * m_boundary
        max_A_b = np.max(A_b)
        exp_A_b = np.exp(A_b - max_A_b)
        sum_exp_A_b = np.sum(exp_A_b)
        
        # Pairwise part
        # Mask diagonal to -inf so exp is 0
        mask = ~np.eye(N, dtype=bool)
        A_p = -beta * m_pairwise
        A_p[~mask] = -np.inf
        max_A_p = np.max(A_p[mask]) # Only consider valid pairs
        # If all are -inf, max is -inf, handle separately? Unlikely.
        exp_A_p = np.exp(A_p - max_A_p)
        exp_A_p[~mask] = 0.0
        sum_exp_A_p = np.sum(exp_A_p)
        
        # Total sum S = exp(max_A_b + max_A_p?) No, we need common shift.
        # Let M = max(max_A_b, max_A_p)
        M = max(max_A_b, max_A_p)
        
        # Recalculate with common shift M
        exp_A_b_shifted = np.exp(A_b - M)
        exp_A_p_shifted = np.exp(A_p - M)
        exp_A_p_shifted[~mask] = 0.0
        
        sum_exp_total = np.sum(exp_A_b_shifted) + np.sum(exp_A_p_shifted)
        
        if sum_exp_total == 0:
            # Fallback
            return 0.0, np.zeros_like(x)
            
        # Objective value: g(x) = (1/beta) * (M + ln(sum_exp_total))
        obj = (1.0 / beta) * (M + np.log(sum_exp_total))
        
        # Weights
        # w_b_i = exp(A_b_i - M) / sum_exp_total
        w_b = exp_A_b_shifted / sum_exp_total
        # w_p_ij = exp(A_p_ij - M) / sum_exp_total
        w_p = exp_A_p_shifted / sum_exp_total
        
        # Gradient accumulation
        grad = np.zeros_like(x)
        
        # Boundary contribution: grad g = - sum_i w_b_i * grad m_b_i
        # grad m_b_i = -points[i] / norms[i]
        # So contribution is + sum_i w_b_i * points[i] / norms[i]
        # Note: grad_m_boundary was defined as -points/safe_norms
        # grad g = - sum w_k grad m_k
        # For boundary: - w_b * (-points/safe_norms) = w_b * points/safe_norms
        
        # Vectorized: (w_b[:, np.newaxis] * grad_m_boundary).sum(axis=0)
        # grad_m_boundary = -points / safe_norms
        # So we sum w_b * (-points / safe_norms) and negate?
        # Let's stick to the formula: grad_g = - sum_k w_k grad_m_k
        
        # Boundary grad_m:
        # grad_m_b_i = -points[i] / safe_norms[i]
        # contrib_b = w_b[:, np.newaxis] * grad_m_boundary
        # sum_contrib_b = contrib_b.sum(axis=0)
        # grad += -sum_contrib_b
        
        # Pairwise grad_m:
        # grad_m_p_ij w.r.t x_i (which is 2*i, 2*i+1) is diff[i,j] / (2*dists[i,j])
        # grad_m_p_ij w.r.t x_j is -diff[i,j] / (2*dists[i,j])
        
        # We need to scatter the weights w_p to the points.
        # For each pair (i,j) with weight w_p[i,j]:
        # add w_p[i,j] * (diff[i,j] / (2*dists[i,j])) to grad[2*i:2*i+2]
        # add w_p[i,j] * (-diff[i,j] / (2*dists[i,j])) to grad[2*j:2*j+2]
        
        # Vectorized scatter:
        # term = w_p * (diff / (2 * dists))[:,:,np.newaxis]
        # dists is N x N. diff is N x N x 2.
        # safe_dists = np.where(dists < 1e-12, 1.0, dists)
        # term = w_p[:,:,np.newaxis] * (diff / (2 * safe_dists[:,:,np.newaxis]))
        # term[~mask, :] = 0
        
        # grad_i = sum_j term[i,j]
        # grad_j = -sum_i term[i,j] ... wait.
        # term[i,j] is the gradient contribution to point i from pair (i,j).
        # The gradient of the objective w.r.t point i is sum_j w_p[i,j] * grad_m_ij_i
        # grad_m_ij_i = diff[i,j] / (2 dists)
        
        # So, grad_p_i = sum_j w_p[i,j] * (diff[i,j] / (2 dists[i,j]))
        # grad_p_j = sum_i w_p[i,j] * (-diff[i,j] / (2 dists[i,j]))
        
        # This is equivalent to:
        # Let V = diff / (2 * dists)[:,:,np.newaxis]
        # W = w_p
        # grad_from_pairs = sum_j W[i,j] V[i,j] for point i
        # And for point j, it receives -V[i,j] W[i,j] from pair (i,j).
        # So total grad for point k is:
        # sum_{j != k} W[k,j] V[k,j] - sum_{i != k} W[i,k] V[i,k]
        
        # Let's compute V
        safe_dists = np.where(dists < 1e-12, 1.0, dists)
        V = diff / (2.0 * safe_dists[:,:,np.newaxis])
        
        # Mask diagonal
        V[~mask, :] = 0.0
        
        # W is w_p
        # Term for point i from outgoing pairs: (W * V).sum(axis=1)
        # Term for point i from incoming pairs: - (W * V).sum(axis=0)
        
        grad_out = (w_p * V).sum(axis=1)
        grad_in = (w_p * V).sum(axis=0)
        
        # Total pairwise gradient contribution to g
        # grad_g_pairs = - (grad_out - grad_in) ?
        # Let's re-verify signs.
        # g = (1/beta) ln sum exp(-beta m)
        # dg/dx = - sum w_k (dm_k/dx)
        # For pair (i,j), m_ij = ||c_i - c_j||/2
        # dm_ij/dc_i = (c_i - c_j) / (2 ||c_i - c_j||) = V[i,j]
        # dm_ij/dc_j = (c_j - c_i) / (2 ||c_i - c_j||) = -V[i,j]
        
        # Contribution to grad_g from pair (i,j) is:
        # - w_p[i,j] * V[i,j] for c_i
        # - w_p[i,j] * (-V[i,j]) = + w_p[i,j] * V[i,j] for c_j
        
        # So for point k:
        # Sum over j != k: - w_p[k,j] V[k,j]  (from pairs where k is i)
        # Sum over i != k: + w_p[i,k] V[i,k]  (from pairs where k is j)
        
        # grad_g_pairs_k = - sum_j w_p[k,j] V[k,j] + sum_i w_p[i,k] V[i,k]
        # grad_g_pairs = - grad_out + grad_in
        
        grad_pairs = -grad_out + grad_in
        
        # Boundary contribution
        # dm_b_i/dc_i = -points[i]/safe_norms[i]
        # Contribution: - w_b[i] * (-points[i]/safe_norms[i]) = + w_b[i] * points[i]/safe_norms[i]
        grad_boundary = (w_b[:, np.newaxis] * (points / safe_norms[:, np.newaxis])).sum(axis=0)
        
        # Total gradient
        grad = grad_pairs + grad_boundary
        
        return obj, grad

    def ring_initialization():
        """Initialize points in concentric rings based on known packing structure."""
        points = []
        
        # Center
        points.append([0.0, 0.0])
        
        # Ring 1: 6 points
        r1 = 0.25
        for i in range(6):
            angle = 2 * np.pi * i / 6
            points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
            
        # Ring 2: 12 points
        r2 = 0.45
        for i in range(12):
            angle = 2 * np.pi * i / 12
            points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            
        # Ring 3: 18 points
        r3 = 0.65
        for i in range(18):
            angle = 2 * np.pi * i / 18
            points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
            
        # Ring 4: 13 points
        r4 = 0.85
        for i in range(13):
            angle = 2 * np.pi * i / 13
            points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
            
        pts = np.array(points)
        
        # Add small random jitter to break symmetry and help optimization
        jitter = np.random.normal(0, 0.01, pts.shape)
        pts += jitter
        
        # Ensure inside unit circle
        norms = np.linalg.norm(pts, axis=1)
        for i in range(len(pts)):
            if norms[i] > 0.99:
                pts[i] *= 0.99 / norms[i]
                
        return pts

    x0 = ring_initialization()
    
    best_centers = x0.copy()
    best_radius = compute_radius(x0)
    
    # Run L-BFGS-B optimization
    # We can try multiple betas or just one high beta
    beta = 100.0
    
    try:
        res = minimize(
            smooth_objective_and_grad, 
            x0, 
            args=(beta,), 
            jac=True, # Function returns (f, g)
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        x_opt = res.x.reshape(N, 2)
        
        # Ensure validity
        center_dists = np.sqrt(np.sum(x_opt**2, axis=1))
        if np.all(center_dists <= 1.0):
            # Clip if slightly outside due to numerical errors
            for i in range(N):
                if center_dists[i] > 1.0:
                    x_opt[i] *= 0.999 / center_dists[i]
                    
            current_radius = compute_radius(x_opt)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = x_opt
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # If time permits, try a second round with higher beta or different init
    if time.time() < deadline - 20:
        # Perturb best solution and optimize again
        x_perturbed = best_centers + np.random.normal(0, 0.01, best_centers.shape)
        norms = np.linalg.norm(x_perturbed, axis=1)
        for i in range(N):
            if norms[i] > 0.99:
                x_perturbed[i] *= 0.99 / norms[i]
                
        try:
            res2 = minimize(
                smooth_objective_and_grad, 
                x_perturbed, 
                args=(200.0,), # Higher beta for sharper minimum
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt2 = res2.x.reshape(N, 2)
            center_dists2 = np.sqrt(np.sum(x_opt2**2, axis=1))
            if np.all(center_dists2 <= 1.0):
                for i in range(N):
                    if center_dists2[i] > 1.0:
                        x_opt2[i] *= 0.999 / center_dists2[i]
                        
                current_radius2 = compute_radius(x_opt2)
                if current_radius2 > best_radius:
                    best_radius = current_radius2
                    best_centers = x_opt2
        except Exception as e:
            print(f"Second optimization failed: {e}")

    return best_centers
# EVOLVE-BLOCK-END
