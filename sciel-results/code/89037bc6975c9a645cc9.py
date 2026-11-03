import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hex_init():
        """Generate hexagonal lattice initialization."""
        # Estimate optimal spacing
        s = np.sqrt(2 * np.pi / (np.sqrt(3) * N)) * 0.95
        y_step = s * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s) + 2
        
        points = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s / 2
            for i in range(-max_i, max_i + 1):
                x = i * s + offset
                if np.hypot(x, y) < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback: random
            return np.random.randn(N, 2) * 0.5
        
        # Sort by distance to origin, take N closest
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        return points[idx[:N]].copy()

    def calc_radius(pts):
        """Compute the maximum radius for given centers."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    # Phase 1: Simulated Annealing for global exploration
    def sa_optimize(x0, time_limit):
        x = x0.copy()
        r = calc_radius(x)
        best_x = x.copy()
        best_r = r
        
        T = 0.02
        T_min = 1e-6
        cooling = 0.9997
        step_scale = 0.03
        
        t0 = time.time()
        iters = 0
        while time.time() - t0 < time_limit:
            # Random perturbation
            idx = np.random.randint(N)
            dx = np.random.randn(2) * step_scale * T
            x_new = x.copy()
            x_new[idx] += dx
            
            # Keep inside circle
            norm_new = np.linalg.norm(x_new[idx])
            if norm_new > 1.0:
                x_new[idx] = x_new[idx] / norm_new * 0.999
            
            r_new = calc_radius(x_new)
            
            # Accept if better or with probability exp(-(r - r_new)/T)
            if r_new > r or np.random.rand() < np.exp((r_new - r) / T):
                x = x_new
                r = r_new
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
            else:
                # Try a different random point
                pass
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            iters += 1
            # Periodically restart temperature if stuck
            if iters % 5000 == 0 and r < best_r * 0.99:
                T = max(T, 0.01)
        
        return best_x, best_r

    # Phase 2: Active-set KKT refinement
    def kkt_refine(x0, r0, time_limit):
        """
        Active-set method: identify contact graph (binding constraints)
        and solve the KKT system to find optimal positions for fixed r.
        """
        x = x0.copy()
        r = r0
        best_x = x.copy()
        best_r = r
        
        t0 = time.time()
        
        # Build contact graph: pairs with distance close to 2r
        def build_contacts(pts, r_val, tol=1e-4):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            # Contacts: dist <= 2*r_val + tol
            mask = dists <= 2 * r_val + tol
            mask = np.triu(mask, k=1)
            pairs = np.argwhere(mask)
            # Boundary contacts
            norms = np.linalg.norm(pts, axis=1)
            bnd_mask = norms >= 1.0 - r_val - tol
            return pairs, bnd_mask
        
        # Solve KKT system for given contact set
        def solve_kkt(pts, pairs, bnd_mask, r_val, max_iter=100):
            """
            Solve for positions given that contacts must be exactly 2r
            and boundary contacts must be exactly at distance 1-r.
            Uses iterative linearization.
            """
            n_free = N
            # Build constraint Jacobian
            # For each pair (i,j): ||x_i - x_j|| = 2r
            # For each boundary i: ||x_i|| = 1-r
            
            for it in range(max_iter):
                # Compute current violations
                if len(pairs) > 0:
                    diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
                    dists_pair = np.linalg.norm(diffs, axis=1)
                    viol_pair = dists_pair - 2 * r_val  # Should be 0
                else:
                    viol_pair = np.array([])
                
                norms = np.linalg.norm(pts, axis=1)
                viol_bnd = norms - (1 - r_val)  # Should be 0 for boundary contacts
                
                # Check convergence
                if len(pairs) > 0 and np.max(np.abs(viol_pair)) < 1e-8:
                    if np.any(bnd_mask) and np.max(np.abs(viol_bnd[bnd_mask])) < 1e-8:
                        break
                
                # Build linear system
                # For pair (i,j): (x_i - x_j)/d_ij · delta_i - (x_i - x_j)/d_ij · delta_j = -viol_pair
                # For boundary i: x_i/||x_i|| · delta_i = -viol_bnd
                
                # This is a large sparse system. Use simple iterative approach.
                # Instead, use gradient descent on the constraint violations.
                
                # Compute gradient of violation w.r.t positions
                grad = np.zeros_like(pts)
                
                if len(pairs) > 0:
                    for k, (i, j) in enumerate(pairs):
                        d = pts[i] - pts[j]
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            continue
                        u = d / dist
                        # d/dx_i (||x_i - x_j|| - 2r) = u
                        # d/dx_j (||x_i - x_j|| - 2r) = -u
                        grad[i] += u
                        grad[j] -= u
                
                if np.any(bnd_mask):
                    for i in np.where(bnd_mask)[0]:
                        norm_i = np.linalg.norm(pts[i])
                        if norm_i < 1e-10:
                            continue
                        u = pts[i] / norm_i
                        grad[i] += u
                
                # Gradient descent step
                step = 0.01
                pts_new = pts - step * grad
                
                # Check if improved
                new_viol = 0
                if len(pairs) > 0:
                    diffs_new = pts_new[pairs[:, 0]] - pts_new[pairs[:, 1]]
                    dists_new = np.linalg.norm(diffs_new, axis=1)
                    new_viol += np.sum(np.abs(dists_new - 2 * r_val))
                if np.any(bnd_mask):
                    norms_new = np.linalg.norm(pts_new, axis=1)
                    new_viol += np.sum(np.abs(norms_new[bnd_mask] - (1 - r_val)))
                
                old_viol = 0
                if len(pairs) > 0:
                    old_viol += np.sum(np.abs(viol_pair))
                if np.any(bnd_mask):
                    old_viol += np.sum(np.abs(viol_bnd[bnd_mask]))
                
                if new_viol < old_viol:
                    pts = pts_new
                else:
                    step *= 0.5
                    if step < 1e-8:
                        break
            
            return pts
        
        # Main KKT loop
        for _ in range(50):  # Max outer iterations
            if time.time() - t0 > time_limit:
                break
            
            # Build contact set
            pairs, bnd_mask = build_contacts(x, r, tol=1e-4)
            
            # Try to increase r slightly
            r_trial = r * 1.001
            
            # Check if feasible with current contacts
            x_trial = solve_kkt(x.copy(), pairs, bnd_mask, r_trial, max_iter=50)
            
            # Check if the trial solution is valid
            r_check = calc_radius(x_trial)
            if r_check >= r_trial - 1e-8:
                x = x_trial
                r = r_trial
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
            else:
                # Try smaller increase
                r_trial = r * 1.0001
                x_trial = solve_kkt(x.copy(), pairs, bnd_mask, r_trial, max_iter=50)
                r_check = calc_radius(x_trial)
                if r_check >= r_trial - 1e-8:
                    x = x_trial
                    r = r_trial
                    if r > best_r:
                        best_r = r
                        best_x = x.copy()
                else:
                    # No progress, break
                    break
        
        return best_x, best_r

    # Phase 3: L-BFGS-B polish with soft objective
    def lbfgs_polish(x0, r0, time_limit):
        """Polish using L-BFGS-B on a smooth objective."""
        x = x0.copy()
        best_x = x.copy()
        best_r = r0
        
        t0 = time.time()
        
        def objective(flat_x):
            pts = flat_x.reshape(N, 2)
            # Soft min objective
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            norms = np.linalg.norm(pts, axis=1)
            boundary = 1.0 - norms
            
            # Combine all distances
            all_dists = np.concatenate([dists[np.triu_indices(N, k=1)] / 2.0, boundary])
            
            # Soft min: -log-sum-exp(-beta * dists)
            beta = 50.0
            scaled = -beta * all_dists
            max_scaled = np.max(scaled)
            log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            return log_sum_exp / beta
        
        def gradient(flat_x):
            pts = flat_x.reshape(N, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            norms = np.linalg.norm(pts, axis=1)
            boundary = 1.0 - norms
            
            beta = 50.0
            all_dists = np.concatenate([dists[np.triu_indices(N, k=1)] / 2.0, boundary])
            scaled = -beta * all_dists
            max_scaled = np.max(scaled)
            exp_scaled = np.exp(scaled - max_scaled)
            sum_exp = np.sum(exp_scaled)
            weights = exp_scaled / sum_exp
            
            grad_flat = np.zeros(2 * N)
            
            # Pairwise gradients
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            pair_dists = dists[mask]
            pair_weights = weights[:len(pair_dists)]
            
            W = np.zeros((N, N))
            W[mask] = pair_weights
            
            safe_dists = np.where(dists > 1e-8, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
            pair_grads = -beta / 2.0 * S
            
            # Boundary gradients
            bnd_weights = weights[len(pair_dists):]
            safe_norms = np.where(norms > 1e-8, norms, 1.0)
            bnd_unit = pts / safe_norms[:, np.newaxis]
            bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit
            
            total_grad = pair_grads + bnd_grads
            grad_flat = total_grad.flatten()
            
            return grad_flat / beta
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, x.flatten(), method='L-BFGS-B', jac=gradient,
                          options={'maxiter': 500, 'maxfun': 5000, 'disp': False})
            x_new = res.x.reshape(N, 2)
            r_new = calc_radius(x_new)
            if r_new > best_r:
                best_r = r_new
                best_x = x_new.copy()
        except Exception:
            pass
        
        return best_x, best_r

    # Main execution
    # Phase 1: SA
    x0 = hex_init()
    r0 = calc_radius(x0)
    best_x = x0.copy()
    best_r = r0
    
    sa_time = 60.0
    x_sa, r_sa = sa_optimize(x0, sa_time)
    if r_sa > best_r:
        best_r = r_sa
        best_x = x_sa.copy()
    
    # Phase 2: KKT refinement
    kkt_time = 40.0
    x_kkt, r_kkt = kkt_refine(x_sa, r_sa, kkt_time)
    if r_kkt > best_r:
        best_r = r_kkt
        best_x = x_kkt.copy()
    
    # Phase 3: L-BFGS polish
    polish_time = 30.0
    x_polish, r_polish = lbfgs_polish(x_kkt, r_kkt, polish_time)
    if r_polish > best_r:
        best_r = r_polish
        best_x = x_polish.copy()
    
    # Final validation
    if np.any(~np.isfinite(best_x)):
        best_x = x0
        best_r = r0
    
    # Ensure all points are inside
    norms = np.linalg.norm(best_x, axis=1)
    if np.any(norms > 1.0):
        # Project back
        for i in range(N):
            if norms[i] > 1.0:
                best_x[i] = best_x[i] / norms[i] * 0.999
        best_r = calc_radius(best_x)
    
    return best_x
