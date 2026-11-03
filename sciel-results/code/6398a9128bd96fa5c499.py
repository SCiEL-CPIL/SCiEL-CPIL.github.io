import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hex_init():
        """Generate hexagonal lattice initialization."""
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
            return np.random.randn(N, 2) * 0.5
        
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

    def soft_min_value(pts, k):
        """Compute soft-min of all constraints (pair distances and boundary)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        norms = np.linalg.norm(pts, axis=1)
        boundary_vals = 1.0 - norms
        
        all_vals = np.concatenate([pair_vals.ravel(), boundary_vals.ravel()])
        m = all_vals.min()
        return m - np.log(np.sum(np.exp(-k * (all_vals - m)))) / k

    def soft_min_grad(pts, k):
        """Compute gradient of soft-min objective."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        norms = np.linalg.norm(pts, axis=1)
        boundary_vals = 1.0 - norms
        
        all_vals = np.concatenate([pair_vals.ravel(), boundary_vals.ravel()])
        m = all_vals.min()
        e = np.exp(-k * (all_vals - m))
        s = e.sum()
        g = -k * e / s
        
        grad = np.zeros_like(pts)
        
        # Boundary part
        gb = g[:N]
        for i in range(N):
            if norms[i] > 1e-12:
                grad[i] += boundary_vals[i] * (-pts[i] / norms[i])
        
        # Pair part
        gp = g[N:].reshape(N, N)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if dists[i, j] > 1e-12:
                    grad[i] += gp[i, j] * 0.5 * (pts[i] - pts[j]) / dists[i, j]
        
        return grad

    def gradient_ascent(x0, time_limit):
        """Main optimization using gradient ascent on soft-min."""
        x = x0.copy()
        best_x = x.copy()
        best_r = calc_radius(x)
        
        t0 = time.time()
        k_soft = 10.0
        lr = 0.002
        stall_count = 0
        stall_threshold = 50
        
        while time.time() - t0 < time_limit:
            # Compute gradient
            grad = soft_min_grad(x, k_soft)
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break
            grad = grad / gnorm
            
            # Line search
            best_step = None
            best_val = soft_min_value(x, k_soft)
            for alpha in [lr, lr * 0.5, lr * 0.25, lr * 2.0]:
                x_new = x + alpha * grad
                # Keep points inside circle
                norms_new = np.linalg.norm(x_new, axis=1)
                mask = norms_new > 1.0
                if np.any(mask):
                    x_new[mask] = x_new[mask] / norms_new[mask] * 0.999
                
                val_new = soft_min_value(x_new, k_soft)
                if val_new > best_val:
                    best_val = val_new
                    best_step = x_new
            
            if best_step is None:
                lr *= 0.5
                stall_count += 1
                if lr < 1e-6 or stall_count > stall_threshold:
                    # Restart with higher k_soft
                    k_soft *= 1.5
                    lr = 0.001
                    stall_count = 0
                    if k_soft > 200:
                        break
            else:
                x = best_step
                stall_count = 0
                if stall_count % 10 == 0:
                    lr *= 1.05
            
            # Update best
            r_curr = calc_radius(x)
            if r_curr > best_r:
                best_r = r_curr
                best_x = x.copy()
        
        return best_x, best_r

    def kkt_refine(x0, r0, time_limit):
        """Active-set KKT refinement for local polish."""
        x = x0.copy()
        r = r0
        best_x = x.copy()
        best_r = r
        
        t0 = time.time()
        
        def build_contacts(pts, r_val, tol=1e-4):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            mask = dists <= 2 * r_val + tol
            mask = np.triu(mask, k=1)
            pairs = np.argwhere(mask)
            norms = np.linalg.norm(pts, axis=1)
            bnd_mask = norms >= 1.0 - r_val - tol
            return pairs, bnd_mask
        
        def solve_kkt(pts, pairs, bnd_mask, r_val, max_iter=100):
            for it in range(max_iter):
                if len(pairs) > 0:
                    diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
                    dists_pair = np.linalg.norm(diffs, axis=1)
                    viol_pair = dists_pair - 2 * r_val
                else:
                    viol_pair = np.array([])
                
                norms = np.linalg.norm(pts, axis=1)
                viol_bnd = norms - (1 - r_val)
                
                if len(pairs) > 0 and np.max(np.abs(viol_pair)) < 1e-8:
                    if np.any(bnd_mask) and np.max(np.abs(viol_bnd[bnd_mask])) < 1e-8:
                        break
                
                grad = np.zeros_like(pts)
                
                if len(pairs) > 0:
                    for k_idx, (i, j) in enumerate(pairs):
                        d = pts[i] - pts[j]
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            continue
                        u = d / dist
                        grad[i] += u
                        grad[j] -= u
                
                if np.any(bnd_mask):
                    for i in np.where(bnd_mask)[0]:
                        norm_i = np.linalg.norm(pts[i])
                        if norm_i < 1e-10:
                            continue
                        u = pts[i] / norm_i
                        grad[i] += u
                
                step = 0.01
                pts_new = pts - step * grad
                
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
        
        for _ in range(30):
            if time.time() - t0 > time_limit:
                break
            
            pairs, bnd_mask = build_contacts(x, r, tol=1e-4)
            
            r_trial = r * 1.001
            x_trial = solve_kkt(x.copy(), pairs, bnd_mask, r_trial, max_iter=50)
            
            r_check = calc_radius(x_trial)
            if r_check >= r_trial - 1e-8:
                x = x_trial
                r = r_trial
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
            else:
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
                    break
        
        return best_x, best_r

    def lbfgs_polish(x0, r0, time_limit):
        """Final high-precision L-BFGS polish."""
        x = x0.copy()
        best_x = x.copy()
        best_r = r0
        
        t0 = time.time()
        
        def objective(flat_x):
            pts = flat_x.reshape(N, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            norms = np.linalg.norm(pts, axis=1)
            boundary = 1.0 - norms
            
            all_dists = np.concatenate([dists[np.triu_indices(N, k=1)] / 2.0, boundary])
            
            beta = 100.0
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
            
            beta = 100.0
            all_dists = np.concatenate([dists[np.triu_indices(N, k=1)] / 2.0, boundary])
            scaled = -beta * all_dists
            max_scaled = np.max(scaled)
            exp_scaled = np.exp(scaled - max_scaled)
            sum_exp = np.sum(exp_scaled)
            weights = exp_scaled / sum_exp
            
            grad_flat = np.zeros(2 * N)
            
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            pair_dists = dists[mask]
            pair_weights = weights[:len(pair_dists)]
            
            W = np.zeros((N, N))
            W[mask] = pair_weights
            
            safe_dists = np.where(dists > 1e-8, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
            pair_grads = -beta / 2.0 * S
            
            bnd_weights = weights[len(pair_dists):]
            safe_norms = np.where(norms > 1e-8, norms, 1.0)
            bnd_unit = pts / safe_norms[:, np.newaxis]
            bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit
            
            total_grad = pair_grads + bnd_grads
            grad_flat = total_grad.flatten()
            
            return grad_flat / beta
        
        try:
            res = minimize(objective, x.flatten(), method='L-BFGS-B', jac=gradient,
                          options={'maxiter': 300, 'maxfun': 3000, 'disp': False})
            x_new = res.x.reshape(N, 2)
            r_new = calc_radius(x_new)
            if r_new > best_r:
                best_r = r_new
                best_x = x_new.copy()
        except Exception:
            pass
        
        return best_x, best_r

    # Main execution
    x0 = hex_init()
    r0 = calc_radius(x0)
    best_x = x0.copy()
    best_r = r0
    
    # Phase 1: Gradient ascent (main optimizer)
    ga_time = 80.0
    x_ga, r_ga = gradient_ascent(x0, ga_time)
    if r_ga > best_r:
        best_r = r_ga
        best_x = x_ga.copy()
    
    # Phase 2: KKT refinement
    kkt_time = 40.0
    x_kkt, r_kkt = kkt_refine(x_ga, r_ga, kkt_time)
    if r_kkt > best_r:
        best_r = r_kkt
        best_x = x_kkt.copy()
    
    # Phase 3: L-BFGS polish
    polish_time = 20.0
    x_polish, r_polish = lbfgs_polish(x_kkt, r_kkt, polish_time)
    if r_polish > best_r:
        best_r = r_polish
        best_x = x_polish.copy()
    
    # Final validation
    if np.any(~np.isfinite(best_x)):
        best_x = x0
        best_r = r0
    
    norms = np.linalg.norm(best_x, axis=1)
    if np.any(norms > 1.0):
        for i in range(N):
            if norms[i] > 1.0:
                best_x[i] = best_x[i] / norms[i] * 0.999
        best_r = calc_radius(best_x)
    
    return best_x
