import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

def _hex_init(n_points, seed=42):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Try a few different spacings to get a good initial packing
    for attempt in range(5):
        # Approximate spacing for hex packing
        # Area per point ~ sqrt(3)/2 * s^2. Total area ~ pi.
        # N * sqrt(3)/2 * s^2 ~ pi  => s ~ sqrt(2*pi / (N*sqrt(3)))
        base_s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
        spacing = base_s * (0.95 + 0.02 * attempt)
        
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        
        if len(rows) < n_points:
            continue
            
        pts = np.array(rows)
        # Select n_points closest to origin
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists, kind="stable")
        selected = pts[order[:n_points]]
        
        # Add small random noise to break symmetry
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.01, selected.shape)
            # Project back inside if necessary
            norms = np.linalg.norm(selected, axis=1)
            mask = norms > 0.99
            selected[mask] *= 0.99 / norms[mask, np.newaxis]
            
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        # Fallback: random
        best_pts = rng.random((n_points, 2)) * 1.5 - 0.75
        best_pts *= 0.9 / np.linalg.norm(best_pts, axis=1, keepdims=True)
        
    return best_pts

def _compute_radius(pts):
    """Compute the maximum valid radius for the given centers."""
    n = len(pts)
    if n == 0:
        return 0.0
        
    # Boundary constraint
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
        
    # Pairwise constraint
    # Efficient distance calculation
    sq_norms = np.sum(pts * pts, axis=1)
    cross = pts @ pts.T
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
    np.maximum(sq_dists, 0.0, out=sq_dists)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair = np.sqrt(sq_dists.min()) / 2.0
    
    return min(min_pair, min_bound)

def _local_refine(pts, time_limit=10.0):
    """
    Refine the packing using an active set approach.
    Iteratively identifies points involved in minimum constraints and optimizes them.
    """
    pts = pts.copy()
    start_time = time.time()
    current_r = _compute_radius(pts)
    
    # Tolerance for "active" constraint
    tol = 1e-4
    
    while time.time() - start_time < time_limit:
        r = _compute_radius(pts)
        if r <= current_r + 1e-10:
            # No improvement, maybe try a kick or break
            # To avoid infinite loop on no improvement, we can break or try a random perturbation
            # For simplicity, we break if no improvement in this specific iteration
            break
        
        current_r = r
        
        # Identify active points
        # 1. Boundary active: 1 - ||c_i|| < r + tol
        norms = np.linalg.norm(pts, axis=1)
        boundary_active = np.where(1.0 - norms < r + tol)[0]
        
        # 2. Pairwise active: d_ij/2 < r + tol
        sq_norms = np.sum(pts * pts, axis=1)
        cross = pts @ pts.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
        np.maximum(sq_dists, 0.0, out=sq_dists)
        
        # Find pairs with dist < 2*(r+tol)
        threshold = 2.0 * (r + tol)
        # We only need upper triangle
        iu = np.triu_indices(N, k=1)
        active_pairs = np.where(sq_dists[iu] < threshold * threshold)[0]
        
        active_indices = set(boundary_active.tolist())
        for idx in active_pairs:
            i, j = iu[0][idx], iu[1][idx]
            active_indices.add(i)
            active_indices.add(j)
            
        active_indices = list(active_indices)
        
        if not active_indices:
            break
            
        # If active set is too large, restrict to the most critical ones
        # Sort by how close they are to the limit
        if len(active_indices) > 50:
            # Calculate "slack" for each active point
            slacks = []
            for i in active_indices:
                # Min slack to boundary
                slack_b = 1.0 - np.linalg.norm(pts[i])
                # Min slack to neighbors
                # We can approximate by looking at the min dist to all other points
                dists_to_i = np.sqrt(np.maximum(sq_dists[i], 0.0))
                np.fill_diagonal(dists_to_i, np.inf)
                slack_p = dists_to_i.min() / 2.0
                slacks.append(min(slack_b, slack_p))
                
            # Select the 50 most critical (smallest slack)
            slacks = np.array(slacks)
            worst_50 = np.argsort(slacks)[:50]
            active_indices = [active_indices[i] for i in worst_50]
            
        # Define local objective
        # Minimize -softmin of constraints for active points
        # Constraints:
        # 1. Dist between active points / 2
        # 2. Dist from active to fixed points / 2
        # 3. 1 - ||active||
        
        fixed_indices = np.setdiff1d(np.arange(N), np.array(active_indices))
        
        def local_obj_grad(x):
            # x is flattened active points
            pts_a = x.reshape(-1, 2)
            
            # Constraints
            cons = []
            grads = []
            
            # 1. Active-Active pairs
            n_a = len(pts_a)
            if n_a > 1:
                sq_n_a = np.sum(pts_a * pts_a, axis=1)
                cross_a = pts_a @ pts_a.T
                sq_d_a = sq_n_a[:, np.newaxis] + sq_n_a[np.newaxis, :] - 2.0 * cross_a
                np.maximum(sq_d_a, 0.0, out=sq_d_a)
                np.fill_diagonal(sq_d_a, np.inf)
                d_a = np.sqrt(sq_d_a)
                iu_a = np.triu_indices(n_a, k=1)
                d_pairs = d_a[iu_a] / 2.0
                cons.append(d_pairs)
                
                # Gradient for these
                # d(d_ij/2)/d(x_i) = 0.5 * (x_i - x_j) / d_ij
                # We'll compute gradient via finite difference or analytical
                # Analytical is better.
                # Let's accumulate gradient for x
                grad_x = np.zeros_like(x)
                
                # We need weights for softmin later, but for local refine,
                # we can just use the min constraint directly with a soft penalty?
                # No, let's stick to softmin for smoothness.
                
            # This analytical gradient derivation for the full softmin on active set is complex.
            # Alternative: Use a simpler objective for local refine:
            # Maximize the minimum of the constraints.
            # We can use a "barrier" or "penalty" method.
            # Let's use a simple penalty: sum of (r_target - constraint)^2 for violated constraints.
            # But we don't know r_target.
            
            # Let's go back to softmin but implement it carefully for the local set.
            
            all_vals = []
            
            # Re-evaluate all constraints involving active points
            
            # 1. Active-Active
            if n_a > 1:
                # Reuse d_pairs from above if we kept it, but let's recalc for clarity in this block
                # Actually, let's just compute the softmin value and gradient numerically? No, too slow.
                
                # Let's define the objective as -softmin(constraints)
                # Constraints:
                # C1: d(i,j)/2 for i,j in active, i!=j
                # C2: d(i,k)/2 for i in active, k in fixed
                # C3: 1 - ||i|| for i in active
                
                vals = []
                # C1
                if n_a > 1:
                    vals.append(d_pairs)
                    
                # C2
                if len(fixed_indices) > 0:
                    pts_f = pts[fixed_indices]
                    # Distances between active and fixed
                    # d(i, k)
                    # sq_d_af = ||a_i||^2 + ||f_k||^2 - 2 a_i . f_k
                    sq_n_f = np.sum(pts_f * pts_f, axis=1)
                    cross_af = pts_a @ pts_f.T
                    sq_d_af = sq_n_a[:, np.newaxis] + sq_n_f[np.newaxis, :] - 2.0 * cross_af
                    np.maximum(sq_d_af, 0.0, out=sq_d_af)
                    d_af = np.sqrt(sq_d_af) / 2.0
                    vals.append(d_af.flatten())
                    
                # C3
                norms_a = np.linalg.norm(pts_a, axis=1)
                b_c = 1.0 - norms_a
                vals.append(b_c)
                
                all_vals = np.concatenate(vals)
                
                # Softmin
                T = 0.001 # Fixed temperature for local refine
                neg_vals = -all_vals / T
                max_neg = np.max(neg_vals)
                if max_neg > 700:
                    # Degenerate
                    min_idx = np.argmin(all_vals)
                    obj = -all_vals.min()
                    # Gradient is one-hot, hard to backprop easily without structure
                    # Let's use a small T to avoid this
                    T = 0.01
                    neg_vals = -all_vals / T
                    max_neg = np.max(neg_vals)
                    
                exp_shifted = np.exp(neg_vals - max_neg)
                weights = exp_shifted / np.sum(exp_shifted)
                soft_min_val = -T * (np.log(np.sum(exp_shifted)) + max_neg)
                obj = -soft_min_val
                
                # Gradient
                # d(obj)/d(x) = - sum_k weights[k] * d(constraint_k)/d(x)
                
                grad_x = np.zeros_like(x)
                
                # Backprop weights to constraints
                # C1: d_pairs
                if n_a > 1:
                    w1 = weights[:len(d_pairs)]
                    # d(d_ij/2)/d(x_i) = 0.5 * u_ij
                    # Build matrix W
                    W = np.zeros((n_a, n_a))
                    W[iu_a] = w1
                    W[iu_a[1], iu_a[0]] = w1
                    
                    # unit vectors
                    safe_d = np.where(d_a > 1e-10, d_a, 1.0)
                    unit_vecs = (pts_a[:, np.newaxis, :] - pts_a[np.newaxis, :, :]) / safe_d[:, :, np.newaxis]
                    
                    grad_pairs = 0.5 * np.einsum('ij,ijc->ic', W, unit_vecs)
                    grad_x += -grad_pairs.ravel()
                    
                # C2: d_af
                if len(fixed_indices) > 0:
                    w2 = weights[len(d_pairs) : len(d_pairs) + d_af.size] if n_a > 1 else weights[:d_af.size]
                    # d(d_ik/2)/d(x_i) = 0.5 * (x_i - x_k) / d_ik
                    safe_d_af = np.where(d_af * 2.0 > 1e-10, d_af * 2.0, 1.0)
                    # d_af is (n_a, n_f)
                    # We need (x_i - x_k) / d_ik
                    diff_af = pts_a[:, np.newaxis, :] - pts_f[np.newaxis, :, :]
                    unit_af = diff_af / safe_d_af[:, :, np.newaxis]
                    
                    # grad_x_i = - sum_k w_ik * 0.5 * unit_ik
                    # w2 is flattened (n_a * n_f)
                    w2_mat = w2.reshape(n_a, -1)
                    grad_af = 0.5 * np.einsum('ij,ijk->ik', w2_mat, unit_af)
                    grad_x += -grad_af.ravel()
                    
                # C3: b_c
                w3 = weights[-n_a:]
                # d(1-||x||)/d(x) = -x/||x||
                safe_norms_a = np.where(norms_a > 1e-10, norms_a, 1.0)
                grad_b = -pts_a / safe_norms_a[:, np.newaxis]
                grad_x += -(w3[:, np.newaxis] * grad_b).ravel()
                
                return obj, grad_x
            
            # Fallback if no constraints (should not happen)
            return 0.0, np.zeros_like(x)

        x0 = pts[active_indices].ravel()
        
        try:
            res = minimize(
                lambda x: local_obj_grad(x)[0],
                x0,
                jac=lambda x: local_obj_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_pts_a = res.x.reshape(-1, 2)
            
            # Check validity
            if np.all(np.isfinite(new_pts_a)):
                # Update global pts
                pts[active_indices] = new_pts_a
                
                # Verify improvement
                new_r = _compute_radius(pts)
                if new_r > current_r:
                    current_r = new_r
                else:
                    # Revert if it didn't improve (or got worse)
                    # Actually, L-BFGS should minimize, so it should improve the soft-min.
                    # But soft-min improvement doesn't guarantee hard-min improvement.
                    # If it didn't improve, we might be stuck.
                    # Let's try a small random kick on the active set
                    pts[active_indices] = x0.reshape(-1, 2) + np.random.normal(0, 0.001, size=x0.shape).reshape(-1, 2)
                    
        except Exception:
            pass
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = _hex_init(N, seed=42)
    best_r = _compute_radius(best_pts)
    
    # Main loop: Local refinement with kicks
    # We will alternate between local refinement and random restarts/kicks if stuck
    
    stuck_count = 0
    max_stuck = 5
    
    while time.time() < deadline - 10.0:
        # Refine
        refined_pts = _local_refine(best_pts, time_limit=10.0)
        new_r = _compute_radius(refined_pts)
        
        if new_r > best_r + 1e-10:
            best_pts = refined_pts
            best_r = new_r
            stuck_count = 0
        else:
            stuck_count += 1
            
            if stuck_count >= max_stuck:
                # Try a kick
                # Pick a random subset of points and perturb them
                n_kick = 20
                kick_indices = np.random.choice(N, n_kick, replace=False)
                best_pts[kick_indices] += np.random.normal(0, 0.01, size=(n_kick, 2))
                
                # Ensure inside
                norms = np.linalg.norm(best_pts[kick_indices], axis=1)
                mask = norms > 0.99
                best_pts[kick_indices][mask] *= 0.99 / norms[mask, np.newaxis]
                
                stuck_count = 0
                
                # If stuck for too long, try a full restart with different seed?
                # No, keep the best found so far.
                if stuck_count > 10:
                     # Try a different initialization?
                     # Let's just continue with kicks.
                     pass

    # Final polish
    if time.time() < deadline - 2.0:
        final_pts = _local_refine(best_pts, time_limit=5.0)
        final_r = _compute_radius(final_pts)
        if final_r > best_r:
            best_pts = final_pts
            best_r = final_r
            
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms >= 1.0):
        # Project back
        mask = norms >= 1.0
        best_pts[mask] *= 0.999 / norms[mask, np.newaxis]
        
    return best_pts
