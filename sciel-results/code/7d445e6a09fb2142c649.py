import numpy as np
import time
from scipy.spatial import Delaunay
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score efficiently."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)
    
    def get_weights_and_grads(pts, k, current_min):
        """
        Compute soft-min weights and gradients stably.
        Returns: weights (for pairs and bounds), grads
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx])
        pair_vals = d_pair / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        shifted = all_vals - current_min
        # Stable weight computation: only consider constraints within a certain distance
        max_shift = 20.0
        mask_active = shifted < max_shift
        
        weights = np.zeros_like(all_vals)
        if np.any(mask_active):
            active_vals = shifted[mask_active]
            active_weights = np.exp(-k * active_vals)
            sum_weights = np.sum(active_weights)
            if sum_weights > 0:
                weights[mask_active] = active_weights / sum_weights
            else:
                weights[mask_active] = 1.0 / np.sum(mask_active)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            vecs = diff[i_idx[idx_m], j_idx[idx_m]] / d_pair[idx_m, np.newaxis]
            vecs *= pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads

    def spectral_regularization_grad(pts, weight=0.0):
        """
        Computes gradient of algebraic connectivity (Fiedler value) of the Delaunay graph.
        Maximizing this promotes global connectivity and structural rigidity.
        """
        if weight <= 0:
            return np.zeros_like(pts)
            
        try:
            tri = Delaunay(pts)
            edges = tri.edges
            if len(edges) < 3:
                return np.zeros_like(pts)
            
            # Build Laplacian
            n_edges = len(edges)
            # Weights: 1/length^2 to emphasize short connections? Or just 1?
            # Let's use 1/length^2 to make it a proper electrical network Laplacian
            # Actually, for algebraic connectivity, standard unweighted is fine, 
            # but weighted by 1/d^2 might be more stable for packing.
            # Let's stick to simple unweighted first, or 1/d.
            p1 = pts[edges[:, 0]]
            p2 = pts[edges[:, 1]]
            diffs = p1 - p2
            lengths = np.linalg.norm(diffs, axis=1)
            lengths_safe = np.where(lengths > 1e-9, lengths, 1.0)
            w = 1.0 / lengths_safe
            
            # Build sparse Laplacian
            row = np.concatenate([edges[:, 0], edges[:, 1]])
            col = np.concatenate([edges[:, 1], edges[:, 0]])
            data = np.concatenate([-w, -w])
            L_off = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
            
            # Diagonal
            deg = np.zeros(N)
            np.add.at(deg, edges[:, 0], w)
            np.add.at(deg, edges[:, 1], w)
            L_diag = sp.diags(deg)
            L = (L_diag + L_off).tocsr()
            
            # Compute Fiedler vector and eigenvalue
            try:
                # k=2 to get the smallest non-zero eigenvalue
                # Need to handle the zero eigenvalue (constant vector)
                # eigsh might struggle if the graph is not connected, but Delaunay is.
                # We want the 2nd smallest eigenvalue.
                # sigma=0, k=2, which='LM' won't work directly for smallest.
                # Use sigma=0 and which='LM' on shifted? No.
                # Use which='SA' (smallest algebraic) or 'SM'.
                # For sparse, 'SA' is slow. 'LM' with shift-invert sigma=0 is better.
                try:
                    eigvals, eigvecs = eigsh(L, k=2, sigma=0, which='LM', v0=np.ones(N))
                except Exception:
                    eigvals, eigvecs = eigsh(L, k=2, which='SA', v0=np.ones(N))
                
                # Find the non-zero eigenvalue
                # Sort by value
                idx_sort = np.argsort(eigvals)
                eigvals = eigvals[idx_sort]
                eigvecs = eigvecs[:, idx_sort]
                
                # The smallest should be ~0. The next is Fiedler.
                if len(eigvals) < 2:
                    return np.zeros_like(pts)
                
                lambda_2 = eigvals[1]
                fiedler_vec = eigvecs[:, 1]
                
                # Gradient of lambda_2 wrt L is v v^T
                # L = sum_e w_e (e_i e_i^T + e_j e_j^T - e_i e_j^T - e_j e_i^T)
                # dL/dw_e = ...
                # dlambda/dw_e = v_i^2 + v_j^2 - 2 v_i v_j = (v_i - v_j)^2
                # dlambda/dp = sum_e (dlambda/dw_e) * (dw_e/dp)
                # w_e = 1/|p_i - p_j|
                # dw_e/dp_i = -1/d^2 * (p_i - p_j)/d = -(p_i - p_j)/d^3
                # dw_e/dp_j = (p_i - p_j)/d^3
                
                # So grad_p = sum_e (v_i - v_j)^2 * dw_e/dp
                # Let delta_v = v_i - v_j
                # term_e = delta_v^2 * (-(p_i - p_j)/d^3) for p_i
                # term_e = delta_v^2 * ((p_i - p_j)/d^3) for p_j
                
                v1 = fiedler_vec[edges[:, 0]]
                v2 = fiedler_vec[edges[:, 1]]
                dv = v1 - v2
                coeff = (dv**2) / (lengths_safe**3)
                
                # Force on p_i is -coeff * diff
                # Force on p_j is +coeff * diff
                forces = coeff[:, np.newaxis] * diffs
                
                grad = np.zeros_like(pts)
                np.add.at(grad, edges[:, 0], -forces)
                np.add.at(grad, edges[:, 1], forces)
                
                # We want to MAXIMIZE lambda_2, so we ascend.
                # The gradient computed is d(lambda)/d(p).
                # So we add weight * grad.
                return weight * grad
                
            except Exception:
                return np.zeros_like(pts)
                
        except Exception:
            return np.zeros_like(pts)

    def optimize_soft_min(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                         stall_threshold=150, noise_scale_base=0.01, polish=False,
                         spectral_weight=0.0):
        """
        Main optimization loop using soft-min gradient ascent.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        iter_count = 0
        end_time = start_time + time_budget
        
        if polish:
            stall_threshold = 300
            cooling = 0.9995
        
        while time.time() < end_time:
            # Compute current min
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute gradient using stable weights
            grads = get_weights_and_grads(pts, k, current_min)
            
            # Add spectral regularization
            if spectral_weight > 0:
                spec_grad = spectral_regularization_grad(pts, spectral_weight)
                grads += spec_grad
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Adaptive step size
            if polish:
                step_alpha = alpha * 0.5
            else:
                step_alpha = alpha
            
            # Update
            pts += step_alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                k = k_init + (k_end - k_init) * (progress ** 2.0)
            else:
                k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                if polish:
                    noise_scale *= 0.3
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005 if not polish else 0.002)
            
            iter_count += 1
        
        return best_pts, best_score
    
    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts
    
    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)
        
    def polish_direct(pts, time_budget):
        """
        High-precision polish using direct gradient ascent with very high k.
        """
        current_pts = pts.copy()
        current_best = compute_score(pts)
        end_time = start_time + time_budget
        
        # Use high k for near-exact min
        k_final = 1000.0
        alpha = 0.001
        cooling = 0.9999
        
        while time.time() < end_time:
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(current_pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > current_best:
                current_best = current_min
            
            grads = get_weights_and_grads(current_pts, k_final, current_min)
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-12:
                grads /= grad_norm
                current_pts += alpha * grads
                
                rad_new = np.linalg.norm(current_pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    current_pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            alpha *= cooling
            if alpha < 1e-6:
                break
                
        return current_pts, current_best

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(15.0, time_limit * 0.18)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.65:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
            # Use spectral regularization for the main phase to enforce uniformity
            spec_w = 0.05 if idx < 3 else 0.02
        else:
            time_budget = time_per_init * 0.6
            spec_w = 0.0
        
        pts_opt, score = optimize_soft_min(pts, 
                                           alpha_init=0.012, 
                                           k_init=5.0, 
                                           k_end=200.0, 
                                           cooling=0.9997,
                                           time_budget=time_budget,
                                           stall_threshold=120,
                                           noise_scale_base=0.006,
                                           spectral_weight=spec_w)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.75:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.75:
                break
            
            pts = random_init()
            time_budget = min(12.0, time_limit * 0.12)
            pts_opt, score = optimize_soft_min(pts,
                                               alpha_init=0.008,
                                               k_init=4.0,
                                               k_end=150.0,
                                               cooling=0.9996,
                                               time_budget=time_budget,
                                               stall_threshold=100,
                                               noise_scale_base=0.008,
                                               spectral_weight=0.0)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish the best result
    if best_pts is not None and time.time() - start_time < time_limit * 0.85:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize_soft_min(best_pts,
                                                             alpha_init=0.005,
                                                             k_init=100.0,
                                                             k_end=300.0,
                                                             cooling=0.9995,
                                                             time_budget=min(remaining_time * 0.25, 6.0),
                                                             stall_threshold=150,
                                                             noise_scale_base=0.003,
                                                             polish=True,
                                                             spectral_weight=0.0)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: High-precision direct polish
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 3.0:
                pts_direct, score_direct = polish_direct(best_pts, min(remaining_time2 * 0.5, 10.0))
                
                if score_direct > best_score:
                    best_score = score_direct
                    best_pts = pts_direct

    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    # Final high-precision polish if time allows
    final_remaining = time_limit * 0.98 - time.time()
    if final_remaining > 2.0:
        pts_final, score_final = polish_direct(best_pts, final_remaining)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
