    # Final Polish with SciPy L-BFGS-B and Soft-Min Objective with Analytic Gradient
    if time.time() < deadline - 5.0:
        def soft_min_obj_grad(flat_points, beta):
            pts = flat_points.reshape(-1, 2)
            n = len(pts)
            
            # Pairwise
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            
            # Avoid division by zero
            dist_safe = np.where(dist > 1e-8, dist, 1.0)
            unit_vecs = diff / dist_safe[:, :, np.newaxis]
            
            pair_dists = dist[mask] / 2.0
            pair_grads = unit_vecs[mask] # Gradient of d_ij w.r.t c_i is (c_i-c_j)/d_ij. 
                                        # Wait, the mask flattens the matrix.
                                        # We need to map back to gradients for each point.
            
            # It's easier to compute the gradient for the full matrix and then sum.
            # Let's compute the weights for all pairs and boundaries.
            
            # Boundary
            radii = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - radii
            bound_unit_vecs = -pts / np.maximum(radii, 1e-8)[:, np.newaxis]
            
            all_dists = np.concatenate([pair_dists, bound_dists])
            
            # Weights
            max_d = np.max(all_dists)
            exp_terms = np.exp(-beta * (all_dists - max_d))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Gradient of f = - sum(w_k * grad_d_k)
            # grad_f = - sum(w_k * grad_d_k)
            
            # We need to accumulate gradients for each point.
            grad = np.zeros_like(pts)
            
            # Pairwise gradients
            # The mask has shape (n, n) with diagonal False.
            # weights[:n*(n-1)] correspond to the flattened pair_dists.
            # We need to unflatten the weights to (n, n) to multiply with unit_vecs.
            
            w_pairs = np.zeros((n, n))
            # Fill the upper triangle? No, the mask is symmetric.
            # Let's just use the full matrix for simplicity, though it's 2x work.
            # Or better, iterate? No, vectorize.
            
            # Let's reconstruct the (n, n) weight matrix for pairs.
            # pair_dists is dist[mask] / 2.
            # We can compute the full weight matrix for pairs.
            
            # Recalculate for full matrix to be safe and simple, as N=300 is small.
            dist_full = dist
            dist_full_safe = np.where(dist_full > 1e-8, dist_full, 1.0)
            unit_vecs_full = diff / dist_full_safe[:, :, np.newaxis]
            
            # Pair dists are dist_full / 2
            pair_dists_full = dist_full / 2.0
            # Set diagonal to large value so weight is 0
            pair_dists_full_flat = pair_dists_full.flatten()
            # We already have all_dists which includes boundaries.
            # Let's just compute the gradient using the full matrix logic.
            
            # Objective: f = - softmin(D) where D is set of all distances.
            # grad f = - sum_{k} w_k grad d_k
            
            # We have weights for `all_dists`.
            # `all_dists` = [pair_dists (flat, no diag), bound_dists]
            
            # Let's compute the gradient contribution from pairs.
            # We need the weights corresponding to the off-diagonal elements of the dist matrix.
            
            # Create a mask for off-diagonal
            off_diag_mask = ~np.eye(n, dtype=bool)
            
            # We have `weights` for the concatenated array.
            # The first n*(n-1) elements are for pairs? 
            # No, `pair_dists = dist[mask]` where mask is off-diagonal.
            # So yes, first n*(n-1) are pairs.
            
            w_pairs_flat = weights[:n*(n-1)]
            w_bounds = weights[n*(n-1):]
            
            # Reshape w_pairs_flat to (n, n)
            w_pairs_matrix = np.zeros((n, n))
            # We need to fill the off-diagonal with w_pairs_flat.
            # The order of `dist[mask]` is row-major.
            # So w_pairs_matrix[off_diag_mask] = w_pairs_flat
            
            w_pairs_matrix[off_diag_mask] = w_pairs_flat
            
            # Gradient of d_ij w.r.t c_i is (c_i - c_j) / d_ij.
            # Gradient of d_ij w.r.t c_j is (c_j - c_i) / d_ij = - (c_i - c_j) / d_ij.
            
            # Sum over j for each i:
            # grad_i = - sum_j w_ij * (c_i - c_j) / d_ij
            
            # We have unit_vecs_full where unit_vecs[i,j] = (c_i - c_j) / d_ij.
            
            # grad_pairs = - sum_j (w_pairs_matrix[i,j] * unit_vecs_full[i,j])
            
            grad_pairs = -np.sum(w_pairs_matrix[:, :, np.newaxis] * unit_vecs_full, axis=1)
            
            # Gradient for boundaries
            # d_i = 1 - |c_i|
            # grad d_i w.r.t c_i = - c_i / |c_i|
            # We have bound_unit_vecs = -pts / radii.
            
            grad_bounds = -np.sum(w_bounds[:, np.newaxis] * bound_unit_vecs, axis=0) # Wait, sum is over what?
            # w_bounds is shape (N,). bound_unit_vecs is (N, 2).
            # grad_bounds[i] = - w_bounds[i] * bound_unit_vecs[i]
            grad_bounds = -(w_bounds[:, np.newaxis] * bound_unit_vecs)
            
            total_grad = grad_pairs + grad_bounds
            
            # f = - softmin
            # grad f = - sum w_k grad d_k
            # We computed grad_pairs as - sum ... which is the gradient of the softmin part?
            # Let's re-verify signs.
            # f = - softmin = 1/beta log(sum exp(-beta d))
            # grad f = - sum w_k grad d_k
            # My `grad_pairs` calculation: `grad_pairs = -np.sum(...)`
            # This is `- sum w_k grad d_k` for the pairs.
            # So `total_grad` is the gradient of f.
            
            return -soft_min_val, total_grad # Wait, objective returns f.
            # f = - soft_min_val
            # So return f, grad_f
            
        # ... (Rest of the polish loop using obj_grad)
