<<<<<<< SEARCH
def _generate_heuristic(W, cap, mode, alpha=1.0):
    """
    Generate a heuristic matrix based on the specified mode.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= cap
    
    H = np.zeros((n, n))
    
    if mode == 'largest_fit':
        # Prefer larger items j
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    elif mode == 'tight_fit':
        # Prefer items j that make w_i + w_j close to capacity
        # H[i,j] = 1 / (capacity - (w_i + w_j))
        gap = cap - pair_sums
        # For fitting, gap >= 0.
        # Avoid division by zero
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
        
    elif mode == 'hybrid':
        # Combine largest fit and tight fit
        # H = w_j / (gap + 1)
        gap = cap - pair_sums
        H = np.where(fit_mask, W[None, :] / (gap + 1.0), 1e-9)
        
    elif mode == 'normalized_tight':
        gap = cap - pair_sums
        H_raw = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        # Normalize to mean 1 for fitting entries
        fitting_vals = H_raw[fit_mask]
        if len(fitting_vals) > 0:
            mean_val = np.mean(fitting_vals)
            if mean_val > 0:
                H = H_raw / mean_val
            else:
                H = H_raw
        else:
            H = H_raw
        H = np.where(H > 0, H, 1e-9)
        
    elif mode == 'blend':
        # Blend inverse-gap with fill-ratio^2
        # fill_ratio = (W[i] + W[j]) / cap
        # inv_gap = 1 / (cap - (W[i] + W[j]))
        # H = alpha * inv_gap + (1-alpha) * fill_ratio^2
        
        gap = cap - pair_sums
        inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        
        fill_ratio = np.where(fit_mask, pair_sums / cap, 0.0)
        fill_sq = fill_ratio ** 2
        
        # Normalize each component to mean 1 for fitting entries to ensure comparability
        inv_gap_fit = inv_gap[fit_mask]
        fill_sq_fit = fill_sq[fit_mask]
        
        if len(inv_gap_fit) > 0:
            mean_inv = np.mean(inv_gap_fit)
            if mean_inv > 0:
                inv_gap = inv_gap / mean_inv
            else:
                inv_gap = np.zeros_like(inv_gap)
                
            mean_fill = np.mean(fill_sq_fit)
            if mean_fill > 0:
                fill_sq = fill_sq / mean_fill
            else:
                fill_sq = np.zeros_like(fill_sq)
        
        H = alpha * inv_gap + (1 - alpha) * fill_sq
        H = np.where(fit_mask, H, 1e-9)
        
    else:
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
=======
def _generate_heuristic(W, cap, mode, alpha=1.0):
    """
    Generate a heuristic matrix based on the specified mode.
    """
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= cap
    
    H = np.zeros((n, n))
    
    if mode == 'largest_fit':
        # Prefer larger items j
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    elif mode == 'tight_fit':
        # Prefer items j that make w_i + w_j close to capacity
        # H[i,j] = 1 / (capacity - (w_i + w_j))
        gap = cap - pair_sums
        # For fitting, gap >= 0.
        # Avoid division by zero
        H = np.where(fit_mask, 1.0 / (gap + 1e-6), 1e-9)
        
    elif mode == 'hybrid':
        # Combine largest fit and tight fit
        # H = w_j / (gap + 1)
        gap = cap - pair_sums
        H = np.where(fit_mask, W[None, :] / (gap + 1.0), 1e-9)
        
    elif mode == 'normalized_tight':
        gap = cap - pair_sums
        H_raw = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        # Normalize to mean 1 for fitting entries
        fitting_vals = H_raw[fit_mask]
        if len(fitting_vals) > 0:
            mean_val = np.mean(fitting_vals)
            if mean_val > 0:
                H = H_raw / mean_val
            else:
                H = H_raw
        else:
            H = H_raw
        H = np.where(H > 0, H, 1e-9)
        
    elif mode == 'blend':
        # Blend inverse-gap with fill-ratio^2
        # fill_ratio = (W[i] + W[j]) / cap
        # inv_gap = 1 / (cap - (W[i] + W[j]))
        # H = alpha * inv_gap + (1-alpha) * fill_ratio^2
        
        gap = cap - pair_sums
        inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        
        fill_ratio = np.where(fit_mask, pair_sums / cap, 0.0)
        fill_sq = fill_ratio ** 2
        
        # Normalize each component to mean 1 for fitting entries to ensure comparability
        inv_gap_fit = inv_gap[fit_mask]
        fill_sq_fit = fill_sq[fit_mask]
        
        if len(inv_gap_fit) > 0:
            mean_inv = np.mean(inv_gap_fit)
            if mean_inv > 0:
                inv_gap = inv_gap / mean_inv
            else:
                inv_gap = np.zeros_like(inv_gap)
                
            mean_fill = np.mean(fill_sq_fit)
            if mean_fill > 0:
                fill_sq = fill_sq / mean_fill
            else:
                fill_sq = np.zeros_like(fill_sq)
        
        H = alpha * inv_gap + (1 - alpha) * fill_sq
        H = np.where(fit_mask, H, 1e-9)
        
    elif mode == 'complementarity':
        # Complementarity signal: rank items by scarcity of valid partners
        # degree[j] = number of items i such that w_i + w_j <= cap
        # Items with low degree are more constrained and should be placed earlier
        # H[i,j] = base_tight * (n / (degree[j] + 1))
        # This makes it more attractive to place constrained items j after any item i
        
        degree = fit_mask.sum(axis=0)  # degree[j] = count of i where fit_mask[i,j]
        # Inverse degree: higher for scarce partners
        inv_degree = n / (degree + 1.0)  # shape (n,)
        
        # Base tight-fit signal
        gap = cap - pair_sums
        inv_gap = np.where(fit_mask, 1.0 / (gap + 1e-6), 0.0)
        
        # Normalize inv_gap to mean 1 for fitting entries
        inv_gap_fit = inv_gap[fit_mask]
        if len(inv_gap_fit) > 0:
            mean_inv = np.mean(inv_gap_fit)
            if mean_inv > 0:
                inv_gap = inv_gap / mean_inv
            else:
                inv_gap = np.zeros_like(inv_gap)
        
        # Blend: alpha * tight_fit + (1-alpha) * complementarity
        # Complementarity component: inv_degree[j] broadcast to all rows i
        comp = inv_degree[None, :]  # shape (1, n)
        
        # Normalize complementarity to mean 1 for fitting entries
        comp_fit = comp[fit_mask]
        if len(comp_fit) > 0:
            mean_comp = np.mean(comp_fit)
            if mean_comp > 0:
                comp = comp / mean_comp
            else:
                comp = np.zeros_like(comp)
        
        H = alpha * inv_gap + (1 - alpha) * comp
        H = np.where(fit_mask, H, 1e-9)
        
    else:
        H = np.where(fit_mask, W[None, :], 1e-9)
        
    # Ensure positivity and finiteness
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
>>>>>>> REPLACE

