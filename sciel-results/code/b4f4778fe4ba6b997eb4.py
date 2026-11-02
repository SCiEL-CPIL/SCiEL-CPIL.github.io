"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Precompute all triangle index triples
    tri_indices = np.array(list(itertools.combinations(range(N), 3)))
    n_tris = len(tri_indices)
    
    def bary_to_cartesian(bary):
        """Convert barycentric coords (l1, l2) to Cartesian in the unit equilateral triangle."""
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        x = 0.5 - 0.5 * l1 + 0.5 * l2
        y = (1 - l1 - l2) * np.sqrt(3) / 2
        return np.column_stack([x, y])
    
    def params_to_bary(params):
        """
        Convert unconstrained parameters to valid barycentric coordinates.
        Uses softmax-like mapping to ensure l1 >= 0, l2 >= 0, l1 + l2 <= 1.
        params: shape (2N,) -> bary: shape (N, 2)
        """
        # Split into l1 and l2 parameters
        p1 = params[:N]
        p2 = params[N:2*N]
        
        # Use sigmoid to map to (0, 1)
        l1 = 1.0 / (1.0 + np.exp(-p1))
        l2 = 1.0 / (1.0 + np.exp(-p2))
        
        # Scale to ensure l1 + l2 <= 1 with margin
        # Use a soft constraint: if l1 + l2 > 1, scale both down
        s = l1 + l2
        # Map s from (0, 2) to (0, 0.99) using a smooth function
        # Use: scale = min(1, 0.99 / s) but smoothly
        # Better: use a smooth saturation
        max_s = 0.99
        # Smoothly clamp: if s > max_s, scale = max_s / s, else scale = 1
        # Use tanh for smoothness
        ratio = s / max_s
        # scale = 1 / max(1, ratio) smoothly
        # Use: scale = 1 / (1 + softplus(ratio - 1)) approximately
        # Simpler: use a piecewise smooth function
        # scale = 1.0 where s <= max_s, and max_s/s where s > max_s
        # Smooth approximation: scale = 1.0 / (1.0 + np.exp(5*(s - max_s)))
        # This gives ~1 when s << max_s, and ~max_s/s when s >> max_s
        scale = 1.0 / (1.0 + np.exp(10.0 * (s - max_s)))
        
        l1 = l1 * scale
        l2 = l2 * scale
        
        # Final safety clip
        l1 = np.clip(l1, 1e-6, 1.0)
        l2 = np.clip(l2, 1e-6, 1.0)
        
        # Ensure l1 + l2 <= 0.999
        s2 = l1 + l2
        mask = s2 > 0.999
        if np.any(mask):
            factor = 0.999 / s2[mask]
            l1[mask] *= factor
            l2[mask] *= factor
        
        return np.column_stack([l1, l2])
    
    def bary_to_params(bary):
        """Inverse of params_to_bary for initialization."""
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        # Inverse sigmoid
        p1 = np.log(l1 / (1 - l1 + 1e-10))
        p2 = np.log(l2 / (1 - l2 + 1e-10))
        return np.concatenate([p1, p2])
    
    def compute_soft_min(params, temperature=0.02):
        """
        Compute the soft-minimum of all triangle areas and its gradient.
        Returns: (soft_min_value, gradient_of_soft_min_wrt_params)
        """
        bary = params_to_bary(params)
        
        # Build full barycentric matrix (N, 3)
        l3 = 1.0 - bary[:, 0] - bary[:, 1]
        bary_full = np.column_stack([bary[:, 0], bary[:, 1], l3])
        
        # Compute all triangle areas using vectorized operations
        # For each triangle (i, j, k), area = |det(bary_full[[i,j,k], :])|
        # det of 3x3 matrix with rows [l1_i, l2_i, l3_i], etc.
        # det = l1_i*(l2_j*l3_k - l3_j*l2_k) - l2_i*(l1_j*l3_k - l3_j*l1_k) + l3_i*(l1_j*l2_k - l2_j*l1_k)
        
        i_idx = tri_indices[:, 0]
        j_idx = tri_indices[:, 1]
        k_idx = tri_indices[:, 2]
        
        l1_i = bary_full[i_idx, 0]
        l2_i = bary_full[i_idx, 1]
        l3_i = bary_full[i_idx, 2]
        l1_j = bary_full[j_idx, 0]
        l2_j = bary_full[j_idx, 1]
        l3_j = bary_full[j_idx, 2]
        l1_k = bary_full[k_idx, 0]
        l2_k = bary_full[k_idx, 1]
        l3_k = bary_full[k_idx, 2]
        
        # Compute determinants
        det_vals = (l1_i * (l2_j * l3_k - l3_j * l2_k) 
                   - l2_i * (l1_j * l3_k - l3_j * l1_k) 
                   + l3_i * (l1_j * l2_k - l2_j * l1_k))
        
        areas = np.abs(det_vals)
        # Ensure no zeros
        areas = np.maximum(areas, 1e-12)
        
        # Soft minimum: -t * log(sum(exp(-A_k / t)))
        # For numerical stability, use the log-sum-exp trick
        neg_areas_over_t = -areas / temperature
        max_neg = np.max(neg_areas_over_t)
        exp_terms = np.exp(neg_areas_over_t - max_neg)
        sum_exp = np.sum(exp_terms)
        
        soft_min = -temperature * (max_neg + np.log(sum_exp))
        
        # Weights for the soft minimum
        weights = exp_terms / sum_exp
        
        # Compute gradient of soft_min wrt barycentric coordinates
        # d(soft_min)/d(bary) = (1/t) * sum_k w_k * d(A_k)/d(bary)
        # But A_k = |det_k|, so d(A_k)/d(bary) = sign(det_k) * d(det_k)/d(bary)
        
        # We need d(det_k)/d(bary_p) for each point p
        # det = l1_i*(l2_j*l3_k - l3_j*l2_k) - l2_i*(l1_j*l3_k - l3_j*l1_k) + l3_i*(l1_j*l2_k - l2_j*l1_k)
        
        # For point p, bary_p = [l1_p, l2_p, l3_p] where l3_p = 1 - l1_p - l2_p
        # So d(det)/dl1_p = (coefficient of l1_p in det) - (coefficient of l3_p in det)
        # d(det)/dl2_p = (coefficient of l2_p in det) - (coefficient of l3_p in det)
        
        # Let's compute the partial derivatives of det_k with respect to l1_p, l2_p, l3_p
        # Then convert to dl1_p, dl2_p using l3_p = 1 - l1_p - l2_p
        
        # Initialize gradient accumulator for bary coordinates (N, 2)
        grad_bary = np.zeros((N, 2))
        
        # For each triangle, compute its contribution
        # We'll vectorize this
        signs = np.sign(det_vals)
        
        # d(det)/dl1_i: derivative of det with respect to l1 of point i
        # From the expansion:
        # d(det)/dl1_i = (l2_j*l3_k - l3_j*l2_k)
        # d(det)/dl2_i = -(l1_j*l3_k - l3_j*l1_k)
        # d(det)/dl3_i = (l1_j*l2_k - l2_j*l1_k)
        
        d_det_dl1_i = l2_j * l3_k - l3_j * l2_k
        d_det_dl2_i = -(l1_j * l3_k - l3_j * l1_k)
        d_det_dl3_i = l1_j * l2_k - l2_j * l1_k
        
        # d(det)/dl1_j: derivative with respect to l1 of point j
        # d(det)/dl1_j = -(l2_i*l3_k - l3_i*l2_k)
        # d(det)/dl2_j = (l1_i*l3_k - l3_i*l1_k)
        # d(det)/dl3_j = -(l1_i*l2_k - l2_i*l1_k)
        
        d_det_dl1_j = -(l2_i * l3_k - l3_i * l2_k)
        d_det_dl2_j = (l1_i * l3_k - l3_i * l1_k)
        d_det_dl3_j = -(l1_i * l2_k - l2_i * l1_k)
        
        # d(det)/dl1_k: derivative with respect to l1 of point k
        # d(det)/dl1_k = l2_i*l3_j - l3_i*l2_j
        # d(det)/dl2_k = -(l1_i*l3_j - l3_i*l1_j)
        # d(det)/dl3_k = l1_i*l2_j - l2_i*l1_j
        
        d_det_dl1_k = l2_i * l3_j - l3_i * l2_j
        d_det_dl2_k = -(l1_i * l3_j - l3_i * l1_j)
        d_det_dl3_k = l1_i * l2_j - l2_i * l1_j
        
        # Multiply by sign to get d|det|/dl
        w_scaled = weights * signs / temperature
        
        # Accumulate gradients for each point
        # For point i in triangle:
        # d(soft_min)/dl1_i += w_scaled * (d_det_dl1_i - d_det_dl3_i)
        # d(soft_min)/dl2_i += w_scaled * (d_det_dl2_i - d_det_dl3_i)
        
        # Use np.add.at for accumulation
        # Gradient contribution to l1 and l2 for point i
        grad_l1_i = (d_det_dl1_i - d_det_dl3_i) * w_scaled
        grad_l2_i = (d_det_dl2_i - d_det_dl3_i) * w_scaled
        
        grad_l1_j = (d_det_dl1_j - d_det_dl3_j) * w_scaled
        grad_l2_j = (d_det_dl2_j - d_det_dl3_j) * w_scaled
        
        grad_l1_k = (d_det_dl1_k - d_det_dl3_k) * w_scaled
        grad_l2_k = (d_det_dl2_k - d_det_dl3_k) * w_scaled
        
        np.add.at(grad_bary, (i_idx, 0), grad_l1_i)
        np.add.at(grad_bary, (i_idx, 1), grad_l2_i)
        np.add.at(grad_bary, (j_idx, 0), grad_l1_j)
        np.add.at(grad_bary, (j_idx, 1), grad_l2_j)
        np.add.at(grad_bary, (j_idx, 0), grad_l1_j)
        np.add.at(grad_bary, (j_idx, 1), grad_l2_j)
        np.add.at(grad_bary, (k_idx, 0), grad_l1_k)
        np.add.at(grad_bary, (k_idx, 1), grad_l2_k)
        
        # Now compute gradient wrt params using chain rule
        # params = [p1_0, ..., p1_{N-1}, p2_0, ..., p2_{N-1}]
        # l1_p = sigmoid(p1_p), l2_p = sigmoid(p2_p)
        # scale = 1 / (1 + exp(10*(s - max_s))) where s = l1 + l2
        # final_l1 = l1 * scale, final_l2 = l2 * scale
        
        # This is complex. Let's use finite differences for the parameter gradient
        # Or compute analytically...
        
        # Actually, let's simplify: use a simpler parameterization
        # Use direct barycentric coordinates with a penalty for constraint violation
        
        # For now, return the soft_min and a placeholder gradient
        # We'll use finite differences in the optimizer
        
        return soft_min, grad_bary
    
    def objective(params):
        """Objective: minimize -soft_min (i.e., maximize soft_min)"""
        soft_min, _ = compute_soft_min(params)
        return -soft_min
    
    def objective_grad(params):
        """Gradient of objective"""
        _, grad_bary = compute_soft_min(params)
        # Need to compute d(soft_min)/d(params) using chain rule
        # This requires the Jacobian of params_to_bary
        # For simplicity, use finite differences
        eps = 1e-7
        grad_params = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            _, grad_plus = compute_soft_min(params_plus)
            # This is too slow. Let's compute analytically
            break
        
        # Let's compute the chain rule properly
        bary = params_to_bary(params)
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        s = l1 + l2
        max_s = 0.99
        scale = 1.0 / (1.0 + np.exp(10.0 * (s - max_s)))
        
        # d(l1_final)/d(p1) = sigmoid'(p1) * scale
        # sigmoid'(p) = sigmoid(p) * (1 - sigmoid(p))
        sig_l1 = l1 / scale  # This is not quite right due to the scaling
        # Actually, l1 = sigmoid(p1) * scale, so d(l1)/dp1 = sigmoid'(p1) * scale + sigmoid(p1) * d(scale)/dp1
        # This is getting complicated. Let's use a simpler parameterization.
        
        # For now, return a zero gradient and rely on numerical differentiation
        return np.zeros_like(params)
    
    def hard_min_area(bary):
        """Compute the actual minimum area (for evaluation)"""
        l3 = 1.0 - bary[:, 0] - bary[:, 1]
        bary_full = np.column_stack([bary[:, 0], bary[:, 1], l3])
        
        i_idx = tri_indices[:, 0]
        j_idx = tri_indices[:, 1]
        k_idx = tri_indices[:, 2]
        
        l1_i = bary_full[i_idx, 0]
        l2_i = bary_full[i_idx, 1]
        l3_i = bary_full[i_idx, 2]
        l1_j = bary_full[j_idx, 0]
        l2_j = bary_full[j_idx, 1]
        l3_j = bary_full[j_idx, 2]
        l1_k = bary_full[k_idx, 0]
        l2_k = bary_full[k_idx, 1]
        l3_k = bary_full[k_idx, 2]
        
        det_vals = (l1_i * (l2_j * l3_k - l3_j * l2_k) 
                   - l2_i * (l1_j * l3_k - l3_j * l1_k) 
                   + l3_i * (l1_j * l2_k - l2_j * l1_k))
        
        areas = np.abs(det_vals)
        return np.min(areas)
    
    # Use a simpler approach: optimize directly in barycentric space with constraints
    # Use scipy's minimize with bounds
    
    def bary_objective(bary_params):
        """Objective in barycentric space"""
        bary = bary_params.reshape(N, 2)
        l3 = 1.0 - bary[:, 0] - bary[:, 1]
        
        # Check validity
        if np.any(l3 < 0) or np.any(bary[:, 0] < 0) or np.any(bary[:, 1] < 0):
            return 1e10
        
        bary_full = np.column_stack([bary[:, 0], bary[:, 1], l3])
        
        i_idx = tri_indices[:, 0]
        j_idx = tri_indices[:, 1]
        k_idx = tri_indices[:, 2]
        
        l1_i = bary_full[i_idx, 0]
        l2_i = bary_full[i_idx, 1]
        l3_i = bary_full[i_idx, 2]
        l1_j = bary_full[j_idx, 0]
        l2_j = bary_full[j_idx, 1]
        l3_j = bary_full[j_idx, 2]
        l1_k = bary_full[k_idx, 0]
        l2_k = bary_full[k_idx, 1]
        l3_k = bary_full[k_idx, 2]
        
        det_vals = (l1_i * (l2_j * l3_k - l3_j * l2_k) 
                   - l2_i * (l1_j * l3_k - l3_j * l1_k) 
                   + l3_i * (l1_j * l2_k - l2_j * l1_k))
        
        areas = np.abs(det_vals)
        areas = np.maximum(areas, 1e-12)
        
        # Soft minimum
        t = 0.02
        neg_over_t = -areas / t
        max_neg = np.max(neg_over_t)
        exp_terms = np.exp(neg_over_t - max_neg)
        sum_exp = np.sum(exp_terms)
        soft_min = -t * (max_neg + np.log(sum_exp))
        
        return -soft_min
    
    best_bary = None
    best_score = -np.inf
    
    # Multi-start with structured initializations
    np.random.seed(42)
    
    # Generate several different initial configurations
    initializations = []
    
    # 1. Uniform random
    for _ in range(5):
        r = np.random.exponential(1.0, (N, 3))
        r /= r.sum(axis=1, keepdims=True)
        bary_init = r[:, :2]
        bary_init = np.clip(bary_init, 0.01, 0.98)
        s = bary_init[:, 0] + bary_init[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary_init[mask, 0] *= 0.98 / s[mask]
            bary_init[mask, 1] *= 0.98 / s[mask]
        initializations.append(bary_init)
    
    # 2. Grid-based
    grid_bary = np.array([
        [0.5, 0.5],
        [0.33, 0.33],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.15, 0.85],
        [0.5, 0.2],
        [0.2, 0.5],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.7],
        [0.7, 0.5]
    ])
    initializations.append(grid_bary)
    
    # 3. Symmetric patterns
    # Vertices and midpoints
    sym_bary = np.array([
        [0.02, 0.02],  # Near vertex 1
        [0.98, 0.02],  # Near vertex 2
        [0.02, 0.98],  # Near vertex 3
        [0.5, 0.02],   # Midpoint of edge 1-2
        [0.02, 0.5],   # Midpoint of edge 1-3
        [0.5, 0.5],    # Midpoint of edge 2-3
        [0.33, 0.33],  # Centroid
        [0.4, 0.4],
        [0.2, 0.4],
        [0.4, 0.2],
        [0.3, 0.5]
    ])
    initializations.append(sym_bary)
    
    # Optimize from each initialization
    bounds = [(0.001, 0.998)] * (2 * N)
    
    for init_bary in initializations:
        try:
            result = minimize(
                bary_objective,
                init_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            final_bary = result.x.reshape(N, 2)
            
            # Validate
            l3 = 1.0 - final_bary[:, 0] - final_bary[:, 1]
            if np.any(l3 < 0) or np.any(final_bary[:, 0] < 0) or np.any(final_bary[:, 1] < 0):
                continue
            
            score = hard_min_area(final_bary)
            
            if score > best_score:
                best_score = score
                best_bary = final_bary.copy()
                
        except Exception as e:
            continue
    
    # If no valid result, use fallback
    if best_bary is None:
        best_bary = np.array([
            [0.5, 0.5],
            [0.33, 0.33],
            [0.1, 0.1],
            [0.8, 0.1],
            [0.1, 0.8],
            [0.5, 0.2],
            [0.2, 0.5],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.5, 0.7],
            [0.7, 0.5]
        ])
    
    # Convert to Cartesian
    cart = bary_to_cartesian(best_bary)
    
    # Ensure points are strictly inside
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
