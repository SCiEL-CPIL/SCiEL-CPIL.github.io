"""Packing 200 equal circles in a unit square using model-and-optimize approach with ADMM refinement."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _energy(params, temperature=0.02, threshold=0.10):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use log-sum-exp based soft minimum for numerical stability
    T = max(temperature, 1e-8)
    
    # For pairwise distances: soft min
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        # Shift for numerical stability
        shifted = -(d_sub - d_min_sub) / T
        shifted_max = np.max(shifted)
        soft_min_pair = d_min_sub - T * (shifted_max + np.log(np.sum(np.exp(shifted - shifted_max))))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_shifted = -(boundary_dists - b_min) / T
    b_shifted_max = np.max(b_shifted)
    soft_min_bound = b_min - T * (b_shifted_max + np.log(np.sum(np.exp(b_shifted - b_shifted_max))))
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound using log-sum-exp
    r_min = min(r_pair, r_bound)
    r_shifted = -(np.array([r_pair, r_bound]) - r_min) / T
    r_shifted_max = np.max(r_shifted)
    soft_min_combined = r_min - T * (r_shifted_max + np.log(np.sum(np.exp(r_shifted - r_shifted_max))))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _local_optimize(c0, max_time=30.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.03,
        0.02,
        0.015,
        0.01,
        0.007,
        0.005,
    ]
    if temperature_schedule is not None:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 3:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _admm_optimize(c0, r0, max_time=40.0, rho=10.0, max_iter=1000):
    """
    ADMM optimization for circle packing.
    Maximizes r subject to pairwise distance constraints and boundary constraints.
    Uses an augmented Lagrangian approach for the pairwise constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if r0 > best_r:
        best_r = r0
    
    # Pre-compute pair indices for efficiency
    pair_indices = []
    for i in range(N):
        for j in range(i+1, N):
            pair_indices.append((i, j))
    
    n_pairs = len(pair_indices)
    
    # Initialize dual variables and auxiliary variables
    nu = np.zeros((n_pairs, 2))
    d = np.zeros((n_pairs, 2))
    
    # Initialize d based on current c
    for k, (i, j) in enumerate(pair_indices):
        d[k] = c0[i] - c0[j]
    
    current_r = best_r
    current_c = c0.copy()
    
    # ADMM parameters
    rho = rho
    
    def _c_update_obj_and_grad(c_flat):
        c = c_flat.reshape(-1, 2)
        grad = np.zeros_like(c)
        
        i_idx, j_idx = np.array(pair_indices).T
        diffs = c[i_idx] - c[j_idx]
        targets = d - nu
        residuals = diffs - targets
        obj = 0.5 * rho * np.sum(residuals**2)
        
        grad_i = np.zeros_like(c)
        np.add.at(grad_i, i_idx, rho * residuals)
        np.add.at(grad_i, j_idx, -rho * residuals)
        
        return obj, grad_i.flatten()
    
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        # 1. c-update
        def c_obj_grad(x):
            val, grad = _c_update_obj_and_grad(x)
            return val, grad
        
        try:
            result = minimize(
                c_obj_grad,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_c = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
        except Exception:
            break
        
        # 2. d-update
        i_idx, j_idx = np.array(pair_indices).T
        c_diffs = current_c[i_idx] - current_c[j_idx]
        targets = c_diffs + nu
        
        norms = np.linalg.norm(targets, axis=1)
        min_dist = 2.0 * current_r
        
        d_new = np.copy(targets)
        mask = norms < min_dist
        if np.any(mask):
            safe_norms = norms[mask]
            safe_norms[safe_norms < 1e-12] = 1e-12
            scale = min_dist / safe_norms
            d_new[mask] = targets[mask] * scale[:, np.newaxis]
        
        d = d_new
        
        # 3. nu-update
        residuals = c_diffs - d
        nu += residuals
        
        # 4. r-update
        new_r = _compute_radius(current_c)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = current_c.copy()
        
        # Check convergence
        if it > 10 and np.linalg.norm(residuals) < 1e-6:
            break
    
    return best_c, best_r


def _slsqp_polish(c0, max_time=30.0):
    """SLSQP-based polish with hard constraints on pairwise and boundary distances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _build_constraints(c_ref, t_ref):
        """Build inequality constraint functions for SLSQP."""
        constraints = []
        
        # Find close pairs (within 2.5 * t_ref)
        dists = pdist(c_ref, metric='euclidean')
        close_pairs = np.where(dists < 2.5 * t_ref)[0]
        
        # Map from pdist index to (i, j)
        def _pdist_to_ij(k, n):
            i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
            j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
            return i, j
        
        for k in close_pairs:
            i, j = _pdist_to_ij(k, N)
            
            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t
            
            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
        
        # Boundary constraints
        for i in range(N):
            for k in range(2):
                def _bound_fun(x, _i=i, _k=k):
                    t = x[0]
                    c_val = x[1 + _i*2 + _k]
                    return 0.5 - np.abs(c_val) - t
                
                def _bound_jac(x, _i=i, _k=k):
                    grad = np.zeros_like(x)
                    grad[0] = -1.0
                    c_val = x[1 + _i*2 + _k]
                    grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                    return grad
                
                constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        return constraints
    
    def _slsqp_obj(x):
        t = x[0]
        return -t
    
    def _slsqp_grad(x):
        grad = np.zeros_like(x)
        grad[0] = -1.0
        return grad
    
    current_c = c0.copy()
    current_r = best_r
    
    for attempt in range(3):
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            constraints = _build_constraints(current_c, current_r)
            
            x0 = np.concatenate([[current_r], current_c.flatten()])
            
            result = minimize(
                _slsqp_obj,
                x0,
                method='SLSQP',
                bounds=[(0, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                constraints=constraints,
                jac=_slsqp_grad,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            t_new = result.x[0]
            c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
                current_r = r_new
            else:
                current_c = c_new
                current_r = r_new
        except Exception:
            continue
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.004,
        0.003,
        0.002,
        0.0015,
    ]
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 200}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _logdet_optimize(c0, max_time=30.0, n_iter=200, lr=0.001):
    """
    Log-determinant based optimization.
    Maximizes log(det(Gram matrix of centered positions)) which promotes uniform spacing.
    This is a concave function of the positions that serves as a surrogate for maximizing minimum distance.
    Uses accelerated proximal gradient with projection onto the square.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Center the points
    c = c0.copy()
    mean_c = c.mean(axis=0)
    c_centered = c - mean_c
    
    # Compute Gram matrix
    G = c_centered @ c_centered.T
    
    # Add small regularization for numerical stability
    G_reg = G + 1e-10 * np.eye(N)
    
    # Compute log determinant
    sign, logdet = np.linalg.slogdet(G_reg)
    if sign <= 0:
        return best_c, best_r
    
    # Gradient of log(det(G)) with respect to c
    # d/dc log(det(c_centered @ c_centered.T)) = 2 * c_centered @ inv(G) @ c_centered.T / det(G) * ... 
    # Actually, let's use the matrix cookbook:
    # d log(det(A)) = tr(A^{-1} dA)
    # A = X X^T, dA = dX X^T + X dX^T
    # tr(A^{-1} dA) = tr(A^{-1} dX X^T) + tr(A^{-1} X dX^T)
    # = tr(X^T A^{-1} dX) + tr(dX^T X A^{-1})
    # = 2 tr(X A^{-1} dX)  (since A is symmetric)
    # So gradient wrt X is 2 X A^{-1}
    
    try:
        G_inv = np.linalg.inv(G_reg)
    except np.linalg.LinAlgError:
        return best_c, best_r
    
    # Momentum for accelerated gradient
    v = np.zeros_like(c)
    c_prev = c.copy()
    t = 1.0
    
    for it in range(n_iter):
        if time.time() > start_time + max_time - 1:
            break
        
        # Recompute centered positions and Gram matrix
        mean_c = c.mean(axis=0)
        c_centered = c - mean_c
        G = c_centered @ c_centered.T
        G_reg = G + 1e-10 * np.eye(N)
        
        try:
            G_inv = np.linalg.inv(G_reg)
        except np.linalg.LinAlgError:
            break
        
        # Gradient of log(det(G_reg)) wrt c
        # grad_c = 2 * c_centered @ G_inv (since the centering introduces a constant shift that doesn't affect gradient)
        grad = 2.0 * c_centered @ G_inv
        
        # Negative gradient (we want to maximize)
        grad_neg = -grad
        
        # Accelerated proximal gradient step
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        beta = (t - 1) / t_new
        
        c_new = c + beta * (c - c_prev)
        c_new = c_new - lr * grad_neg
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        c_prev = c
        c = c_new
        t = t_new
        
        # Evaluate
        r_new = _compute_radius(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r


def _critical_polish(c0, max_time=30.0):
    """
    High-precision polish using a combination of:
    1. Very tight L-BFGS-B with multiple temperature levels
    2. SLSQP with hard constraints for final refinement
    3. Active-set refinement to focus on binding constraints
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Very tight L-BFGS-B with multiple temperature levels
    temperatures = [0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0005]
    current_c = c0.copy()
    
    for T in temperatures:
        if time.time() > start_time + max_time - 12:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T, threshold=0.08)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 300}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    # Phase 2: SLSQP with active-set refinement
    if time.time() < start_time + max_time - 5:
        def _pdist_to_ij(k, n):
            i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
            j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
            return i, j
        
        def _build_active_constraints(c_ref, t_ref, margin=1.02):
            """Build inequality constraint functions for SLSQP focusing on active constraints."""
            constraints = []
            
            # Find close pairs (within margin * t_ref)
            dists = pdist(c_ref, metric='euclidean')
            close_pairs = np.where(dists < margin * t_ref)[0]
            
            for k in close_pairs:
                i, j = _pdist_to_ij(k, N)
                
                def _pair_fun(x, _i=i, _j=j):
                    t = x[0]
                    ci = x[1 + _i*2:1 + _i*2+2]
                    cj = x[1 + _j*2:1 + _j*2+2]
                    d = np.linalg.norm(ci - cj)
                    return d - 2.0 * t
                
                def _pair_jac(x, _i=i, _j=j):
                    t = x[0]
                    ci = x[1 + _i*2:1 + _i*2+2]
                    cj = x[1 + _j*2:1 + _j*2+2]
                    diff = ci - cj
                    d = np.linalg.norm(diff)
                    if d < 1e-12:
                        d = 1e-12
                    grad = np.zeros_like(x)
                    grad[0] = -2.0
                    grad[1 + _i*2:1 + _i*2+2] = diff / d
                    grad[1 + _j*2:1 + _j*2+2] = -diff / d
                    return grad
                
                constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
            
            # Boundary constraints
            for i in range(N):
                for k in range(2):
                    def _bound_fun(x, _i=i, _k=k):
                        t = x[0]
                        c_val = x[1 + _i*2 + _k]
                        return 0.5 - np.abs(c_val) - t
                    
                    def _bound_jac(x, _i=i, _k=k):
                        grad = np.zeros_like(x)
                        grad[0] = -1.0
                        c_val = x[1 + _i*2 + _k]
                        grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                        return grad
                    
                    constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
            
            return constraints
        
        def _slsqp_obj(x):
            t = x[0]
            return -t
        
        def _slsqp_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -1.0
            return grad
        
        current_c = best_c.copy()
        current_r = best_r
        
        # Multiple SLSQP passes with progressively tighter active sets
        for attempt in range(5):
            if time.time() > start_time + max_time - 2:
                break
            
            # Use progressively tighter margins to focus on truly active constraints
            margins = [1.05, 1.03, 1.02, 1.01, 1.005]
            margin = margins[min(attempt, len(margins)-1)]
            
            try:
                constraints = _build_active_constraints(current_c, current_r, margin=margin)
                
                x0 = np.concatenate([[current_r], current_c.flatten()])
                
                result = minimize(
                    _slsqp_obj,
                    x0,
                    method='SLSQP',
                    bounds=[(0, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                    constraints=constraints,
                    jac=_slsqp_grad,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
                )
                
                t_new = result.x[0]
                c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                
                if r_new > best_r:
                    best_c = c_new.copy()
                    best_r = r_new
                    current_c = c_new
                    current_r = r_new
                else:
                    current_c = c_new
                    current_r = r_new
            except Exception:
                continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # Safety margin: 7s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from hex
    if time.time() < deadline - 55:
        max_opt_time = min(35.0, deadline - time.time() - 25)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Log-det optimization (new)
    if time.time() < deadline - 45:
        remaining = deadline - time.time() - 15
        if remaining > 15:
            c_ld, r_ld = _logdet_optimize(best_c, max_time=min(20.0, remaining))
            if r_ld > best_radius:
                best_radius = r_ld
                best_c = c_ld.copy()
    
    # Step 4: ADMM optimization
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 15
        if remaining > 20:
            c_admm, r_admm = _admm_optimize(best_c, best_radius, max_time=min(30.0, remaining))
            if r_admm > best_radius:
                best_radius = r_admm
                best_c = c_admm.copy()
    
    # Step 5: Multiple random restarts
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 35:
            break
        
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 25
        if remaining > 5:
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 6: SLSQP hard-constrained polish
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(25.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    
    # Step 7: Critical polish pass (combined L-BFGS-B + SLSQP)
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 5
        if remaining > 8:
            c_crit, r_crit = _critical_polish(best_c, max_time=min(35.0, remaining))
            if r_crit > best_radius:
                best_radius = r_crit
                best_c = c_crit.copy()
    
    # Step 8: Final ultra-precise single pass if time allows
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            def obj_final(params):
                return _energy(params, temperature=0.0003, threshold=0.06)
            
            try:
                result = minimize(
                    obj_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={'maxiter': 8000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 800}
                )
                
                c_final = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                r_final = _compute_radius(c_final)
                
                if r_final > best_radius:
                    best_radius = r_final
                    best_c = c_final
            except Exception:
                pass
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
