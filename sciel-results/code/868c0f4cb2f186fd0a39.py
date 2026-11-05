"""Convex QP trust-region approach for peak autoconvolution suppression."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.linalg import solve as la_solve

N = 400
DEADLINE = 148.0

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _autoconv(h):
    """Compute full autoconvolution."""
    return np.convolve(h, h)

def _build_conv_matrix(k):
    """
    Build the N x N symmetric matrix M_k such that conv[k] = h^T M_k h.
    For lag k (0 <= k < N), conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k].
    This is h^T M_k h where M_k has 0.5 on the (i, i+k) and (i+k, i) positions.
    """
    M = np.zeros((N, N))
    for i in range(N - k):
        M[i, i + k] = 0.5
        M[i + k, i] = 0.5
    return M

def _qp_solve_direct(h0, M, c, r, delta):
    """
    Solve: min h^T M h + c^T h  s.t. ||h - h0||_2 <= delta
    Using projected gradient with line search on the sphere constraint.
    M is PSD (sum of rank-1 outer products with non-negative coefficients).
    """
    h = h0.copy()
    best_val = h @ M @ h + c @ h
    best_h = h.copy()
    
    # Precompute Cholesky-like factorization of (M + mu*I) for varying mu
    # Use simple projected gradient with backtracking
    
    for it in range(200):
        # Gradient of objective
        g = 2.0 * M @ h + c
        
        # Projected gradient step: minimize along -g within ball
        # Optimal step: h_new = h0 - alpha * g, with alpha chosen so ||h_new - h0|| <= delta
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        
        # Try full step in direction -g, clipped to ball
        d = -g / g_norm  # unit direction
        # h_new = h0 + t*d, need ||t*d|| <= delta => t <= delta
        # Also want to move from h toward h0 - alpha*g
        # Simple approach: step = min(1/g_norm, delta) * (-g)
        step_len = min(1.0 / g_norm, delta)
        h_new = h + step_len * (-g)
        
        # Project onto ball around h0
        diff = h_new - h0
        diff_norm = np.linalg.norm(diff)
        if diff_norm > delta:
            h_new = h0 + (delta / diff_norm) * diff
        
        # Line search: try to improve
        obj_new = h_new @ M @ h_new + c @ h_new
        if obj_new < best_val - 1e-15:
            best_val = obj_new
            best_h = h_new.copy()
            h = h_new
        else:
            # Halve step
            step_len *= 0.5
            h_new = h + step_len * (-g)
            diff = h_new - h0
            diff_norm = np.linalg.norm(diff)
            if diff_norm > delta:
                h_new = h0 + (delta / diff_norm) * diff
            obj_new = h_new @ M @ h_new + c @ h_new
            if obj_new < best_val - 1e-15:
                best_val = obj_new
                best_h = h_new.copy()
                h = h_new
            else:
                break
    
    return best_h, best_val

def _trust_region_iterate(h, delta, max_iter=50):
    """
    Trust-region iteration: repeatedly solve QP for the active lag,
    update the active lag, until convergence.
    Returns improved h.
    """
    for it in range(max_iter):
        conv = _autoconv(h)
        k_star = int(np.argmax(conv))
        max_val = conv[k_star]
        
        # Build M_k_star
        M = _build_conv_matrix(k_star)
        
        # Linearize: c = 2*M*h (gradient of h^T M h at h)
        c = 2.0 * M @ h
        
        # Solve QP: min h^T M h + c^T h s.t. ||h - h||_2 <= delta
        # But we want to minimize h^T M h, so the QP is:
        # min h^T M h  s.t. ||h_new - h|| <= delta
        # Gradient at h: 2*M*h, so linear approx: h^T M h ≈ 2*M*h @ h_new + const
        # So c = 2*M*h, and we minimize c^T h_new subject to ball constraint
        # But we also keep the quadratic term for better convergence
        
        # Actually, let's minimize h_new^T M h_new directly (convex since M is PSD)
        # subject to ||h_new - h|| <= delta
        h_new, obj = _qp_solve_direct(h, M, np.zeros(N), None, delta)
        
        # Check if improvement
        conv_new = _autoconv(h_new)
        max_new = np.max(conv_new)
        
        if max_new < max_val - 1e-10:
            h = h_new
            delta = min(2.0 * delta, 5.0)
        else:
            delta *= 0.5
            if delta < 1e-6:
                break
    
    return h

def _eigen_init():
    """
    Initialize using eigenvector of the sum of all conv matrices.
    The sum of M_k over k=0..N-1 gives a matrix whose eigenvector
    minimizes the average autoconvolution.
    """
    # Sum of all M_k: this is the matrix for sum_k conv[k] = (sum h)^2
    # So the eigenvector that minimizes (sum h)^2 / ||h||^2 is the one
    # orthogonal to the all-ones vector (if we constrain ||h||=1)
    # But we want to minimize max_k conv[k], not the sum.
    
    # Alternative: use the matrix for k=0 (which is sum h_i^2)
    # Its minimum eigenvector (with eigenvalue 0) is any vector in the null space
    # of the identity... that's not helpful.
    
    # Better: random initialization with zero mean (to make sum small, 
    # but we need sum != 0)
    # Actually, let's just use structured initializations.
    return None

def _multi_start_optimize(initial_points, time_budget):
    """Run trust-region optimization from multiple starts."""
    best_h = initial_points[0].copy()
    best_val = _eval_value(best_h)
    t0 = time.time()
    
    for init_h in initial_points:
        if time.time() - t0 > time_budget:
            break
        
        h = init_h.copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            h[0] += 0.1
            s = np.sum(h)
        
        # Run trust region with decreasing delta
        delta = 1.0
        for dr in range(15):
            if time.time() - t0 > time_budget:
                break
            h = _trust_region_iterate(h, delta, max_iter=30)
            delta *= 0.7
        
        v = _eval_value(h)
        if v < best_val and np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-10:
            best_val = v
            best_h = h.copy()
    
    return best_h, best_val

def construct():
    t0 = time.time()
    rng = np.random.RandomState(123)
    
    # Generate diverse initial points
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random with small mean
    for _ in range(10):
        h = rng.randn(N) * 0.5 + 0.1
        initial_points.append(h)
    
    # Low-frequency patterns
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 8, 13, 21]:
        initial_points.append(0.5 + 0.5 * np.cos(2 * np.pi * freq * t))
        initial_points.append(0.5 + 0.5 * np.sin(2 * np.pi * freq * t))
    
    # Bimodal spikes
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        h += 0.05  # ensure non-zero sum
        initial_points.append(h)
    
    # Gaussian bumps
    i_arr = np.arange(N, dtype=float)
    for center in [N//4, N//2, 3*N//4]:
        for width in [30, 60, 100]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(0.1 + h_bump)
            initial_points.append(0.1 - h_bump + 0.2)
    
    # Smooth random
    for _ in range(5):
        h = rng.randn(N)
        # Simple smoothing via FFT
        H = np.fft.rfft(h)
        freqs = np.fft.rfftfreq(N)
        H *= np.exp(-20 * freqs**2)
        h_smooth = np.fft.irfft(H, N)
        initial_points.append(0.1 + h_smooth)
    
    # Evaluate and sort
    init_vals = []
    for h in initial_points:
        v = _eval_value(h)
        if np.isfinite(v):
            init_vals.append((v, h))
    init_vals.sort(key=lambda x: x[0])
    
    # Take top 20
    top_inits = [h for _, h in init_vals[:20]]
    
    # Run multi-start optimization
    time_budget = min(DEADLINE - 10 - (time.time() - t0), 100.0)
    best_h, best_val = _multi_start_optimize(top_inits, time_budget)
    
    # Final safety
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
