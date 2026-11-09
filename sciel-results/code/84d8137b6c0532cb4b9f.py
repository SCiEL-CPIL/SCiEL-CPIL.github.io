# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _shannon_entropy(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))

def _get_projection_values(n, r):
    """Precompute the projection values x + r*y for all cells."""
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    vals = xs + r * ys
    unique_vals, inverse = np.unique(vals, return_inverse=True)
    return unique_vals, inverse, vals

# Precompute for n=10
N = 10
PROJ_DATA = {}
for r in [-1.0, 0.0, 1.0, 2.0]:
    unique_vals, inverse, vals = _get_projection_values(N, r)
    PROJ_DATA[r] = {
        'unique_vals': unique_vals,
        'inverse': inverse,
        'vals': vals,
        'num_bins': len(unique_vals)
    }

def _compute_proj_probs(P, r):
    """Compute the projected probability distribution for slope r."""
    inverse = PROJ_DATA[r]['inverse']
    num_bins = PROJ_DATA[r]['num_bins']
    # P is flattened
    proj = np.bincount(inverse.flatten(), weights=P.flatten(), minlength=num_bins)
    return proj

def _entropy_and_gradient(P, r):
    """
    Compute Shannon entropy H and its gradient with respect to P for projection r.
    P is a flattened array of shape (100,).
    """
    proj = _compute_proj_probs(P, r)
    
    # Mask for positive probabilities
    mask = proj > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(P)
    
    p = proj[mask]
    H = float(-np.sum(p * np.log2(p)))
    
    # Gradient of H with respect to proj
    # dH/dproj_v = -1/ln(2) * (1 + log2(proj_v)) for proj_v > 0
    # 0 otherwise
    grad_proj = np.zeros_like(proj)
    grad_proj[mask] = -(1.0 / np.log(2.0)) * (1.0 + np.log2(p))
    
    # Gradient with respect to P
    # proj = sum_{(x,y) in bin v} P[y,x]
    # dH/dP[y,x] = dH/dproj_v(y,x)
    inverse = PROJ_DATA[r]['inverse']
    grad_P = grad_proj[inverse.flatten()]
    
    return H, grad_P

def _soft_max(values, beta):
    """Compute soft max of values."""
    # values is a list or array of H_0, H_1, H_2
    v = np.array(values)
    # Scale by beta
    v_scaled = beta * v
    # Shift for numerical stability
    max_v = np.max(v_scaled)
    exps = np.exp(v_scaled - max_v)
    sum_exps = np.sum(exps)
    return (1.0 / beta) * (max_v + np.log(sum_exps))

def _objective_Z(Z):
    """
    Objective function for optimization in Z-space.
    P = softmax(Z)
    Returns -D / M_soft
    """
    # Compute P from Z
    Z_shifted = Z - np.max(Z)
    exps = np.exp(Z_shifted)
    P_flat = exps / np.sum(exps)
    P = P_flat.reshape(N, N)
    
    # Compute Entropies and Gradients
    H_m1, grad_m1 = _entropy_and_gradient(P_flat, -1.0)
    H_0, grad_0 = _entropy_and_gradient(P_flat, 0.0)
    H_1, grad_1 = _entropy_and_gradient(P_flat, 1.0)
    H_2, grad_2 = _entropy_and_gradient(P_flat, 2.0)
    
    D = H_m1
    M_vals = [H_0, H_1, H_2]
    
    # Use soft max for M
    beta = 20.0 # Temperature parameter
    M_soft = _soft_max(M_vals, beta)
    
    if M_soft < 1e-9 or D < 1e-12:
        return 1e6, np.zeros_like(Z)
    
    val = -D / M_soft
    
    # Compute Gradient
    # dM_soft/dH_j = w_j
    v_scaled = beta * np.array(M_vals)
    max_v = np.max(v_scaled)
    exps = np.exp(v_scaled - max_v)
    sum_exps = np.sum(exps)
    weights = exps / sum_exps # w_0, w_1, w_2
    
    grad_M = weights[0] * grad_0 + weights[1] * grad_1 + weights[2] * grad_2
    grad_D = grad_m1
    
    # grad_g wrt P
    # g = -D / M
    # dg/dP = (D * dM/dP - M * dD/dP) / M^2
    grad_g_P = (D * grad_M - M_soft * grad_D) / (M_soft * M_soft)
    
    # Transform gradient from P-space to Z-space
    # P_i = exp(Z_i) / S
    # dP_i/dZ_k = P_i (delta_ik - P_k)
    # grad_Z_k = sum_i (dg/dP_i) * P_i (delta_ik - P_k)
    #           = (dg/dP_k) * P_k - P_k * sum_i (dg/dP_i) P_i
    V = np.sum(grad_g_P * P_flat)
    grad_Z = P_flat * (grad_g_P - V)
    
    return val, grad_Z

def _make_initial_Z_patterns(n):
    """Generate diverse initial Z patterns (log-probabilities)."""
    patterns = []
    
    # 1. Uniform
    P = np.ones((n, n)) / (n * n)
    patterns.append(np.log(P + 1e-15))
    
    # 2. Diagonal x=y
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1.0 / n
    patterns.append(np.log(P + 1e-15))
    
    # 3. Anti-diagonal x+y=n-1
    P = np.zeros((n, n))
    for i in range(n):
        P[i, n - 1 - i] = 1.0 / n
    patterns.append(np.log(P + 1e-15))
    
    # 4. Concentrated on a few parallel lines with slope -1
    for offset in range(-4, 5):
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            x = y + offset
            if 0 <= x < n:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            patterns.append(np.log(P + 1e-15))
            
    # 5. Random sparse
    rng = np.random.RandomState(42)
    for _ in range(5):
        P = np.zeros((n, n))
        k = rng.randint(5, 15)
        indices = rng.choice(n * n, size=k, replace=False)
        P.flatten()[indices] = 1.0
        P /= P.sum()
        patterns.append(np.log(P + 1e-15))
        
    # 6. Random dense
    for _ in range(5):
        P = rng.rand(n, n)
        P /= P.sum()
        patterns.append(np.log(P + 1e-15))
        
    # 7. "Block" patterns - mass in a square block
    for size in [2, 3, 4]:
        for start_y in range(0, n - size + 1, 2):
            for start_x in range(0, n - size + 1, 2):
                P = np.zeros((n, n))
                P[start_y:start_y+size, start_x:start_x+size] = 1.0
                P /= P.sum()
                patterns.append(np.log(P + 1e-15))
                
    return patterns

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 580
    
    initial_Z_patterns = _make_initial_Z_patterns(n)
    
    best_Z = initial_Z_patterns[0].copy()
    best_P = np.exp(best_Z - np.max(best_Z))
    best_P /= best_P.sum()
    best_score = -_objective_Z(best_Z)[0]
    
    # L-BFGS-B is unconstrained, so no constraints needed.
    # Bounds on Z can be wide, e.g., [-50, 50] to allow P to be very small or large relative to others.
    bounds = [(-50.0, 50.0) for _ in range(n * n)]
    
    num_starts = min(len(initial_Z_patterns), 20)
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        x0 = initial_Z_patterns[i].flatten().copy()
        
        try:
            result = minimize(
                _objective_Z,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
            )
            
            P_candidate = np.exp(result.x - np.max(result.x))
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                P_candidate = P_candidate.reshape(n, n)
                score, _ = _objective_Z(result.x)
                score = -score
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
                    best_Z = result.x.copy()
                    
        except Exception as e:
            continue
            
    # If time remains, polish with local search around best_Z
    if time.time() < deadline - 30:
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > deadline:
                break
            noise = rng.randn(n, n) * 0.5
            Z_start = best_Z + noise
            Z_start = np.clip(Z_start, -50, 50)
            
            try:
                result = minimize(
                    _objective_Z,
                    Z_start.flatten().copy(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
                )
                P_candidate = np.exp(result.x - np.max(result.x))
                s = P_candidate.sum()
                if s > 1e-12:
                    P_candidate /= s
                    P_candidate = P_candidate.reshape(n, n)
                    score, _ = _objective_Z(result.x)
                    score = -score
                    if score > best_score:
                        best_score = score
                        best_P = P_candidate.copy()
                        best_Z = result.x.copy()
            except Exception:
                continue

    # Final normalization and clipping
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-12:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
