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

def _objective_and_grad(P_flat):
    """
    Objective: -D / M
    D = H_{-1}
    M = max(H_0, H_1, H_2)
    """
    P = P_flat.reshape(N, N)
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return 1e6, np.zeros_like(P_flat)
    P_norm = P / s
    
    H_m1, grad_m1 = _entropy_and_gradient(P_norm.flatten(), -1.0)
    H_0, grad_0 = _entropy_and_gradient(P_norm.flatten(), 0.0)
    H_1, grad_1 = _entropy_and_gradient(P_norm.flatten(), 1.0)
    H_2, grad_2 = _entropy_and_gradient(P_norm.flatten(), 2.0)
    
    D = H_m1
    M_vals = [H_0, H_1, H_2]
    M = max(M_vals)
    
    if M < 1e-9 or D < 1e-12:
        return 1e6, np.zeros_like(P_flat)
    
    val = -D / M
    
    # Gradient of -D/M
    # Let f = -D/M. 
    # df = - (M dD - D dM) / M^2 = (D dM - M dD) / M^2
    # We need dM. M = max(H_0, H_1, H_2).
    # If unique max, dM = grad of that H.
    # If tie, use subgradient (average of active grads) or just pick one. 
    # For smoothness, let's identify active indices.
    active_indices = [i for i, h in enumerate(M_vals) if abs(h - M) < 1e-8]
    
    grad_M = np.zeros_like(P_flat)
    for idx in active_indices:
        if idx == 0:
            grad_M += grad_0
        elif idx == 1:
            grad_M += grad_1
        else:
            grad_M += grad_2
    grad_M /= len(active_indices)
    
    grad_D = grad_m1
    
    grad_val = (D * grad_M - M * grad_D) / (M * M)
    
    return val, grad_val

def _make_initial_patterns(n):
    """Generate diverse initial patterns."""
    patterns = []
    
    # 1. Uniform
    patterns.append(np.ones((n, n)) / (n * n))
    
    # 2. Diagonal x=y
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 1.0 / n
    patterns.append(P)
    
    # 3. Anti-diagonal x+y=n-1
    P = np.zeros((n, n))
    for i in range(n):
        P[i, n - 1 - i] = 1.0 / n
    patterns.append(P)
    
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
            patterns.append(P.copy())
            
    # 5. Random sparse
    rng = np.random.RandomState(42)
    for _ in range(5):
        P = np.zeros((n, n))
        k = rng.randint(5, 15)
        indices = rng.choice(n * n, size=k, replace=False)
        P.flatten()[indices] = 1.0
        P /= P.sum()
        patterns.append(P)
        
    # 6. Random dense
    for _ in range(5):
        P = rng.rand(n, n)
        P /= P.sum()
        patterns.append(P)
        
    # 7. "Block" patterns - mass in a square block
    for size in [2, 3, 4]:
        for start_y in range(0, n - size + 1, 2):
            for start_x in range(0, n - size + 1, 2):
                P = np.zeros((n, n))
                P[start_y:start_y+size, start_x:start_x+size] = 1.0
                P /= P.sum()
                patterns.append(P)
                
    return patterns

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 580
    
    initial_patterns = _make_initial_patterns(n)
    
    best_P = initial_patterns[0].copy()
    best_score = -_objective_and_grad(best_P.flatten())[0]
    
    # SLSQP constraints
    constraints = [{'type': 'eq', 'fun': lambda P: np.sum(P) - 1.0, 'jac': lambda P: np.ones_like(P)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    # Multi-start SLSQP
    num_starts = min(len(initial_patterns), 20)
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        x0 = initial_patterns[i].flatten().copy()
        
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                jac=True, # Pass gradient
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            
            P_candidate = result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score, _ = _objective_and_grad(P_candidate.flatten())
                score = -score # Convert back to max ratio
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
                    
        except Exception as e:
            continue
            
    # If time remains, polish with local search around best_P
    if time.time() < deadline - 30:
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > deadline:
                break
            noise = rng.randn(n, n) * 0.05
            P_start = best_P + noise
            P_start = np.clip(P_start, 0, None)
            s = P_start.sum()
            if s > 1e-12:
                P_start /= s
            else:
                continue
                
            try:
                result = minimize(
                    _objective_and_grad,
                    P_start.flatten().copy(),
                    jac=True,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
                )
                P_candidate = result.x.reshape(n, n)
                P_candidate = np.clip(P_candidate, 0, None)
                s = P_candidate.sum()
                if s > 1e-12:
                    P_candidate /= s
                    score, _ = _objective_and_grad(P_candidate.flatten())
                    score = -score
                    if score > best_score:
                        best_score = score
                        best_P = P_candidate.copy()
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
