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
    proj = np.bincount(inverse.flatten(), weights=P.flatten(), minlength=num_bins)
    return proj

def _entropy_and_gradient(P, r):
    """
    Compute Shannon entropy H and its gradient with respect to P for projection r.
    P is a flattened array of shape (100,).
    """
    proj = _compute_proj_probs(P, r)
    
    mask = proj > 1e-15
    if not np.any(mask):
        return 0.0, np.zeros_like(P)
    
    p = proj[mask]
    H = float(-np.sum(p * np.log2(p)))
    
    grad_proj = np.zeros_like(proj)
    grad_proj[mask] = -(1.0 / np.log(2.0)) * (1.0 + np.log2(p))
    
    inverse = PROJ_DATA[r]['inverse']
    grad_P = grad_proj[inverse.flatten()]
    
    return H, grad_P

def _ratio_from_P(P):
    """Compute D/M ratio from a valid probability array."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return 0.0
    P_norm = P / s
    
    H_m1, _ = _entropy_and_gradient(P_norm.flatten(), -1.0)
    H_0, _ = _entropy_and_gradient(P_norm.flatten(), 0.0)
    H_1, _ = _entropy_and_gradient(P_norm.flatten(), 1.0)
    H_2, _ = _entropy_and_gradient(P_norm.flatten(), 2.0)
    
    D = H_m1
    M = max(H_0, H_1, H_2)
    
    if M < 1e-9 or D < 1e-12:
        return 0.0
    
    return D / M

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

def _coordinate_descent(P, max_iters=500, tol=1e-10, deadline=None):
    """
    Coordinate descent refinement: adjust each cell's weight to improve D/M.
    For each cell, try to find the optimal value that maximizes the ratio.
    """
    n = P.shape[0]
    P = P.copy()
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return P
    
    P /= s
    best_score = _ratio_from_P(P)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for y in range(n):
            for x in range(n):
                if P[y, x] < 1e-15:
                    continue
                
                old_val = P[y, x]
                # Try a few candidate values around the current one
                candidates = [0.0]
                if old_val > 0:
                    candidates.extend([old_val * 0.5, old_val * 0.8, old_val * 1.2, old_val * 1.5])
                candidates.extend([1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1])
                
                best_cand = old_val
                best_cand_score = best_score
                
                for cand in candidates:
                    if cand < 0 or cand > 1:
                        continue
                    P[y, x] = cand
                    s = P.sum()
                    if s < 1e-12:
                        P[y, x] = old_val
                        continue
                    P_norm = P / s
                    score = _ratio_from_P(P_norm)
                    if score > best_cand_score + 1e-12:
                        best_cand_score = score
                        best_cand = cand
                
                if best_cand != old_val:
                    P[y, x] = best_cand
                    s = P.sum()
                    if s > 1e-12:
                        P /= s
                        best_score = best_cand_score
                        improved = True
        
        if not improved:
            break
    
    return P

def _pairwise_descent(P, max_iters=100, deadline=None):
    """
    Pairwise coordinate descent: try swapping/adjusting pairs of cells.
    This can escape local optima that single-cell updates miss.
    """
    n = P.shape[0]
    P = P.copy()
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return P
    
    P /= s
    best_score = _ratio_from_P(P)
    
    # Precompute all cell indices
    cells = [(y, x) for y in range(n) for x in range(n)]
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try random pairs
        for _ in range(50):
            if deadline is not None and time.time() > deadline:
                break
            i, j = rng.choice(len(cells), size=2, replace=False)
            y1, x1 = cells[i]
            y2, x2 = cells[j]
            
            v1 = P[y1, x1]
            v2 = P[y2, x2]
            
            if v1 < 1e-15 and v2 < 1e-15:
                continue
            
            # Try several transfer amounts
            for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                # Move frac of v1 to v2
                new_v1 = v1 * (1 - frac)
                new_v2 = v2 + v1 * frac
                
                P[y1, x1] = new_v1
                P[y2, x2] = new_v2
                s = P.sum()
                if s < 1e-12:
                    P[y1, x1] = v1
                    P[y2, x2] = v2
                    continue
                P_norm = P / s
                score = _ratio_from_P(P_norm)
                if score > best_score + 1e-12:
                    best_score = score
                    P /= s
                    improved = True
                    break
                else:
                    P[y1, x1] = v1
                    P[y2, x2] = v2
            
            # Also try reverse direction
            for frac in [0.1, 0.3, 0.5]:
                new_v1 = v1 + v2 * frac
                new_v2 = v2 * (1 - frac)
                
                P[y1, x1] = new_v1
                P[y2, x2] = new_v2
                s = P.sum()
                if s < 1e-12:
                    P[y1, x1] = v1
                    P[y2, x2] = v2
                    continue
                P_norm = P / s
                score = _ratio_from_P(P_norm)
                if score > best_score + 1e-12:
                    best_score = score
                    P /= s
                    improved = True
                    break
                else:
                    P[y1, x1] = v1
                    P[y2, x2] = v2
        
        if not improved:
            break
    
    return P

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
    
    # 4. Concentrated on parallel lines with slope -1 (all offsets)
    for offset in range(-n+1, n):
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
    
    # 5. Horizontal lines (slope 0)
    for y0 in range(0, n, 2):
        P = np.zeros((n, n))
        P[y0, :] = 1.0
        P /= P.sum()
        patterns.append(P)
    
    # 6. Diagonal lines (slope 1)
    for offset in range(-n+1, n, 2):
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
    
    # 7. Lines with slope 2
    for offset in range(-2*n+2, 2*n-1, 2):
        P = np.zeros((n, n))
        count = 0
        for y in range(n):
            x = 2 * y + offset
            if 0 <= x < n:
                P[y, x] = 1.0
                count += 1
        if count > 0:
            P /= count
            patterns.append(P.copy())
    
    # 8. Random sparse
    rng = np.random.RandomState(42)
    for _ in range(10):
        P = np.zeros((n, n))
        k = rng.randint(5, 20)
        indices = rng.choice(n * n, size=k, replace=False)
        P.flatten()[indices] = 1.0
        P /= P.sum()
        patterns.append(P)
        
    # 9. Random dense
    for _ in range(10):
        P = rng.rand(n, n)
        P /= P.sum()
        patterns.append(P)
        
    # 10. Block patterns
    for size in [2, 3, 4, 5]:
        for start_y in range(0, n - size + 1, 2):
            for start_x in range(0, n - size + 1, 2):
                P = np.zeros((n, n))
                P[start_y:start_y+size, start_x:start_x+size] = 1.0
                P /= P.sum()
                patterns.append(P)
                
    # 11. Two-line patterns (slope -1)
    for offset1 in range(-n+1, n, 2):
        for offset2 in range(offset1 + 2, n, 2):
            P = np.zeros((n, n))
            count = 0
            for y in range(n):
                for off in [offset1, offset2]:
                    x = y + off
                    if 0 <= x < n:
                        P[y, x] = 1.0
                        count += 1
            if count > 0:
                P /= count
                patterns.append(P.copy())
    
    # 12. Three-line patterns (slope -1)
    for offset1 in range(-n+1, n, 3):
        for offset2 in range(offset1 + 3, n, 3):
            for offset3 in range(offset2 + 3, n, 3):
                P = np.zeros((n, n))
                count = 0
                for y in range(n):
                    for off in [offset1, offset2, offset3]:
                        x = y + off
                        if 0 <= x < n:
                            P[y, x] = 1.0
                            count += 1
                if count > 0:
                    P /= count
                    patterns.append(P.copy())
    
    # 13. Checkerboard patterns
    for phase in [0, 1]:
        P = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x + y + phase) % 2 == 0:
                    P[y, x] = 1.0
        P /= P.sum()
        patterns.append(P)
    
    # 14. Concentrate on specific anti-diagonals to maximize H_{-1}
    # Use multiple anti-diagonals with equal weight
    for num_diags in [3, 4, 5, 6]:
        for start in range(0, 2*n-1 - num_diags + 1, 2):
            P = np.zeros((n, n))
            count = 0
            for d in range(start, start + num_diags):
                for y in range(n):
                    x = y + d
                    if 0 <= x < n:
                        P[y, x] = 1.0
                        count += 1
            if count > 0:
                P /= count
                patterns.append(P.copy())
    
    # 15. 2D lattice patterns: select cells based on modular conditions
    # These can create structured distributions with high H_{-1} but controlled H_0, H_1
    for mod_x in [2, 3, 4, 5]:
        for mod_y in [2, 3, 4, 5]:
            if mod_x == mod_y:
                continue
            for phase_x in range(mod_x):
                for phase_y in range(mod_y):
                    P = np.zeros((n, n))
                    count = 0
                    for y in range(n):
                        for x in range(n):
                            if x % mod_x == phase_x and y % mod_y == phase_y:
                                P[y, x] = 1.0
                                count += 1
                    if count >= 3:
                        P /= count
                        patterns.append(P.copy())
    
    # 16. Weighted anti-diagonals with varying weights to control M
    for num_diags in [4, 5, 6, 7, 8]:
        for start in range(0, 2*n-1 - num_diags + 1):
            P = np.zeros((n, n))
            count = 0
            for di, d in enumerate(range(start, start + num_diags)):
                # Vary weight by diagonal index
                w = 1.0 + 0.1 * di
                for y in range(n):
                    x = y + d
                    if 0 <= x < n:
                        P[y, x] = w
                        count += 1
            if count > 0 and P.sum() > 0:
                P /= P.sum()
                patterns.append(P.copy())
    
    # 17. Sparse random with specific structure (few cells per row/column)
    for seed in range(5, 15):
        rng2 = np.random.RandomState(seed)
        P = np.zeros((n, n))
        # Select 1-2 cells per row with some correlation
        for y in range(n):
            num_cells = rng2.randint(1, 3)
            xs = rng2.choice(n, size=num_cells, replace=False)
            for x in xs:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            patterns.append(P.copy())
    
    return patterns

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 580
    
    initial_patterns = _make_initial_patterns(n)
    
    # Evaluate all initial patterns and pick the best few
    scored_patterns = []
    for i, P in enumerate(initial_patterns):
        score = _ratio_from_P(P)
        scored_patterns.append((score, i, P))
    
    scored_patterns.sort(key=lambda x: x[0], reverse=True)
    
    best_P = scored_patterns[0][2].copy()
    best_score = scored_patterns[0][0]
    
    constraints = [{'type': 'eq', 'fun': lambda P: np.sum(P) - 1.0, 'jac': lambda P: np.ones_like(P)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    # SLSQP from top initial patterns
    num_starts = min(len(initial_patterns), 25)
    
    for i in range(num_starts):
        if time.time() > deadline - 120:
            break
        
        score, idx, P_init = scored_patterns[i]
        x0 = P_init.flatten().copy()
        
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-12, 'disp': False}
            )
            
            P_candidate = result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score = _ratio_from_P(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
                    
        except Exception:
            continue
    
    # Pairwise descent (new: escapes single-cell local optima)
    if time.time() < deadline - 80:
        P_pw = _pairwise_descent(best_P, max_iters=50, deadline=deadline - 80)
        score = _ratio_from_P(P_pw)
        if score > best_score:
            best_score = score
            best_P = P_pw.copy()
    
    # Coordinate descent refinement
    if time.time() < deadline - 60:
        P_cd = _coordinate_descent(best_P, max_iters=200, deadline=deadline - 60)
        score = _ratio_from_P(P_cd)
        if score > best_score:
            best_score = score
            best_P = P_cd.copy()
    
    # Differential Evolution global search (handles non-smooth max in M)
    if time.time() < deadline - 40:
        try:
            from scipy.optimize import differential_evolution
            def _de_objective(x_flat):
                P = x_flat.reshape(n, n)
                P = np.clip(P, 0, None)
                s = P.sum()
                if s < 1e-12:
                    return 1e6
                P_norm = P / s
                return -_ratio_from_P(P_norm)
            
            de_result = differential_evolution(
                _de_objective,
                bounds=[(0.0, 1.0)] * (n * n),
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                polish=True,
                update_interval=5
            )
            
            P_candidate = de_result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score = _ratio_from_P(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish (derivative-free, handles non-smoothness)
    if time.time() < deadline - 20:
        rng = np.random.RandomState(123)
        for _ in range(8):
            if time.time() > deadline - 20:
                break
            
            noise_scale = 0.03
            noise = rng.randn(n, n) * noise_scale
            P_start = best_P + noise
            P_start = np.clip(P_start, 0, None)
            s = P_start.sum()
            if s > 1e-12:
                P_start /= s
            else:
                continue
            
            try:
                result = minimize(
                    lambda x: -_ratio_from_P(x.reshape(n, n)),
                    P_start.flatten().copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                
                P_candidate = result.x.reshape(n, n)
                P_candidate = np.clip(P_candidate, 0, None)
                s = P_candidate.sum()
                if s > 1e-12:
                    P_candidate /= s
                    score = _ratio_from_P(P_candidate)
                    if score > best_score:
                        best_score = score
                        best_P = P_candidate.copy()
            except Exception:
                continue
    
    # Final pairwise + coordinate descent + SLSQP polish
    if time.time() < deadline - 15:
        P_pw = _pairwise_descent(best_P, max_iters=30, deadline=deadline - 15)
        score = _ratio_from_P(P_pw)
        if score > best_score:
            best_score = score
            best_P = P_pw.copy()
    
    if time.time() < deadline - 10:
        P_cd = _coordinate_descent(best_P, max_iters=100, deadline=deadline - 10)
        score = _ratio_from_P(P_cd)
        if score > best_score:
            best_score = score
            best_P = P_cd.copy()
    
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective_and_grad,
                best_P.flatten().copy(),
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            
            P_candidate = result.x.reshape(n, n)
            P_candidate = np.clip(P_candidate, 0, None)
            s = P_candidate.sum()
            if s > 1e-12:
                P_candidate /= s
                score = _ratio_from_P(P_candidate)
                if score > best_score:
                    best_score = score
                    best_P = P_candidate.copy()
        except Exception:
            pass
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-12:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
