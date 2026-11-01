"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_sq_dists(P):
    """Compute squared pairwise distances for points P of shape (n, d)."""
    # P: (n, d)
    # dist_sq[i,j] = ||P[i] - P[j]||^2
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)

def _score(P):
    """Compute (max/min pairwise distance)^2."""
    d2 = _pairwise_sq_dists(P)
    # Get upper triangle, exclude diagonal
    iu = np.triu_indices(N, k=1)
    vals = d2[iu]
    return (vals.max() / vals.min()) ** 2

def _soft_max_min_ratio(P, temperature=0.1):
    """Smooth surrogate for max/min ratio using log-sum-exp."""
    d2 = _pairwise_sq_dists(P)
    iu = np.triu_indices(N, k=1)
    vals = d2[iu]
    # soft max: temperature * log(sum(exp(vals/T)))
    v_max = vals.max()
    soft_max = v_max + temperature * np.log(np.sum(np.exp((vals - v_max) / temperature)))
    v_min = vals.min()
    # soft min: -temperature * log(sum(exp(-vals/T)))
    soft_min = v_min - temperature * np.log(np.sum(np.exp(-(vals - v_min) / temperature)))
    if soft_min <= 0:
        return 1e10
    ratio = soft_max / soft_min
    return ratio ** 2

def _objective(P_flat, temperature=0.1):
    P = P_flat.reshape(N, D)
    return _soft_max_min_ratio(P, temperature)

def _grad_objective(P_flat, temperature=0.1):
    P = P_flat.reshape(N, D)
    d2 = _pairwise_sq_dists(P)
    iu = np.triu_indices(N, k=1)
    vals = d2[iu]
    
    v_max = vals.max()
    exp_vals = np.exp((vals - v_max) / temperature)
    sum_exp = np.sum(exp_vals)
    soft_max = v_max + temperature * np.log(sum_exp)
    
    v_min = vals.min()
    exp_vals_neg = np.exp(-(vals - v_min) / temperature)
    sum_exp_neg = np.sum(exp_vals_neg)
    soft_min = v_min - temperature * np.log(sum_exp_neg)
    
    if soft_min <= 0:
        return np.zeros_like(P_flat)
    
    ratio = soft_max / soft_min
    obj = ratio ** 2
    
    # d(obj)/d(soft_max) = 2 * ratio / soft_min
    d_obj_d_sm = 2 * ratio / soft_min
    # d(obj)/d(soft_min) = -2 * ratio * soft_max / soft_min^2
    d_obj_d_smin = -2 * ratio * soft_max / (soft_min ** 2)
    
    # d(soft_max)/d(vals_k) = exp_vals[k] / sum_exp
    d_sm_d_vals = exp_vals / sum_exp
    # d(soft_min)/d(vals_k) = -exp_vals_neg[k] / sum_exp_neg
    d_smin_d_vals = -exp_vals_neg / sum_exp_neg
    
    # d(obj)/d(vals_k) = d_obj_d_sm * d_sm_d_vals[k] + d_obj_d_smin * d_smin_d_vals[k]
    d_obj_d_vals = d_obj_d_sm * d_sm_d_vals + d_obj_d_smin * d_smin_d_vals
    
    # Now propagate gradient to P
    # d2[i,j] = sum_d (P[i,d] - P[j,d])^2
    # d(d2[i,j])/d(P[i,d]) = 2*(P[i,d] - P[j,d])
    # d(d2[i,j])/d(P[j,d]) = -2*(P[i,d] - P[j,d])
    
    grad = np.zeros((N, D))
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        coeff = 2 * d_obj_d_vals[k]
        diff = P[i] - P[j]
        grad[i] += coeff * diff
        grad[j] -= coeff * diff
    
    return grad.reshape(-1)

def _icosahedron_points():
    """Generate 12 icosahedron vertices."""
    phi = (1 + np.sqrt(5)) / 2
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _generate_candidates():
    """Generate multiple candidate starting configurations."""
    candidates = []
    
    # Candidate 1: icosahedron + 2 points at opposite poles of octahedron
    ico = _icosahedron_points()
    # Add 2 more points - try various positions
    for trial in range(10):
        extra = np.random.randn(2, D) * 0.5
        P = np.vstack([ico, extra])
        # Center
        P = P - P.mean(axis=0)
        candidates.append(P)
    
    # Candidate 2: Regular tetrahedron-based expansion
    # Start with octahedron (6 points) + 8 more
    oct_pts = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    for trial in range(10):
        extra = np.random.randn(8, D) * 0.3
        P = np.vstack([oct_pts, extra])
        P = P - P.mean(axis=0)
        candidates.append(P)
    
    # Candidate 3: Random on sphere
    for trial in range(20):
        P = np.random.randn(N, D)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        P = P - P.mean(axis=0)
        candidates.append(P)
    
    # Candidate 4: Grid-based
    side = int(np.ceil(N ** (1/D)))
    import itertools
    grid = list(itertools.product(range(side), repeat=D))[:N]
    P = np.array(grid, dtype=float)
    P = P - P.mean(axis=0)
    candidates.append(P)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_P = None
    best_score = np.inf
    
    candidates = _generate_candidates()
    
    for idx, P0 in enumerate(candidates):
        if time.time() > deadline:
            break
        
        P0_flat = P0.reshape(-1)
        
        # Try with different temperatures for soft surrogate
        for temp in [0.05, 0.1, 0.2, 0.5]:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _objective, P0_flat, 
                    args=(temp,),
                    method='L-BFGS-B',
                    jac=lambda x, t=temp: _grad_objective(x, t),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                P_opt = result.x.reshape(N, D)
                s = _score(P_opt)
                
                if s < best_score:
                    best_score = s
                    best_P = P_opt.copy()
                    
            except Exception:
                continue
    
    if best_P is None:
        # Fallback: return grid
        import itertools
        side = int(np.ceil(N ** (1/D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_P = np.array(grid, dtype=float)
    
    # Final polish: try a few more iterations with tighter tolerance
    if time.time() < deadline - 5:
        try:
            P_flat = best_P.reshape(-1)
            result = minimize(
                _objective, P_flat,
                args=(0.01,),
                method='L-BFGS-B',
                jac=lambda x, t=0.01: _grad_objective(x, t),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            P_polished = result.x.reshape(N, D)
            s = _score(P_polished)
            if s < best_score:
                best_P = P_polished
                best_score = s
        except Exception:
            pass
    
    return best_P
# EVOLVE-BLOCK-END
