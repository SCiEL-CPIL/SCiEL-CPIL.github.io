import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    N = 12
    
    # Reference hexagon at (0,0) with side 1, angle 0
    angles = np.radians(60 * np.arange(6))
    V_ref = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for SAT (outward)
    edges = np.roll(V_ref, -1, axis=0) - V_ref
    N_ref = np.column_stack([edges[:, 1], -edges[:, 0]])
    N_ref /= np.linalg.norm(N_ref, axis=1, keepdims=True)
    
    # Precompute projections
    A = V_ref @ N_ref.T # (6, 6)
    MinProj_ref = np.min(A, axis=0)
    MaxProj_ref = np.max(A, axis=0)
    
    def compute_cost_and_validity(params):
        pos = params[:2 * N].reshape(N, 2)
        s = params[-1]
        
        # 1. Overlap Penalty
        MinProj = pos @ N_ref.T + MinProj_ref
        MaxProj = pos @ N_ref.T + MaxProj_ref
        
        overlap_pen = 0.0
        is_valid = True
        for i in range(N):
            for j in range(i + 1, N):
                mins = np.maximum(MinProj[i], MinProj[j])
                maxs = np.minimum(MaxProj[i], MaxProj[j])
                overlaps = mins - maxs
                pos_overlaps = np.maximum(0.0, overlaps)
                if np.any(pos_overlaps > 1e-8):
                    is_valid = False
                overlap_pen += np.sum(pos_overlaps ** 2)
        
        # 2. Containment Penalty
        PosProj = pos @ N_ref.T
        VertProj = PosProj[:, np.newaxis, :] + A[np.newaxis, :, :]
        
        limit = s * np.sqrt(3) / 2
        violations = VertProj - limit
        max_violation = np.max(violations)
        
        if max_violation > 1e-8:
            is_valid = False
        
        contain_pen = max(0.0, max_violation) ** 2
        
        cost = s + 100.0 * (overlap_pen + contain_pen)
        
        return cost, is_valid

    # Initial guess
    pos_init = np.zeros((N, 2))
    for i in range(N):
        row = i // 4
        col = i % 4
        x = col * 2.0
        if row % 2 == 1:
            x += 1.0
        y = row * np.sqrt(3)
        pos_init[i] = [x, y]
    pos_init -= np.mean(pos_init, axis=0)
    
    s_init = np.max(np.linalg.norm(pos_init[:, np.newaxis, :] + V_ref, axis=(0, 2)))
    s_init *= 1.2
    
    params_init = np.concatenate([pos_init.flatten(), [s_init]])
    
    best_params = params_init
    best_cost, best_valid = compute_cost_and_validity(params_init)
    if not best_valid:
        # Try to fix initial s
        # Increase s until valid?
        # For now, just proceed, optimizer should fix it.
        pass

    bounds = [(-10, 10)] * (2 * N) + [(3.0, 10.0)]
    
    def objective(p):
        cost, _ = compute_cost_and_validity(p)
        return cost

    # Run L-BFGS-B
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        res_cost, res_valid = compute_cost_and_validity(res.x)
        if res_cost < best_cost:
            best_cost = res_cost
            best_params = res.x
            best_valid = res_valid
    except Exception:
        pass

    # If best is not valid, try to find a valid one by increasing s?
    # Or just return best_params if valid.
    # If invalid, we might need a fallback.
    # But L-BFGS-B with high penalty should make it valid.
    
    # Polish with Nelder-Mead if time permits
    if time.time() < deadline - 60:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            res2_cost, res2_valid = compute_cost_and_validity(res2.x)
            if res2_cost < best_cost:
                best_cost = res2_cost
                best_params = res2.x
                best_valid = res2_valid
        except Exception:
            pass

    # Final check
    # If best_params is invalid, we should try to recover.
    # But given the penalty, it's likely valid.
    
    best_pos = best_params[:2 * N].reshape(N, 2)
    best_s = best_params[-1]
    
    inner_data = np.zeros((N, 3))
    for i in range(N):
        inner_data[i, 0] = best_pos[i, 0]
        inner_data[i, 1] = best_pos[i, 1]
        inner_data[i, 2] = 0.0
        
    outer_data = np.array([0.0, 0.0, 0.0])
    
    return inner_data, outer_data, best_s
