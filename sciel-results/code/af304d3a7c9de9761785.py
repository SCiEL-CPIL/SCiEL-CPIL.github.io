"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        r = 0.1
        rows = 5
        for row in range(rows):
            y = 0.1 + row * r * np.sqrt(3)
            if row % 2 == 0:
                xs = np.arange(5) * 2 * r + r
            else:
                xs = np.arange(5) * 2 * r + 2 * r
            for col, x in enumerate(xs):
                idx = row * 5 + col
                if idx < N:
                    pos[idx] = [x, y]
                    radii[idx] = r
        return pos, radii
    
    elif strategy == 'center_ring':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        pos[0] = [0.5, 0.5]
        radii[0] = 0.15
        r_inner = 0.12
        dist_inner = 0.3
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
            radii[i+1] = r_inner
        r_outer = 0.08
        dist_outer = 0.42
        for i in range(17):
            angle = 2 * np.pi * i / 17
            pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
            radii[i+9] = r_outer
        return pos, radii
    
    elif strategy == 'random':
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.98)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.98))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.98))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.98))
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = (dist / (min_dist + 1e-12)) * 0.98
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _smooth_objective(x, beta=50.0, smooth_eps=0.005):
    """Smooth objective using log-sum-exp for soft-min of violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Collect all violation terms
    violations = []
    
    # Boundary violations
    for i in range(n):
        for coord in range(2):
            v_left = radii[i] - pos[i, coord]
            v_right = pos[i, coord] - (1.0 - radii[i])
            violations.append(v_left)
            violations.append(v_right)
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if min_dist > 0:
                # Smooth approximation: violation = max(0, min_dist - dist)
                # Use soft approximation: min_dist - sqrt(dist_sq + eps^2) + eps
                dist_smooth = np.sqrt(dist_sq + smooth_eps * smooth_eps)
                v = min_dist - dist_smooth + smooth_eps
                violations.append(v)
    
    # Log-sum-exp for smooth max of violations
    if violations:
        arr = np.array(violations)
        max_v = np.max(arr)
        if max_v > 1e-10:
            # Soft max via log-sum-exp
            penalty = max_v + smooth_eps * np.log(np.sum(np.exp((arr - max_v) / smooth_eps)))
        else:
            penalty = 0.0
    else:
        penalty = 0.0
    
    return -np.sum(radii) + beta * penalty


def _qp_objective_and_gradient(x, pos_ref, radii_ref):
    """
    Build QP objective and gradient for sequential QP.
    Linearize distance constraints around (pos_ref, radii_ref).
    Variables: x = [dx_0, dy_0, ..., dx_{n-1}, dy_{n-1}, dr_0, ..., dr_{n-1}]
    """
    n = N
    pos = pos_ref + x[:2*n].reshape(n, 2)
    radii = radii_ref + x[2*n:]
    
    # Objective: -sum(radii)
    obj = -np.sum(radii)
    
    # Gradient of objective
    grad = np.zeros(3 * n)
    grad[2*n:] = -1.0
    
    # Add linearized constraint penalties to make it a QP
    # For each pair, linearize: ||pos_i - pos_j|| >= r_i + r_j
    # Around reference: dist_ref + grad_dist . dx >= r_i_ref + r_j_ref + dr_i + dr_j
    
    penalty = 0.0
    penalty_grad = np.zeros(3 * n)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx_ref = pos_ref[i, 0] - pos_ref[j, 0]
            dy_ref = pos_ref[i, 1] - pos_ref[j, 1]
            dist_ref = np.sqrt(dx_ref * dx_ref + dy_ref * dy_ref)
            
            if dist_ref < 1e-8:
                continue
            
            # Gradient of distance w.r.t. positions
            gxi = dx_ref / dist_ref
            gyi = dy_ref / dist_ref
            gxj = -dx_ref / dist_ref
            gyj = -dy_ref / dist_ref
            
            # Current distance (linearized)
            dx_curr = (pos_ref[i, 0] + x[2*i]) - (pos_ref[j, 0] + x[2*j])
            dy_curr = (pos_ref[i, 1] + x[2*i+1]) - (pos_ref[j, 1] + x[2*j+1])
            dist_curr = np.sqrt(dx_curr * dx_curr + dy_curr * dy_curr)
            
            min_dist_curr = radii[i] + radii[j]
            
            violation = min_dist_curr - dist_curr
            if violation > 0:
                penalty += violation * violation
                # Gradient of penalty
                dpen_ddist = -2 * violation
                dpen_dxi = dpen_ddist * (dx_curr / dist_curr) if dist_curr > 1e-8 else 0
                dpen_dyi = dpen_ddist * (dy_curr / dist_curr) if dist_curr > 1e-8 else 0
                dpen_dxj = dpen_ddist * (-dx_curr / dist_curr) if dist_curr > 1e-8 else 0
                dpen_dyj = dpen_ddist * (-dy_curr / dist_curr) if dist_curr > 1e-8 else 0
                dpen_dri = 2 * violation
                dpen_drj = 2 * violation
                
                penalty_grad[2*i] += dpen_dxi
                penalty_grad[2*i+1] += dpen_dyi
                penalty_grad[2*j] += dpen_dxj
                penalty_grad[2*j+1] += dpen_dyj
                penalty_grad[2*n+i] += dpen_dri
                penalty_grad[2*n+j] += dpen_drj
    
    # Boundary penalties (quadratic)
    for i in range(n):
        for coord in range(2):
            v_left = radii[i] - pos[i, coord]
            v_right = pos[i, coord] - (1.0 - radii[i])
            if v_left > 0:
                penalty += v_left * v_left
                penalty_grad[2*n+i] += 2 * v_left
                penalty_grad[2*i+coord] += -2 * v_left
            if v_right > 0:
                penalty += v_right * v_right
                penalty_grad[2*n+i] += 2 * v_right
                penalty_grad[2*i+coord] += -2 * v_right
    
    total_obj = obj + 50.0 * penalty
    total_grad = grad + 50.0 * penalty_grad
    
    return total_obj, total_grad


def _run_sequential_qp(pos, radii, n_substeps=5, deadline=None):
    """
    Sequential QP: solve a sequence of convex QP subproblems.
    Each subproblem linearizes distance constraints around current config.
    """
    pos_curr = pos.copy()
    radii_curr = radii.copy()
    best_pos = pos_curr.copy()
    best_radii = radii_curr.copy()
    best_score = np.sum(radii_curr)
    
    for step in range(n_substeps):
        if deadline is not None and time.time() > deadline:
            break
        
        x0 = np.zeros(3 * N)
        
        # Use SLSQP with gradient for the QP subproblem
        bounds = []
        for i in range(N):
            # Position bounds: keep within [0.005, 0.995]
            bounds.append((0.005 - pos_curr[i, 0], 0.995 - pos_curr[i, 0]))
            bounds.append((0.005 - pos_curr[i, 1], 0.995 - pos_curr[i, 1]))
            # Radius bounds: keep positive and reasonable
            bounds.append((-radii_curr[i] * 0.5, 0.1))
        
        try:
            result = minimize(
                fun=lambda x: _qp_objective_and_gradient(x, pos_curr, radii_curr)[0],
                x0=x0,
                jac=lambda x: _qp_objective_and_gradient(x, pos_curr, radii_curr)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < _qp_objective_and_gradient(x0, pos_curr, radii_curr)[0]:
                x_opt = result.x
                pos_new = pos_curr + x_opt[:2*N].reshape(N, 2)
                radii_new = radii_curr + x_opt[2*N:]
                
                # Ensure radii are positive
                radii_new = np.maximum(radii_new, 0.001)
                
                # Post-process to ensure validity
                pos_pp, radii_pp = _post_process(pos_new, radii_new)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
                    pos_curr = pos_pp
                    radii_curr = radii_pp
        except Exception:
            break
    
    return best_pos, best_radii


def _run_smooth_optimization(x0, deadline):
    """Run L-BFGS-B with smooth penalty objective."""
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))
    
    try:
        result = minimize(
            _smooth_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < _smooth_objective(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return None, None


def _force_directed_anneal(pos, radii, n_iter=800, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = 0.015 * np.exp(-0.003 * it)
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            margin = 0.005
            if pos[i, 0] < radii[i] + margin:
                forces[i, 0] += 10.0 * (radii[i] + margin - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - margin:
                forces[i, 0] -= 10.0 * (pos[i, 0] - (1.0 - radii[i] - margin))
            if pos[i, 1] < radii[i] + margin:
                forces[i, 1] += 10.0 * (radii[i] + margin - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - margin:
                forces[i, 1] -= 10.0 * (pos[i, 1] - (1.0 - radii[i] - margin))
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 15.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.3 and dist > 1e-6:
                    force_mag = 2.0 * (min_dist * 1.3 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.85 * vel + 0.15 * forces
        pos += vel * 0.008 + rng.normal(0, temp, size=pos.shape)
        
        # Clip positions
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Grow radii where possible
        for i in range(N):
            max_grow = min(pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0005:
                radii[i] += max_grow * 0.6
        
        # Ensure minimum radius
        radii = np.maximum(radii, 0.002)
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _polish(pos, radii, deadline=None):
    """High-precision polish using sequential QP then smooth optimization."""
    # Try sequential QP
    if deadline is None:
        deadline = time.time() + 20
    
    pos_qp, radii_qp = _run_sequential_qp(pos, radii, n_substeps=8, deadline=deadline - 5)
    if _is_valid(pos_qp, radii_qp) and np.sum(radii_qp) > np.sum(radii):
        pos = pos_qp
        radii = radii_qp
    
    # Try smooth optimization from improved state
    x0 = np.concatenate([pos.flatten(), radii])
    result_pos, result_radii = _run_smooth_optimization(x0, deadline)
    if result_pos is not None and np.sum(result_radii) > np.sum(radii):
        return result_pos, result_radii
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    
    # Phase 1: Multi-start with smooth optimization
    for strategy in strategies:
        if time.time() > deadline - 60:
            break
        
        for seed in range(3):
            if time.time() > deadline - 55:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Try smooth optimization
            result_pos, result_radii = _run_smooth_optimization(x0, deadline)
            if result_pos is not None:
                score = np.sum(result_radii)
                if score > best_score:
                    best_score = score
                    best_pos = result_pos.copy()
                    best_radii = result_radii.copy()
    
    # Phase 2: Sequential QP from best result
    if best_pos is not None and time.time() < deadline - 35:
        pos_qp, radii_qp = _run_sequential_qp(best_pos, best_radii, n_substeps=10, deadline=deadline - 30)
        if _is_valid(pos_qp, radii_qp) and np.sum(radii_qp) > best_score:
            best_score = np.sum(radii_qp)
            best_pos = pos_qp.copy()
            best_radii = radii_qp.copy()
    
    # Phase 3: Force-directed annealing
    if best_pos is not None and time.time() < deadline - 25:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=800, deadline=deadline - 20)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 4: Another sequential QP pass after annealing
    if best_pos is not None and time.time() < deadline - 15:
        pos_qp2, radii_qp2 = _run_sequential_qp(best_pos, best_radii, n_substeps=8, deadline=deadline - 10)
        if _is_valid(pos_qp2, radii_qp2) and np.sum(radii_qp2) > best_score:
            best_score = np.sum(radii_qp2)
            best_pos = pos_qp2.copy()
            best_radii = radii_qp2.copy()
    
    # Phase 5: Final polish
    if best_pos is not None and time.time() < deadline - 5:
        pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 2)
        score_pol = np.sum(radii_pol)
        if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
            best_score = score_pol
            best_pos = pos_pol.copy()
            best_radii = radii_pol.copy()
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
