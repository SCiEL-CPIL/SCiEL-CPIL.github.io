"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Contact Graph / Force-Directed Relaxation.
1. Initialize with hexagonal lattice.
2. Build contact graph based on proximity.
3. Iteratively solve for positions and radii by minimizing a force potential 
   (repulsion for overlaps, attraction for contacts) using Newton-Raphson.
4. Polish with SLSQP hard constraints."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        # Boundary check
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink the smaller one more, or both proportionally
                    # To preserve sum, shrink both slightly
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # Try to fit a hexagonal pattern
    # 5x5 grid is 25, we need 26.
    # Let's use a slightly denser hexagonal packing
    r = 0.10
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    
    # If we didn't fill N, add random circles
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.9
            k += 1
        else:
            # Force it in a corner or edge if random fails
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _force_relaxation(pos, radii, max_iters=5000, tol=1e-6):
    """
    Force-directed relaxation.
    We treat the system as a set of particles.
    Forces:
    1. Repulsion if dist < r_i + r_j (overlap)
    2. Attraction if dist > r_i + r_j (to maintain packing density) - actually, we want to maximize radii, 
       so we want to push them apart to allow larger radii? 
       No, standard approach: fix positions, maximize radii? 
       Or fix radii, adjust positions?
       
       Better: Joint optimization.
       Objective: Minimize Sum( (r_i + r_j - dist_ij)^2 ) for overlapping pairs? 
       Or Minimize Potential Energy where V = Sum_{i<j} max(0, r_i + r_j - dist_ij)^2 + Boundary_Penalty.
       And we want to maximize Sum(r_i).
       
       Let's use a penalty method:
       Minimize: -Sum(r_i) + Lambda * Sum( max(0, r_i + r_j - dist_ij)^2 ) + Lambda * Boundary_Violation
       
       Gradient Descent with Adaptive Step Size (Simulated Annealing style but smoother).
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Parameters
    lambda_overlap = 100.0
    lambda_boundary = 100.0
    step_size = 0.01
    decay = 0.999
    min_step = 1e-5
    
    for it in range(max_iters):
        # Compute gradients
        grad_pos = np.zeros_like(pos)
        grad_radii = np.zeros_like(radii)
        
        # Boundary forces
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Left/Right
            if x < r:
                # Push right, shrink r? 
                # Force on x: +lambda
                # Force on r: -lambda (if we shrink r, we reduce violation)
                # Actually, let's just project or use gradient of penalty.
                # Penalty: (r - x)^2 if x < r. dP/dr = 2(r-x), dP/dx = -2(r-x)
                viol = r - x
                if viol > 0:
                    grad_radii[i] += 2 * lambda_boundary * viol
                    grad_pos[i, 0] -= 2 * lambda_boundary * viol
            if 1.0 - x < r:
                viol = r - (1.0 - x)
                if viol > 0:
                    grad_radii[i] += 2 * lambda_boundary * viol
                    grad_pos[i, 0] += 2 * lambda_boundary * viol
                    
            # Bottom/Top
            if y < r:
                viol = r - y
                if viol > 0:
                    grad_radii[i] += 2 * lambda_boundary * viol
                    grad_pos[i, 1] -= 2 * lambda_boundary * viol
            if 1.0 - y < r:
                viol = r - (1.0 - y)
                if viol > 0:
                    grad_radii[i] += 2 * lambda_boundary * viol
                    grad_pos[i, 1] += 2 * lambda_boundary * viol
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r:
                    # Overlap
                    # Penalty: (sum_r - dist)^2
                    # dP/dr_i = 2(sum_r - dist)
                    # dP/dx_i = 2(sum_r - dist) * d(dist)/d(x_i) ... 
                    # dist = sqrt((xi-xj)^2 + ...). d(dist)/dx_i = (xi-xj)/dist
                    # So dP/dx_i = 2(sum_r - dist) * (xi-xj)/dist
                    # Note: (xi-xj)/dist is unit vector from j to i.
                    # Force on i should be away from j.
                    # Gradient of penalty wrt pos_i points towards j? 
                    # Let's check: P = (sum_r - dist)^2. 
                    # If we move i away from j, dist increases, P decreases. 
                    # So gradient of P wrt pos_i points towards j (increases P if we move that way).
                    # We want to move AGAINST gradient.
                    # So force on i = -grad_P = -2(sum_r - dist) * (pos_i - pos_j)/dist
                    # This points from i to j? No. (pos_i - pos_j)/dist is vector from j to i.
                    # So -grad points from i to j. That's wrong. We want to push i away from j.
                    # Let's re-evaluate.
                    # P = (sum_r - dist)^2. 
                    # dP/dx_i = 2(sum_r - dist) * (-1) * d(dist)/dx_i = -2(sum_r - dist) * (x_i - x_j)/dist.
                    # Gradient vector for pos_i is [ -2(sum_r - dist) * dx/dist, ... ]
                    # This vector points from i to j (since dx is x_i-x_j, and we have a negative sign, it points opposite to dx, i.e., towards j).
                    # So -Gradient points from j to i. Correct. It pushes i away from j.
                    
                    factor = 2 * lambda_overlap * (sum_r - dist) / dist
                    grad_pos[i, 0] -= factor * dx # -grad
                    grad_pos[i, 1] -= factor * dy
                    grad_pos[j, 0] += factor * dx # Gradient for j is opposite
                    grad_pos[j, 1] += factor * dy
                    
                    grad_radii[i] += 2 * lambda_overlap * (sum_r - dist)
                    grad_radii[j] += 2 * lambda_overlap * (sum_r - dist)
        
        # Objective gradient: -1 for radii
        grad_radii -= 1.0
        
        # Update
        pos -= step_size * grad_pos
        radii -= step_size * grad_radii
        
        # Enforce positivity
        radii = np.maximum(radii, 1e-5)
        
        # Project back to box? No, let penalty handle it, but keep them roughly inside for stability
        # If they fly out, step size is too big or lambda too small.
        
        # Decay step size
        step_size *= decay
        if step_size < min_step:
            break
            
    return pos, radii

def _slsqp_polish(pos, radii, deadline):
    """
    SLSQP with hard inequality constraints for non-overlap and boundaries.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.4))
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        # Use SLSQP
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
        if result.success or result.fun < objective(x0):
            x = result.x
            pos_out = x[:2*N].reshape(N, 2)
            radii_out = np.maximum(x[2*N:], 1e-4)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    pos, radii = _init_hexagonal()
    
    # Phase 1: Force Relaxation
    # Run multiple times with different initial perturbations or just one long run
    pos, radii = _force_relaxation(pos, radii, max_iters=2000)
    pos, radii = _post_process(pos, radii)
    
    current_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    # Phase 2: SLSQP Polish
    if time.time() < deadline - 20:
        pos_sl, rad_sl = _slsqp_polish(pos, radii, deadline)
        pos_pp, rad_pp = _post_process(pos_sl, rad_sl)
        score = np.sum(rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
    
    # Phase 3: Local Search / Restart if time permits
    # Try a few random restarts of force relaxation from perturbed best solution
    rng = np.random.RandomState(123)
    time_left = deadline - time.time()
    if time_left > 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            pos_p = best_pos + rng.normal(0, 0.01, (N, 2))
            rad_p = best_radii * (1 + rng.normal(0, 0.05, N))
            rad_p = np.maximum(rad_p, 1e-3)
            pos_p, rad_p = _post_process(pos_p, rad_p)
            
            # Force Relaxation
            pos_r, rad_r = _force_relaxation(pos_p, rad_p, max_iters=1000)
            pos_pp, rad_pp = _post_process(pos_r, rad_r)
            
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy
                    
                    # Polish the new best
                    if time.time() < deadline - 15:
                        pos_sl, rad_sl = _slsqp_polish(pos_pp, rad_pp, deadline)
                        pos_pp2, rad_pp2 = _post_process(pos_sl, rad_sl)
                        score2 = np.sum(rad_pp2)
                        if _is_valid(pos_pp2, rad_pp2) and score2 > best_score:
                            best_score = score2
                            best_pos = pos_pp2.copy()
                            best_radii = rad_pp2.copy()
    
    # Final Polish
    if time.time() < deadline - 5:
        pos_final, rad_final = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_final, rad_final)
        score = np.sum(rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
    
    # Final Validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos, radii = _init_hexagonal()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
