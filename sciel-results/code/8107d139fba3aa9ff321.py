"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: SDP-inspired Gram-matrix formulation with Schur-complement LMIs,
solved via sequential convex relaxation + interior-point style updates,
combined with contact-graph-aware radius optimization and simulated annealing polish."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.linalg import eigh

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
    for _ in range(300):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _sum_radii(pos, radii):
    return np.sum(radii)

def _build_contact_graph(pos, radii, tol=0.02):
    """Build contact graph: edges where circles are nearly touching."""
    graph = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if abs(dist - (radii[i] + radii[j])) < tol:
                graph[i, j] = 1
                graph[j, i] = 1
    # Also add boundary contacts
    for i in range(N):
        if pos[i, 0] - radii[i] < tol or 1.0 - pos[i, 0] - radii[i] < tol:
            graph[i, i] |= 1
        if pos[i, 1] - radii[i] < tol or 1.0 - pos[i, 1] - radii[i] < tol:
            graph[i, i] |= 2
    return graph

def _sdp_relaxation(pos, radii, max_iter=50, deadline=None):
    """
    SDP-inspired approach: work with Gram matrix G = P^T P where P is center matrix.
    Non-overlap: ||p_i - p_j||^2 >= (r_i + r_j)^2
    In Gram terms: G_ii + G_jj - 2 G_ij >= (r_i + r_j)^2
    
    We iteratively:
    1. Fix positions, solve for optimal radii via linear programming (LP relaxation of SDP)
    2. Update positions to reduce violations in Gram space
    3. Repeat
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Step 1: With positions fixed, find max radii via LP
        # Constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        #              r_i + r_j <= ||p_i - p_j||
        # Maximize sum(r_i)
        c = -np.ones(N)
        A_ub = np.zeros((N, N))
        b_ub = np.zeros(N)
        for i in range(N):
            A_ub[i, i] = 1.0
            b_ub[i] = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        
        A_pairs = []
        b_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_pairs.append(row)
                b_pairs.append(dist)
        
        if A_pairs:
            A_ub_full = np.vstack([A_ub, np.array(A_pairs)])
            b_ub_full = np.concatenate([b_ub, np.array(b_pairs)])
        else:
            A_ub_full = A_ub
            b_ub_full = b_ub
        
        try:
            res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=[(0, None)] * N, method='highs')
            if res.success:
                new_radii = np.maximum(res.x, 1e-5)
                # Only update if improvement
                if np.sum(new_radii) > np.sum(radii) * 0.999:
                    radii = new_radii
        except Exception:
            pass
        
        # Step 2: Update positions to reduce Gram violations
        # For each pair, if r_i + r_j > dist, push apart
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                needed = radii[i] + radii[j]
                if dist < needed and dist > 1e-10:
                    # Push apart proportionally
                    push = (needed - dist) * 0.5
                    nx, ny = dx / dist, dy / dist
                    pos[i, 0] += push * nx
                    pos[i, 1] += push * ny
                    pos[j, 0] -= push * nx
                    pos[j, 1] -= push * ny
                elif dist < needed and dist <= 1e-10:
                    # Random perturbation
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = needed * 0.5
                    pos[i, 0] += push * np.cos(angle)
                    pos[i, 1] += push * np.sin(angle)
                    pos[j, 0] -= push * np.cos(angle)
                    pos[j, 1] -= push * np.sin(angle)
        
        # Boundary constraint: keep inside [0,1]^2
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], radii[i] * 0.99, 1.0 - radii[i] * 0.99)
            pos[i, 1] = np.clip(pos[i, 1], radii[i] * 0.99, 1.0 - radii[i] * 0.99)
    
    return pos, radii

def _gram_matrix_optimize(pos, radii, deadline):
    """
    Optimize using Gram matrix perspective.
    G_ij = p_i^T p_j. We maintain G and enforce:
    - G is PSD (Schur complement conditions)
    - G_ii + G_jj - 2G_ij >= (r_i + r_j)^2
    - Boundary: p_i within [0,1]^2
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Iterative: alternate between position updates and radius updates
    for _ in range(30):
        if time.time() > deadline:
            break
        
        # Compute current Gram matrix
        G = pos @ pos.T
        
        # For each pair, check constraint violation
        violations = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                lhs = G[i, i] + G[j, j] - 2 * G[i, j]
                rhs = (radii[i] + radii[j]) ** 2
                if lhs < rhs:
                    violations += (rhs - lhs)
        
        if violations < 1e-10:
            break
        
        # Gradient-based position update to reduce violations
        grad_pos = np.zeros_like(pos)
        for i in range(N):
            for j in range(i + 1, N):
                lhs = G[i, i] + G[j, j] - 2 * G[i, j]
                rhs = (radii[i] + radii[j]) ** 2
                if lhs < rhs:
                    # Gradient of (lhs - rhs) w.r.t. positions
                    # d(lhs)/d(p_i) = 2*p_i - 2*p_j
                    # d(lhs)/d(p_j) = 2*p_j - 2*p_i
                    diff = 2.0 * (pos[i] - pos[j])
                    grad_pos[i] += diff
                    grad_pos[j] -= diff
        
        # Small step
        step = 0.01
        pos += step * grad_pos
        pos = np.clip(pos, 0.001, 0.999)
        
        # Re-solve radii
        pos, radii = _sdp_relaxation(pos, radii, max_iter=5, deadline=deadline)
    
    return pos, radii

def _simulated_annealing(pos, radii, deadline, n_iter=5000):
    """Simulated annealing with contact-aware moves."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _sum_radii(best_pos, best_radii)
    
    T = 0.05
    T_min = 0.001
    cooling = 0.995
    
    rng = np.random.RandomState(12345)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move a random circle
            i = rng.randint(N)
            dx = rng.normal(0, 0.03)
            dy = rng.normal(0, 0.03)
            new_pos = pos.copy()
            new_pos[i, 0] = np.clip(new_pos[i, 0] + dx, radii[i], 1.0 - radii[i])
            new_pos[i, 1] = np.clip(new_pos[i, 1] + dy, radii[i], 1.0 - radii[i])
            new_radii = radii.copy()
            
        elif move_type < 0.7:
            # Grow/shrink a random circle
            i = rng.randint(N)
            factor = 1.0 + rng.normal(0, 0.1)
            new_radii = radii.copy()
            new_radii[i] = np.clip(new_radii[i] * factor, 1e-4, 0.4)
            # Check boundary
            max_r = min(new_pos[i, 0], 1.0 - new_pos[i, 0], new_pos[i, 1], 1.0 - new_pos[i, 1])
            new_radii[i] = min(new_radii[i], max_r)
            new_pos = pos.copy()
            
        else:
            # Swap positions of two circles
            i, j = rng.choice(N, size=2, replace=False)
            new_pos = pos.copy()
            new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
            new_radii = radii.copy()
        
        # Validate
        if _is_valid(new_pos, new_radii, tol=1e-6):
            new_score = _sum_radii(new_pos, new_radii)
            delta = new_score - _sum_radii(pos, radii)
            
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                pos = new_pos
                radii = new_radii
                if new_score > best_score:
                    best_score = new_score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        
        T = max(T * cooling, T_min)
    
    return best_pos, best_radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.11
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
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_corner_cluster():
    """Corner-cluster initialization: large circles in corners, small in center."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 large corner circles
    corners = [(0.18, 0.18), (0.82, 0.18), (0.18, 0.82), (0.82, 0.82)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.18
    
    # 4 edge-mid circles
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        pos[i + 4] = [cx, cy]
        radii[i + 4] = 0.08
    
    # 18 remaining: random placement
    rng = np.random.RandomState(42)
    for i in range(8, N):
        for attempt in range(300):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r = min(x, 1 - x, y, 1 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > 0.02:
                pos[i] = [x, y]
                radii[i] = max_r * 0.9
                break
        if i >= 8:
            pos[i] = [0.5, 0.5]
            radii[i] = 0.01
    
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _slsqp_polish(pos, radii, deadline):
    """SLSQP with hard inequality constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.4))
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 150, 'ftol': 1e-12})
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

def _grow_radii(pos, deadline):
    """Fix positions, maximize radii via SLSQP."""
    x0 = np.concatenate([pos.flatten(), np.full(N, 0.01)])
    
    bounds = [(0.001, 0.999)] * (2 * N) + [(0.001, 0.4)] * N
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-12})
        if result.success or result.fun < objective(x0):
            x = result.x
            radii_out = np.maximum(x[2*N:], 1e-4)
            pos_out, radii_out = _post_process(pos, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, pos  # fallback

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate diverse initializations
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cor, rad_cor = _init_corner_cluster()
    inits.append((pos_cor, rad_cor))
    
    for s in [100, 200, 300, 400]:
        pos_r, rad_r = _init_random(s)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = _sum_radii(best_pos, best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: SDP relaxation (LP-based radius optimization + position adjustment)
        pos_sdp, rad_sdp = _sdp_relaxation(pos_init, rad_init, max_iter=20, deadline=deadline - 15)
        pos_sdp, rad_sdp = _post_process(pos_sdp, rad_sdp)
        score = _sum_radii(pos_sdp, rad_sdp)
        if _is_valid(pos_sdp, rad_sdp) and score > best_score:
            best_score = score
            best_pos = pos_sdp.copy()
            best_radii = rad_sdp.copy()
        
        # Phase 2: Gram matrix optimization
        if time.time() < deadline - 15:
            pos_gram, rad_gram = _gram_matrix_optimize(pos_sdp, rad_sdp, deadline=deadline - 10)
            pos_gram, rad_gram = _post_process(pos_gram, rad_gram)
            score2 = _sum_radii(pos_gram, rad_gram)
            if _is_valid(pos_gram, rad_gram) and score2 > best_score:
                best_score = score2
                best_pos = pos_gram.copy()
                best_radii = rad_gram.copy()
        
        # Phase 3: Simulated annealing
        if time.time() < deadline - 10:
            pos_sa, rad_sa = _simulated_annealing(pos_gram, rad_gram, deadline=deadline - 8, n_iter=3000)
            pos_sa, rad_sa = _post_process(pos_sa, rad_sa)
            score3 = _sum_radii(pos_sa, rad_sa)
            if _is_valid(pos_sa, rad_sa) and score3 > best_score:
                best_score = score3
                best_pos = pos_sa.copy()
                best_radii = rad_sa.copy()
        
        # Phase 4: SLSQP polish
        if time.time() < deadline - 8:
            pos_sl, rad_sl = _slsqp_polish(pos_sa, rad_sa, deadline=deadline - 5)
            pos_sl, rad_sl = _post_process(pos_sl, rad_sl)
            score4 = _sum_radii(pos_sl, rad_sl)
            if _is_valid(pos_sl, rad_sl) and score4 > best_score:
                best_score = score4
                best_pos = pos_sl.copy()
                best_radii = rad_sl.copy()
    
    # Final: grow radii with fixed positions
    if time.time() < deadline - 5:
        pos_grow, rad_grow = _grow_radii(best_pos, deadline=deadline - 3)
        pos_grow, rad_grow = _post_process(pos_grow, rad_grow)
        score_grow = _sum_radii(pos_grow, rad_grow)
        if _is_valid(pos_grow, rad_grow) and score_grow > best_score:
            best_score = score_grow
            best_pos = pos_grow.copy()
            best_radii = rad_grow.copy()
    
    # Final SLSQP
    if time.time() < deadline - 3:
        pos_final, rad_final = _slsqp_polish(best_pos, best_radii, deadline=deadline - 1)
        pos_final, rad_final = _post_process(pos_final, rad_final)
        score_final = _sum_radii(pos_final, rad_final)
        if _is_valid(pos_final, rad_final) and score_final > best_score:
            best_score = score_final
            best_pos = pos_final.copy()
            best_radii = rad_final.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
