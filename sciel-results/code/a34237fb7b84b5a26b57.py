import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes D/M where D = H(x-y), M = max(H(x), H(x+y), H(x+2y)).
    Uses a combination of random search, simulated annealing, and gradient-based polishing.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]

    # Flatten for binning
    d_val = (xs - ys).flatten()       # x - y
    m0_val = xs.flatten()              # x
    m1_val = (xs + ys).flatten()       # x + y
    m2_val = (xs + 2 * ys).flatten()   # x + 2y

    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)

    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def shannon(p_dist):
        p_nonzero = p_dist[p_dist > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    def get_score_smooth(p_flat, beta=100.0):
        """Smooth version using soft-max for gradient-friendly optimization."""
        H_d, M_components = compute_entropies_smooth(p_flat, beta)
        if M_components < 1e-9:
            return -1e9
        return H_d / M_components

    def compute_entropies_smooth(p_flat, beta=100.0):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)

        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)

        # Soft-max approximation: M ≈ (1/beta) * log(sum(exp(beta * H_i)))
        # As beta -> inf, this approaches max(H_m0, H_m1, H_m2)
        H_vals = np.array([H_m0, H_m1, H_m2])
        H_max_val = np.max(H_vals)
        if H_max_val < 1e-15:
            M_smooth = 0.0
        else:
            exp_vals = np.exp(beta * (H_vals - H_max_val))
            M_smooth = H_max_val + np.log(np.sum(exp_vals)) / beta

        return H_d, M_smooth

    def log_softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    # --- Generate Initializers ---
    rng = np.random.default_rng(42)
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(100) / 100.0)
    
    # 2. Structured starts
    # Diagonals
    for offset in range(-9, 10):
        p = np.zeros(100)
        count = 0
        for y in range(10):
            for x in range(10):
                if x - y == offset:
                    p[y*10+x] = 1.0
                    count += 1
        if count > 0:
            p /= count
            starts.append(p)
            
    # Anti-diagonals
    for offset in range(0, 20):
        p = np.zeros(100)
        count = 0
        for y in range(10):
            for x in range(10):
                if x + y == offset:
                    p[y*10+x] = 1.0
                    count += 1
        if count > 0:
            p /= count
            starts.append(p)
            
    # Lines x = const
    for x0 in range(10):
        p = np.zeros(100)
        p[x0*10:(x0+1)*10] = 1.0/10.0
        starts.append(p)
        
    # Lines y = const
    for y0 in range(10):
        p = np.zeros(100)
        for x in range(10):
            p[y0*10+x] = 1.0/10.0
        starts.append(p)
        
    # 3. Random Dirichlet starts
    for _ in range(50):
        alpha = rng.uniform(0.1, 2.0, 100)
        p = rng.dirichlet(alpha)
        starts.append(p)
        
    # 4. Gaussian blobs
    for _ in range(20):
        cy, cx = rng.uniform(1, 8, 2)
        sigma = rng.uniform(0.5, 2.0)
        yy, xx = np.mgrid[0:10, 0:10]
        z = -((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2)
        p = np.exp(z).flatten()
        p /= np.sum(p)
        starts.append(p)

    # 5. Multi-anti-diagonal patterns (target high H_{-1} by spreading across x-y diagonals)
    for num_diags in [3, 5, 7]:
        for offset_start in range(0, 19 - num_diags + 1):
            p = np.zeros(100)
            count = 0
            for y in range(10):
                for x in range(10):
                    d = x - y
                    if offset_start <= d < offset_start + num_diags:
                        p[y*10+x] = 1.0
                        count += 1
            if count > 0:
                p /= count
                starts.append(p)

    # 6. Concentrated on few x-y diagonals (high D/M potential)
    for d_target in range(-9, 10):
        for spread in [1, 2, 3]:
            p = np.zeros(100)
            count = 0
            for y in range(10):
                for x in range(10):
                    d = x - y
                    if abs(d - d_target) < spread:
                        p[y*10+x] = 1.0
                        count += 1
            if count > 0:
                p /= count
                starts.append(p)

    # --- Phase 1: Quick Evaluation and Selection ---
    best_score = -1e9
    p_best = starts[0].copy()
    
    # Evaluate all starts
    for p in starts:
        s = get_score(p)
        if s > best_score:
            best_score = s
            p_best = p.copy()
            
    # Select top 5 for further optimization
    scores = [get_score(p) for p in starts]
    top_indices = np.argsort(scores)[-5:]
    top_starts = [starts[i].copy() for i in top_indices]
    top_starts.append(p_best.copy())

    # --- Phase 2: Simulated Annealing on top starts ---
    # SA helps escape local minima of the ratio function
    
    def simulated_annealing(p_init, max_iters=2000, T_start=1.0, T_end=0.01):
        if time.time() > deadline - 30.0:
            return p_init, get_score(p_init)
            
        p = p_init.copy()
        current_score = get_score(p)
        best_p_local = p.copy()
        best_score_local = current_score
        
        for i in range(max_iters):
            if time.time() > deadline - 30.0:
                break
                
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Propose move: move mass from one cell to another
            i_from = rng.integers(0, 100)
            i_to = rng.integers(0, 100)
            
            if i_from == i_to:
                continue
                
            # Amount to move: small fraction
            delta = 0.01 * rng.uniform(0.1, 1.0)
            
            if p[i_from] < delta:
                continue
                
            p_new = p.copy()
            p_new[i_from] -= delta
            p_new[i_to] += delta
            
            score_new = get_score(p_new)
            
            # Acceptance criterion
            delta_score = score_new - current_score
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
                p = p_new
                current_score = score_new
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_p_local = p.copy()
                    
        return best_p_local, best_score_local

    for p_init in top_starts:
        if time.time() > deadline - 60.0:
            break
        p_sa, score_sa = simulated_annealing(p_init)
        if score_sa > best_score:
            best_score = score_sa
            p_best = p_sa

    # --- Phase 3: Coordinate Descent Refinement ---
    # Directly optimize individual cell weights for better local convergence
    
    def coordinate_descent(p_init, max_iters=5000):
        if time.time() > deadline - 50.0:
            return p_init, get_score(p_init)
            
        p = p_init.copy()
        current_score = get_score(p)
        best_p_local = p.copy()
        best_score_local = current_score
        
        step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
        
        for it in range(max_iters):
            if time.time() > deadline - 50.0:
                break
                
            improved = False
            # Try moving mass from each cell to a few random targets
            for i in range(100):
                if time.time() > deadline - 50.0:
                    break
                if p[i] < 1e-10:
                    continue
                    
                # Try a few random target cells
                for _ in range(5):
                    j = rng.integers(0, 100)
                    if i == j:
                        continue
                        
                    for step in step_sizes:
                        if p[i] < step:
                            continue
                        p_new = p.copy()
                        p_new[i] -= step
                        p_new[j] += step
                        s_new = get_score(p_new)
                        if s_new > current_score + 1e-12:
                            p = p_new
                            current_score = s_new
                            improved = True
                            if current_score > best_score_local:
                                best_score_local = current_score
                                best_p_local = p.copy()
                            break
                    if improved:
                        break
                if not improved and it > 100:
                    # Reduce step sizes if stuck
                    step_sizes = [s * 0.5 for s in step_sizes]
                    
        return best_p_local, best_score_local

    if time.time() < deadline - 60.0:
        p_cd, score_cd = coordinate_descent(p_best)
        if score_cd > best_score:
            best_score = score_cd
            p_best = p_cd

    # --- Phase 4: Nelder-Mead Polishing with smooth objective ---
    def objective_z_smooth(z, beta=50.0):
        p = log_softmax(z)
        s = get_score_smooth(p, beta)
        return -s

    z_best = np.log(p_best + 1e-15)
    
    if time.time() < deadline - 40.0:
        try:
            res_nm = minimize(
                objective_z_smooth,
                z_best,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'fatol': 1e-10, 'xatol': 1e-12}
            )
            p_nm = log_softmax(res_nm.x)
            score_nm = get_score(p_nm)
            if score_nm > best_score:
                best_score = score_nm
                p_best = p_nm
        except Exception:
            pass

    # --- Phase 5: SLSQP Final Polish with smooth objective ---
    bounds_p = [(0.0, 1.0) for _ in range(100)]
    constraints_p = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    
    def objective_p_smooth(p, beta=50.0):
        return -get_score_smooth(p, beta)
        
    if time.time() < deadline - 20.0:
        try:
            res_slsqp = minimize(
                lambda p: objective_p_smooth(p, beta=50.0),
                p_best,
                method='SLSQP',
                bounds=bounds_p,
                constraints=constraints_p,
                options={'maxiter': 800, 'ftol': 1e-14}
            )
            p_final = np.clip(res_slsqp.x, 0, None)
            s = np.sum(p_final)
            if s > 1e-9:
                p_final /= s
            else:
                p_final = p_best
            score_final = get_score(p_final)
            if score_final > best_score:
                best_score = score_final
                p_best = p_final
        except Exception:
            pass

    # --- Phase 6: Final coordinate descent polish ---
    if time.time() < deadline - 10.0:
        p_cd2, score_cd2 = coordinate_descent(p_best, max_iters=2000)
        if score_cd2 > best_score:
            best_score = score_cd2
            p_best = p_cd2

    # --- Final Validation ---
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)

    return P
