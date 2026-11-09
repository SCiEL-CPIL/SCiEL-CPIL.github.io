"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    N = 24

    # Pre-compute evaluation grid for the FAST objective (used during search)
    n_fast = 2048
    theta_fast = np.linspace(0, 2 * np.pi, n_fast, endpoint=False)
    z_fast = np.exp(1j * theta_fast)
    Z_fast = np.zeros((N, n_fast), dtype=complex)
    Z_fast[0] = 1.0
    for k in range(1, N):
        Z_fast[k] = Z_fast[k - 1] * z_fast

    # Pre-compute evaluation grid for the PRECISE objective (used for final candidates)
    n_precise = 16384
    theta_precise = np.linspace(0, 2 * np.pi, n_precise, endpoint=False)
    z_precise = np.exp(1j * theta_precise)
    Z_precise = np.zeros((N, n_precise), dtype=complex)
    Z_precise[0] = 1.0
    for k in range(1, N):
        Z_precise[k] = Z_precise[k - 1] * z_precise

    def fast_eval(c):
        """Fast sup-norm estimate using coarse grid (vectorized)."""
        vals = np.dot(c, Z_fast)
        return np.max(np.abs(vals))

    def precise_eval(c):
        """Precise sup-norm with golden-section refinement of top peaks."""
        vals = np.dot(c, Z_precise)
        abs_vals = np.abs(vals)
        
        # Find top 8 peaks for refinement
        peak_indices = np.argsort(abs_vals)[-8:]
        
        best_val = abs_vals[peak_indices[0]]
        
        step = 2 * np.pi / n_precise
        
        for pi in peak_indices:
            a = theta_precise[pi - 1]
            b = theta_precise[(pi + 1) % n_precise]
            if a > b:
                b += 2 * np.pi
            
            def P_abs(tv):
                zv = np.exp(1j * tv)
                val = 0.0 + 0.0j
                zp = 1.0 + 0.0j
                for k in range(N):
                    val += c[k] * zp
                    zp *= zv
                return abs(val)
            
            gr = (np.sqrt(5) - 1) / 2
            inv_gr = 1.0 / gr
            x1 = a + (b - a) * inv_gr
            x2 = a + (b - a) * gr
            f1 = P_abs(x1)
            f2 = P_abs(x2)
            
            for _ in range(50):
                if f1 < f2:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + (b - a) * gr
                    f2 = P_abs(x2)
                else:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = a + (b - a) * inv_gr
                    f1 = P_abs(x1)
                if b - a < 1e-13:
                    break
            
            tv = (x1 + x2) / 2
            fv = P_abs(tv)
            if fv > best_val:
                best_val = fv
        
        return best_val

    # --- Simulated Annealing with fast objective ---
    # Start from a random initialization
    rng = np.random.default_rng(42)
    current = rng.choice([-1, 1], size=N).astype(np.float64)
    current_fast = fast_eval(current)
    
    best = current.copy()
    best_fast = current_fast
    best_precise = None
    
    # Track top candidates by fast score for precise evaluation
    candidate_pool = []  # list of (fast_score, coeffs_copy)
    
    # Annealing schedule
    T_start = 0.5
    T_end = 0.005
    n_steps = 50000  # Will be limited by time
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    # Periodic precise evaluation interval
    precise_interval = 5000
    last_precise_step = 0
    
    for step in range(n_steps):
        if time.time() > deadline - 10.0:
            break
        
        # Pick a random position to flip
        idx = rng.integers(0, N)
        new = current.copy()
        new[idx] *= -1
        new_fast = fast_eval(new)
        
        delta = new_fast - current_fast
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = new
            current_fast = new_fast
            if current_fast < best_fast:
                best_fast = current_fast
                best = current.copy()
        
        # Cool down
        T *= cooling
        
        # Periodically evaluate best candidate precisely
        if step - last_precise_step >= precise_interval:
            last_precise_step = step
            if best_precise is None or best_fast < best_precise[0] - 0.01:
                bp = precise_eval(best)
                if best_precise is None or bp < best_precise[0]:
                    best_precise = (bp, best.copy())
        
        # Also periodically evaluate current if it's promising
        if step % 2000 == 0 and current_fast < best_precise[0] * np.sqrt(N) * 1.01 if best_precise else True:
            cp = precise_eval(current)
            if best_precise is None or cp < best_precise[0]:
                best_precise = (cp, current.copy())

    # --- Final precise evaluation of best candidates ---
    # Evaluate the best found by fast metric
    final_precise = precise_eval(best)
    final_coeffs = best.copy()
    
    if best_precise is not None:
        if best_precise[0] < final_precise:
            final_precise = best_precise[0]
            final_coeffs = best_precise[1].copy()

    # --- Local polish: try single and double flips on the best precise candidate ---
    current_best_precise = final_precise
    current_best_coeffs = final_coeffs
    
    improved = True
    polish_iters = 0
    max_polish = 5
    
    while improved and polish_iters < max_polish and time.time() < deadline - 5.0:
        improved = False
        polish_iters += 1
        
        # Single flips
        best_single_score = current_best_precise
        best_single_idx = -1
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            tmp = current_best_coeffs.copy()
            tmp[i] *= -1
            s = precise_eval(tmp)
            if s < best_single_score - 1e-12:
                best_single_score = s
                best_single_idx = i
        
        # Double flips
        best_double_score = current_best_precise
        best_double_idx = (-1, -1)
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5.0:
                    break
                tmp = current_best_coeffs.copy()
                tmp[i] *= -1
                tmp[j] *= -1
                s = precise_eval(tmp)
                if s < best_double_score - 1e-12:
                    best_double_score = s
                    best_double_idx = (i, j)
        
        # Apply best improvement
        if best_single_score < best_double_score:
            if best_single_score < current_best_precise - 1e-12:
                current_best_precise = best_single_score
                current_best_coeffs[best_single_idx] *= -1
                improved = True
        else:
            if best_double_score < current_best_precise - 1e-12:
                current_best_precise = best_double_score
                i, j = best_double_idx
                current_best_coeffs[i] *= -1
                current_best_coeffs[j] *= -1
                improved = True
    
    # --- Multi-start: try a few different random starts if time permits ---
    if time.time() < deadline - 30.0:
        for seed_try in range(5):
            if time.time() > deadline - 10.0:
                break
            c_try = rng.choice([-1, 1], size=N).astype(np.float64)
            # Quick anneal from this start
            T_local = 0.3
            cool_local = 0.9995
            c_best_local = c_try.copy()
            f_best_local = fast_eval(c_best_local)
            c_cur_local = c_try.copy()
            f_cur_local = f_best_local
            
            for _ in range(3000):
                idx_l = rng.integers(0, N)
                c_new_local = c_cur_local.copy()
                c_new_local[idx_l] *= -1
                f_new_local = fast_eval(c_new_local)
                d = f_new_local - f_cur_local
                if d < 0 or rng.random() < np.exp(-d / T_local):
                    c_cur_local = c_new_local
                    f_cur_local = f_new_local
                    if f_cur_local < f_best_local:
                        f_best_local = f_cur_local
                        c_best_local = c_new_local.copy()
                T_local *= cool_local
            
            # Precise eval of the best from this start
            p = precise_eval(c_best_local)
            if p < current_best_precise:
                current_best_precise = p
                current_best_coeffs = c_best_local.copy()

    return current_best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
