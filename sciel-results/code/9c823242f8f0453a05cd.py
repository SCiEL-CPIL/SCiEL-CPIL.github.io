import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _sidon_sequences(max_len=50, max_val=50):
    """Generate sequences with minimal additive collisions (Sidon-like)."""
    sequences = []
    # Try various starting points and step patterns
    for start in range(0, 10):
        for step1 in range(1, 8):
            for step2 in range(1, 8):
                if step1 == step2:
                    continue
                seq = []
                val = start
                for i in range(max_len):
                    if val < max_val:
                        seq.append(val)
                    else:
                        break
                    val += step1 if i % 2 == 0 else step2
                if len(seq) >= 3:
                    sequences.append(seq)
    # Also try quadratic-like sequences
    for a in range(1, 6):
        for b in range(0, 6):
            seq = []
            for i in range(max_len):
                val = a * i * i + b * i
                if val < max_val:
                    seq.append(val)
                else:
                    break
            if len(seq) >= 3:
                sequences.append(seq)
    return sequences

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured
    np.random.seed(42)
    for trial in range(30):
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: pairs symmetric around center
    for k in range(3, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
                
    # 7. Spike-like (single peak)
    for pos in range(N):
        h_spike = np.zeros(N)
        h_spike[pos] = 1.0
        inits.append(h_spike)
        
    # 8. Two-spike
    for pos1 in range(0, N, 3):
        for pos2 in range(pos1+2, N, 3):
            h_two = np.zeros(N)
            h_two[pos1] = 1.0
            h_two[pos2] = 1.0
            h_two /= np.sum(h_two)
            inits.append(h_two)
    
    # 9. Sidon-based initializations (minimize convolution collisions)
    sidon_seqs = _sidon_sequences(max_len=N, max_val=N)
    for seq in sidon_seqs:
        if len(seq) >= 3 and len(seq) <= N:
            h_sidon = np.zeros(N)
            for pos in seq:
                if 0 <= pos < N:
                    h_sidon[pos] = 1.0
            if np.sum(h_sidon) > 1e-10:
                h_sidon /= np.sum(h_sidon)
                inits.append(h_sidon)
    
    # 10. Random Sidon-like: pick positions with large minimum pairwise difference
    np.random.seed(123)
    for trial in range(20):
        k = np.random.randint(5, 12)
        # Greedy selection with minimum spacing
        positions = []
        for _ in range(k):
            candidates = np.arange(N)
            # Penalize positions close to already selected
            scores = np.ones(N)
            for p in positions:
                dist = np.abs(candidates - p)
                scores *= (1 + dist / N)
            # Also penalize based on current convolution contribution
            if positions:
                h_temp = np.zeros(N)
                for p in positions:
                    h_temp[p] = 1.0
                conv_temp = np.convolve(h_temp, h_temp, mode='full')
                # Prefer positions that add less to max conv
                for c in range(N):
                    add_conv = h_temp[c] * 1.0  # approximate
                scores *= (1 + conv_temp[:N] / (np.max(conv_temp) + 1e-10))
            # Pick position with highest score
            best_pos = np.argmax(scores)
            positions.append(best_pos)
        h_sidon2 = np.zeros(N)
        for pos in positions:
            h_sidon2[pos] = 1.0
        if np.sum(h_sidon2) > 1e-10:
            h_sidon2 /= np.sum(h_sidon2)
            inits.append(h_sidon2)

    # --- Phase 2: Epigraph Formulation Optimization ---
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M (x[N])
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0 (via bounds)
    # 3. M - conv_k >= 0 for all k
    
    num_conv = 2 * N - 1
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        # d(M - conv_k)/dh_j = -2 * h_{k-j} where valid
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
        
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter=500, ftol=1e-14):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-8
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None
    
    # Sort inits by initial score (best first) to prioritize good starting points
    inits_with_scores = [(h, _score_raw(h)) for h in inits]
    inits_with_scores.sort(key=lambda x: -x[1])
    
    # Run epigraph from multiple inits, prioritizing better initializations
    for h_init, init_score in inits_with_scores:
        if time.time() > deadline - 40:
            break
        # Skip if initial score is much worse than current best
        if init_score < best_score * 0.5:
            continue
        h_candidate = run_epigraph(h_init, maxiter=500, ftol=1e-14)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Block-Coordinate Descent Refinement ---
    # Identify dominant peak and optimize symmetric pairs
    def block_coordinate_refine(h_start, max_iter=80):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 15:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                hi = h[i]
                hj = h[j]
                
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                # Finer search with more points
                for t in np.linspace(t_min * 0.9, t_max * 0.9, 20):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_t = t
                
                if abs(best_t) > 1e-10:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    if np.sum(h) < 1e-15:
                        break
                    h /= np.sum(h)
                    improved = True
            
            # Also try single-variable moves: increase h[i], decrease others proportionally
            for i in range(N):
                if h[i] < 1e-15:
                    continue
                # Try reducing h[i] slightly and distributing to other positions
                for delta_frac in [0.05, 0.1, 0.2]:
                    delta = h[i] * delta_frac
                    if delta < 1e-15:
                        continue
                    h_test = h.copy()
                    h_test[i] -= delta
                    # Distribute to other positions proportional to their current values
                    others = np.ones(N)
                    others[i] = 0
                    others *= h
                    total_others = np.sum(others)
                    if total_others < 1e-15:
                        continue
                    others /= total_others
                    h_test += delta * others
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > _score_raw(h):
                        h = h_test.copy()
                        improved = True
                        break
        
        return h
    
    # Refine best solution with block coordinate descent
    if time.time() < deadline - 20:
        h_refined = block_coordinate_refine(best_h, max_iter=80)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
        
        # Second round of refinement
        if time.time() < deadline - 10:
            h_refined2 = block_coordinate_refine(best_h, max_iter=50)
            s_refined2 = _score_raw(h_refined2)
            if s_refined2 > best_score:
                best_score = s_refined2
                best_h = h_refined2.copy()
    
    # --- Phase 4: Local search with random perturbations ---
    if time.time() < deadline - 8:
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            h_pert = best_h.copy()
            # Random perturbation with varying strength
            strength = 0.1 + 0.2 * np.random.rand()
            mask = np.random.rand(N) < 0.3
            h_pert[mask] *= (1 + strength * np.random.randn(np.sum(mask)))
            h_pert = np.clip(h_pert, 0, None)
            if np.sum(h_pert) < 1e-15:
                continue
            h_pert /= np.sum(h_pert)
            
            s_pert = _score_raw(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                
                # Refine the new best
                if time.time() < deadline - 5:
                    h_refined = block_coordinate_refine(best_h, max_iter=30)
                    s_refined = _score_raw(h_refined)
                    if s_refined > best_score:
                        best_score = s_refined
                        best_h = h_refined.copy()
    
    # Final polish: run epigraph one more time from the best solution
    if time.time() < deadline - 5:
        h_candidate = run_epigraph(best_h, maxiter=300, ftol=1e-14)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    return list(best_h)
