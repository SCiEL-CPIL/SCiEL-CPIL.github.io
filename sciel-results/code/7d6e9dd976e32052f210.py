import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

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

def _find_sidon_support(target_size, n):
    """Greedy construction of a Sidon set (distinct pairwise sums)."""
    if target_size <= 0:
        return []
    
    support = [0]
    used_sums = {0}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_collisions = n + 1
        
        for candidate in range(n):
            if candidate in support:
                continue
            
            collisions = 0
            for s in support:
                if candidate + s in used_sums:
                    collisions += 1
            
            if collisions < best_collisions:
                best_collisions = collisions
                best_candidate = candidate
            
            if collisions == 0:
                break
        
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                used_sums.add(best_candidate + s)
        else:
            break
    
    return support

def _epigraph_optimize(h_init, maxiter=500, ftol=1e-10):
    """
    Optimize h to minimize peak autoconvolution, subject to sum(h)=1, h>=0.
    Variables: x = [h_0, ..., h_{N-1}, M]
    Objective: minimize M
    Constraints: sum(h)=1, M >= (h*h)_k for all k
    """
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
        h = np.clip(x[:N], 0, None)
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
    
    def constraint_ineq_jac(x):
        h = np.clip(x[:N], 0, None)
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        for k in range(num_conv):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    jac[k, i] = -2.0 * h[j]
        return jac
    
    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 1e-15:
        h_init = np.ones(N) / N
        s = 1.0
    h_init = h_init / s
    
    conv_init = np.convolve(h_init, h_init, mode='full')
    M_init = np.max(conv_init) + 1e-8
    x0 = np.concatenate([h_init, [M_init]])
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    constraints = [
        {'type': 'eq', 'fun': constraint_eq_fun, 'jac': constraint_eq_jac},
        {'type': 'ineq', 'fun': constraint_ineq_fun, 'jac': constraint_ineq_jac}
    ]
    
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            jac=objective_grad, bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        if res.success or res.fun < M_init:
            h_res = np.clip(res.x[:N], 0, None)
            s_res = np.sum(h_res)
            if s_res > 1e-15:
                h_res /= s_res
                return h_res
    except Exception:
        pass
    return None

def _block_coordinate_refine(h_start, deadline, max_iter=40):
    """Refine by pairwise adjustments exploiting autoconvolution structure."""
    h = h_start.copy()
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-15:
        return h
    h /= np.sum(h)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        if time.time() > deadline - 8:
            break
        improved = False
        iter_count += 1
        
        conv = np.convolve(h, h, mode='full')
        k_star = np.argmax(conv)
        
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
            
            for t in np.linspace(t_min * 0.8, t_max * 0.8, 12):
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
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sidon sets (structured sparse with distinct pairwise sums)
    np.random.seed(42)
    for target_k in [4, 5, 6, 7, 8, 10, 12]:
        support = _find_sidon_support(target_k, N)
        if len(support) >= 3:
            h_s = np.zeros(N)
            h_s[support] = 1.0 / len(support)
            inits.append(h_s)
    
    # 3. Random sparse
    for trial in range(30):
        k = np.random.randint(3, 14)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
    
    # 4. Concentrated Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.8, N/3, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
    
    # 5. Triangular
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(2, N, 4):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
    
    # 6. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
    
    # 7. Alternating patterns
    for start in range(0, N, 2):
        for gap in [3, 4, 5, 7]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)
    
    # 8. Two-spike
    for pos1 in range(0, N, 4):
        for pos2 in range(pos1+3, N, 4):
            h_two = np.zeros(N)
            h_two[pos1] = 1.0
            h_two[pos2] = 1.0
            h_two /= 2.0
            inits.append(h_two)
    
    # Sort inits by initial score
    scored_inits = []
    for h_init in inits:
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Phase 2: Epigraph optimization from diverse starts
    max_runs = min(len(scored_inits), 35)
    for i, (s_init, h_init) in enumerate(scored_inits[:max_runs]):
        if time.time() > deadline - 40:
            break
        h_candidate = _epigraph_optimize(h_init, maxiter=400, ftol=1e-10)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # Phase 3: Block coordinate refinement
    if time.time() < deadline - 25:
        h_refined = _block_coordinate_refine(best_h, deadline, max_iter=35)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Phase 4: Local search with perturbations
    if time.time() < deadline - 15:
        for _ in range(25):
            if time.time() > deadline - 8:
                break
            h_pert = best_h.copy()
            mask = np.random.rand(N) < 0.3
            h_pert[mask] *= (1 + 0.2 * np.random.randn(np.sum(mask)))
            h_pert = np.clip(h_pert, 0, None)
            if np.sum(h_pert) < 1e-15:
                continue
            h_pert /= np.sum(h_pert)
            
            s_pert = _score_raw(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                if time.time() < deadline - 12:
                    h_refined = _block_coordinate_refine(best_h, deadline, max_iter=20)
                    s_refined = _score_raw(h_refined)
                    if s_refined > best_score:
                        best_score = s_refined
                        best_h = h_refined.copy()
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 10:
        h_polish_init = best_h.copy()
        h_polish_init[h_polish_init < 1e-8] = 0.0
        s_p = np.sum(h_polish_init)
        if s_p > 1e-15:
            h_polish_init /= s_p
        h_polish = _epigraph_optimize(h_polish_init, maxiter=800, ftol=1e-12)
        if h_polish is not None:
            s_polish = _score_raw(h_polish)
            if s_polish > best_score:
                best_score = s_polish
                best_h = h_polish.copy()
    
    if time.time() < deadline - 5:
        h_final = _block_coordinate_refine(best_h, deadline, max_iter=25)
        s_final = _score_raw(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    return list(best_h)
