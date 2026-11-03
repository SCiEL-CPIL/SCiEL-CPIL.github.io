import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1

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
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # --- New approach: Direct optimization of the ratio ---
    # We want to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Subject to h >= 0, sum(h) = 1 (wlog, since the ratio is scale-invariant)
    # 
    # Key insight: For nonnegative h, (h*h)[0] = ||h||_2^2, so ||h*h||_inf >= ||h||_2^2
    # Thus R <= 1/||h||_1 = 1 (when sum(h)=1). But we can do better if the peak
    # of the convolution is NOT at lag 0, or if we structure h so that the
    # convolution peak is small relative to ||h||_2^2.
    #
    # Actually, for nonnegative h, (h*h)[0] = sum(h_i^2) = ||h||_2^2, so the
    # peak is always at least ||h||_2^2. The ratio is exactly 1 when the peak
    # IS at lag 0. To get ratio > 1, we'd need... wait, that's impossible for
    # nonnegative h since (h*h)[0] = ||h||_2^2 and ||h*h||_inf >= (h*h)[0].
    #
    # Hmm, let me re-read. The ratio is ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # For nonnegative h, ||h*h||_inf >= ||h||_2^2, so ratio <= 1/||h||_1.
    # If we normalize sum(h) = 1, ratio <= 1.
    #
    # But the seed scores 1.32! So either my analysis is wrong or the
    # normalization is different. Let me check: the score is l2_sq / (l1 * DELTA * max_conv).
    # Note the DELTA factor! So it's not just ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # but has a 1/DELTA factor. DELTA = 0.5/50 = 0.01, so 1/DELTA = 100.
    #
    # So the actual quantity is ||h||_2^2 / (||h||_1 * ||h*h||_inf) * (1/DELTA)
    # = ||h||_2^2 / (||h||_1 * ||h*h||_inf) * 100
    #
    # For a uniform vector h = ones(N)/N:
    # ||h||_2^2 = N * (1/N)^2 = 1/N = 0.02
    # ||h||_1 = 1
    # (h*h)[0] = ||h||_2^2 = 0.02, but the peak of convolution is at the center
    # For uniform, (h*h)[k] for k in [0, N-1] is (N-k)/N^2, peak at k=0: 1/N = 0.02
    # Wait no, (h*h)[k] = sum_{j} h_j h_{k-j}. For h_j = 1/N,
    # (h*h)[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} (1/N)(1/N) = (min(N-1,k) - max(0,k-N+1) + 1)/N^2
    # At k=0: 1/N^2 = 1/2500 = 0.0004
    # At k=N-1=49: 50/2500 = 0.02
    # So peak is at k=N-1 (or symmetrically), value = N/N^2 = 1/N = 0.02
    #
    # Ratio = 0.02 / (1 * 0.01 * 0.02) = 0.02 / 0.0002 = 100
    # But the seed scores 1.32, not 100. So I'm misunderstanding something.
    #
    # Let me recompute more carefully.
    # h = ones(50)/50, so h_i = 0.02
    # l1 = sum(h) = 50 * 0.02 = 1
    # l2_sq = sum(h^2) = 50 * 0.0004 = 0.02
    # conv = np.convolve(h, h, 'full') has length 99
    # conv[k] = sum_{j=0}^{49} h[j] * h[k-j] where h is 0 outside [0,49]
    # conv[0] = h[0]*h[0] = 0.0004
    # conv[1] = h[0]*h[1] + h[1]*h[0] = 2*0.0004 = 0.0008
    # ...
    # conv[49] = sum_{j=0}^{49} h[j]*h[49-j] = 50 * 0.0004 = 0.02
    # conv[50] = sum_{j=1}^{49} h[j]*h[50-j] = 49 * 0.0004 = 0.0196
    # So max_conv = 0.02
    #
    # score = l2_sq / (l1 * DELTA * max_conv) = 0.02 / (1 * 0.01 * 0.02) = 0.02/0.0002 = 100
    #
    # That can't be right if the seed scores 1.32. Let me check the actual seed.
    # The seed is best_h = np.ones(N)/N initially, but then it gets optimized.
    # The final score of 1.32 is after optimization. So the uniform gives 100?
    # That seems way too high. Let me re-examine.
    
    # Actually wait, I think I need to just run the code and see. Let me just
    # implement a solid optimization approach.
    
    # Strategy: Use a combination of:
    # 1. SLSQP on the epigraph form (minimize peak conv subject to sum(h)=1, h>=0)
    #    Then maximize l2_sq / (DELTA * peak) since l1=1
    # 2. Simulated annealing / random restarts
    # 3. Coordinate descent refinement
    
    x_idx = np.arange(N)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse patterns
    np.random.seed(123)
    for trial in range(50):
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Gaussians
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(2, N-2, 12):
        for w in np.linspace(1, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(np.random.randint(3, 8))/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Alternating
    for start in range(N):
        for gap in [2, 3, 4, 5, 6]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)

    # Epigraph optimization: minimize M subject to conv_k <= M, sum(h)=1, h>=0
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
        jac = np.zeros((NUM_CONV, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            k_start = j
            k_end = min(j + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] -= h[indices_h]
            
            k_start2 = max(0, j - N + 1)
            k_end2 = min(j, NUM_CONV)
            if k_start2 < NUM_CONV and k_end2 >= 0:
                indices_k = np.arange(k_start2, k_end2 + 1)
                indices_h = j - indices_k
                jac[indices_k, j] -= h[indices_h]
                
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
    
    def run_epigraph(h_init, maxiter=200, ftol=1e-12):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
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
    
    # Sort inits by initial score
    scored_inits = []
    for h_init in inits:
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Run epigraph from top candidates
    max_epigraph_runs = min(len(scored_inits), 40)
    for i in range(max_epigraph_runs):
        if time.time() > deadline - 30:
            break
        s_init, h_init = scored_inits[i]
        h_candidate = run_epigraph(h_init, maxiter=150, ftol=1e-12)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # Local refinement
    def refine_local(h_start, max_iter=30):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            M_val = np.max(conv)
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            current_score = l2_sq / (l1 * DELTA * M_val)
            
            epsilon = 1e-4
            best_move = None
            best_move_score = current_score
            
            for i in range(N):
                if h[i] < epsilon:
                    continue
                for j in range(N):
                    if i == j:
                        continue
                    h_new = h.copy()
                    h_new[i] -= epsilon
                    h_new[j] += epsilon
                    if h_new[i] < 0:
                        continue
                    
                    l2_sq_new = l2_sq + 2*epsilon*(h[j]-h[i]) + 2*epsilon**2
                    conv_new = np.convolve(h_new, h_new, mode='full')
                    M_new = np.max(conv_new)
                    
                    if M_new <= 0:
                        continue
                        
                    score_new = l2_sq_new / (l1 * DELTA * M_new)
                    
                    if score_new > best_move_score + 1e-8:
                        best_move_score = score_new
                        best_move = (i, j, epsilon)
            
            if best_move:
                i, j, eps = best_move
                h[i] -= eps
                h[j] += eps
                improved = True
        
        return h

    if time.time() < deadline - 15:
        h_refined = refine_local(best_h, max_iter=25)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined

    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) > 0:
        best_h /= np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    return list(best_h)
