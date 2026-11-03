import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

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

    # 9. Sidon-set-like: use a known maximal Sidon set on Z_50 or random search for low-collision sets
    # A Sidon set (B_2 set) has all pairwise sums distinct (mod N), minimizing convolution collisions.
    # We'll try to construct such sets by greedy or random search.
    
    def find_sidon_like(n_points, n_trials=500):
        """Find a set of indices with minimal collision count in pairwise sums mod N."""
        best_set = None
        best_collisions = float('inf')
        
        for _ in range(n_trials):
            idx = np.random.choice(N, n_points, replace=False)
            # Count collisions in pairwise sums
            sums = np.array([(i + j) % N for i in idx for j in idx if i <= j])
            unique_sums = np.unique(sums)
            collisions = len(sums) - len(unique_sums)
            if collisions < best_collisions:
                best_collisions = collisions
                best_set = idx.copy()
        
        return best_set
    
    # Try various sizes of Sidon-like sets
    for k in range(5, 12):
        sidon_set = find_sidon_like(k, n_trials=200)
        if sidon_set is not None:
            h_sidon = np.zeros(N)
            h_sidon[sidon_set] = 1.0
            h_sidon /= np.sum(h_sidon)
            inits.append(h_sidon)

    # 10. Weighted Sidon-like: vary weights on Sidon sets
    for k in range(5, 10):
        sidon_set = find_sidon_like(k, n_trials=100)
        if sidon_set is not None:
            for _ in range(10):
                weights = np.random.rand(k) + 0.1
                weights /= np.sum(weights)
                h_w = np.zeros(N)
                h_w[sidon_set] = weights
                inits.append(h_w)

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
    
    def run_epigraph(h_init, maxiter=300, ftol=1e-12):
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
    
    # Run epigraph from multiple inits
    for h_init in inits:
        if time.time() > deadline - 30:
            break
        h_candidate = run_epigraph(h_init, maxiter=300, ftol=1e-12)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Block-Coordinate Descent Refinement ---
    # Identify dominant peak and optimize symmetric pairs
    def block_coordinate_refine(h_start, max_iter=50):
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
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            # This exploits the structure that conv[k_star] = sum_j h[j]*h[k_star-j]
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                # Current contribution to peak: h[i]*h[j]
                # If we increase h[i] by delta and decrease h[j] by delta (keeping sum constant)
                # New peak contribution changes
                # For simplicity, do a 1D search on t where h[i] -> h[i]+t, h[j] -> h[j]-t
                # constrained to keep both non-negative
                
                hi = h[i]
                hj = h[j]
                
                # t range: [-hi, hj] to keep both non-negative
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                # Evaluate score at several t values
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 10):
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
    
    # Refine best solution with block coordinate descent
    if time.time() < deadline - 15:
        h_refined = block_coordinate_refine(best_h, max_iter=30)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # --- Phase 4: Local search with random perturbations ---
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            h_pert = best_h.copy()
            # Random perturbation
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
                
                # Refine the new best
                if time.time() < deadline - 10:
                    h_refined = block_coordinate_refine(best_h, max_iter=20)
                    s_refined = _score_raw(h_refined)
                    if s_refined > best_score:
                        best_score = s_refined
                        best_h = h_refined.copy()
    
    # --- Phase 5: Spectral-based refinement using FFT ---
    # The autoconvolution can be computed via FFT: h*h = IFFT(FFT(h)^2)
    # We can use this for faster evaluation and potentially a different optimization landscape
    
    def _score_raw_fft(h):
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        # Use FFT for convolution
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
        max_conv = np.max(conv)
        if max_conv <= 1e-15:
            return 0.0
        return l2_sq / (l1 * DELTA * max_conv)
    
    # Try spectral-based local search: optimize in Fourier domain
    if time.time() < deadline - 10:
        h_curr = best_h.copy()
        h_curr = np.clip(h_curr, 0, None)
        if np.sum(h_curr) < 1e-15:
            h_curr = np.ones(N) / N
        h_curr /= np.sum(h_curr)
        
        n_fft = 2 * N
        H = np.fft.rfft(h_curr, n=n_fft)
        
        # Perturb in Fourier domain (preserves real-valuedness via conjugate symmetry)
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            
            H_pert = H.copy()
            # Small random perturbation in frequency domain
            noise = (np.random.randn(len(H_pert)) + 1j * np.random.randn(len(H_pert))) * 0.05
            H_pert += noise
            # Enforce conjugate symmetry for real signal
            H_pert[1:] = H_pert[1:]  # rfft already handles this
            h_pert = np.fft.irfft(H_pert, n=n_fft)[:N]
            h_pert = np.clip(h_pert, 0, None)
            if np.sum(h_pert) < 1e-15:
                continue
            h_pert /= np.sum(h_pert)
            
            s_pert = _score_raw_fft(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                H = np.fft.rfft(best_h, n=n_fft)
                
                # Refine
                if time.time() < deadline - 5:
                    h_refined = block_coordinate_refine(best_h, max_iter=15)
                    s_refined = _score_raw(h_refined)
                    if s_refined > best_score:
                        best_score = s_refined
                        best_h = h_refined.copy()
                        H = np.fft.rfft(best_h, n=n_fft)
    
    return list(best_h)
