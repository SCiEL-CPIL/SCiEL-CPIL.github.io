import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Autoconvolution: (h * h)[k] = sum_i h[i] * h[i+k] for k >= 0
    # Sup norm is max over all k of the autoconvolution
    # We want to maximize ||h||_2^2 / (||h||_1 * max_k (h*h)[k])
    
    # Epigraph formulation:
    # Maximize t subject to:
    #   ||h||_2^2 >= t * ||h||_1 * M  (where M = max autoconvolution)
    #   (h*h)[k] <= M for all k
    #   h[i] >= 0
    #   ||h||_1 = 1 (normalization, since ratio is scale-invariant)
    
    # Actually, let's use a different approach:
    # Fix ||h||_1 = 1, then maximize ||h||_2^2 / M where M = max autoconvolution
    # Equivalently: minimize M / ||h||_2^2 subject to ||h||_1 = 1, h >= 0
    # Or: minimize log(M) - 2*log(||h||_2) subject to constraints
    
    # Let's use the epigraph with SLSQP:
    # Variables: h[0..N-1], M
    # Minimize: M / sum(h^2)  -- but this is nonsmooth at h=0
    # Better: minimize M * sum(1/h_i^2) ... no that's bad
    # 
    # Let's just directly minimize M subject to:
    #   sum(h^2) >= 1  (normalize L2 to 1)
    #   (h*h)[k] <= M for all k
    #   h[i] >= 0
    #   sum(h) = 1  (normalize L1 to 1)
    # 
    # Wait, if we normalize L1=1 and L2>=1, that's inconsistent unless h is binary-like.
    # 
    # Let me reconsider. The ratio is R = ||h||_2^2 / (||h||_1 * M)
    # This is scale invariant. Let's fix ||h||_1 = 1.
    # Then R = ||h||_2^2 / M.
    # Maximize ||h||_2^2 / M subject to sum(h)=1, h>=0, M >= (h*h)[k] for all k.
    # Equivalently: minimize M / ||h||_2^2 subject to sum(h)=1, h>=0, (h*h)[k]<=M.
    #
    # Let's use log barrier or just direct SLSQP on:
    # Minimize: M / sum(h^2)
    # Subject to: sum(h) - 1 = 0
    #             (h*h)[k] - M <= 0 for k=0..N-1
    #             h[i] >= 0
    
    def autoconv_peak(h):
        """Compute max of autoconvolution of h."""
        # h*h at lag k: sum_{i=0}^{N-1-k} h[i]*h[i+k]
        # We can compute all lags
        ac = np.correlate(h, h, mode='full')
        # 'full' gives lags from -(N-1) to N-1, but since h is real, ac is symmetric
        # The peak is the max of the full autoconvolution
        return np.max(ac)
    
    def objective_and_grad(x):
        """x = [h_0, ..., h_{N-1}, M]"""
        h = x[:N]
        M = x[N]
        
        # Objective: M / sum(h^2)
        s2 = np.sum(h**2)
        if s2 < 1e-12:
            s2 = 1e-12
        obj = M / s2
        
        # Gradient
        # d(obj)/dM = 1/s2
        # d(obj)/dh_i = -M * 2*h_i / s2^2
        
        grad = np.zeros(N + 1)
        grad[:N] = -2.0 * M * h / (s2 * s2)
        grad[N] = 1.0 / s2
        
        return obj, grad
    
    def constraint_autoconv(x, k):
        """(h*h)[k] - M <= 0"""
        h = x[:N]
        M = x[N]
        # Autoconv at lag k (for k >= 0)
        if k < N:
            val = np.sum(h[:N-k] * h[k:])
        else:
            val = 0.0
        return val - M
    
    def constraint_autoconv_grad(x, k):
        h = x[:N]
        grad = np.zeros(N + 1)
        if k < N:
            # d/dh_i of sum_{j=0}^{N-1-k} h[j]*h[j+k]
            # For i in [0, N-1-k]: contributes h[i+k]
            # For i in [k, N-1]: contributes h[i-k]
            if k == 0:
                grad[:N] = 2.0 * h
            else:
                # h[i] appears in term i (as first factor) if i < N-k
                # and in term i-k (as second factor) if i >= k
                grad[:N-k] += h[k:]
                grad[k:] += h[:N-k]
        grad[N] = -1.0
        return grad
    
    # Build constraints
    constraints = []
    # Equality: sum(h) = 1
    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.sum(x[:N]) - 1.0,
        'jac': lambda x: np.concatenate([np.ones(N), [0.0]])
    })
    
    # Inequality: (h*h)[k] <= M for k = 0, ..., N-1
    for k in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, kk=k: constraint_autoconv(x, kk),
            'jac': lambda x, kk=k: constraint_autoconv_grad(x, kk)
        })
    
    # Bounds: h[i] >= 0, M >= 0
    bounds = [(0.0, None)] * N + [(1e-6, None)]
    
    # Initial guesses - try several different starting points
    best_score = 0.0
    best_h = np.ones(N) / N
    
    # Strategy: start with various structured initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Sparse: pick indices from a Sidon-like set
    # Use a simple approach: pick indices that are spread out
    for density in [0.3, 0.5, 0.7, 1.0]:
        n_active = max(1, int(N * density))
        # Spread out indices
        indices = np.linspace(0, N-1, n_active, dtype=int)
        h_init = np.zeros(N)
        h_init[indices] = 1.0 / n_active
        initializations.append(h_init)
    
    # 3. Random sparse
    rng = np.random.RandomState(42)
    for trial in range(5):
        n_active = rng.randint(5, N)
        indices = rng.choice(N, n_active, replace=False)
        h_init = np.zeros(N)
        h_init[indices] = 1.0 / n_active
        initializations.append(h_init)
    
    # 4. Gaussian-like bump
    x_grid = np.linspace(0, N-1, N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [3, 5, 10]:
            h_init = np.exp(-((x_grid - center)**2) / (2 * width**2))
            h_init /= np.sum(h_init)
            initializations.append(h_init)
    
    # 5. Alternating pattern
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0 / (N//2 + 1)
    initializations.append(h_alt)
    
    # 6. Power-law decay
    for alpha in [0.5, 1.0, 2.0]:
        h_pw = np.array([(i+1)**(-alpha) for i in range(N)])
        h_pw /= np.sum(h_pw)
        initializations.append(h_pw)
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        x0 = np.concatenate([h0, [autoconv_peak(h0)]])
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < 1e6:
                h_opt = result.x[:N]
                M_opt = result.x[N]
                s2 = np.sum(h_opt**2)
                if s2 > 1e-12 and M_opt > 1e-12:
                    score = s2 / M_opt  # This is ||h||_2^2 / M with ||h||_1=1
                    # But wait, we need to verify the actual ratio
                    ac_peak = autoconv_peak(h_opt)
                    l1 = np.sum(h_opt)
                    l2_sq = np.sum(h_opt**2)
                    actual_ratio = l2_sq / (l1 * ac_peak)
                    
                    if actual_ratio > best_score:
                        best_score = actual_ratio
                        best_h = h_opt.copy()
                        
        except Exception:
            continue
    
    # Final polish: run SLSQP from the best solution with tighter tolerance
    if best_score > 0 and time.time() < deadline - 5.0:
        x0 = np.concatenate([best_h, [autoconv_peak(best_h)]])
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='SLSQP',
                jac=True,
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            if result.fun < 1e6:
                h_opt = result.x[:N]
                ac_peak = autoconv_peak(h_opt)
                l1 = np.sum(h_opt)
                l2_sq = np.sum(h_opt**2)
                if l1 > 1e-12 and ac_peak > 1e-12:
                    actual_ratio = l2_sq / (l1 * ac_peak)
                    if actual_ratio > best_score:
                        best_score = actual_ratio
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Return the best heights found
    return list(best_h)
