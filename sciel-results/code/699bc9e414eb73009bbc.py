import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def _correlation_peaks(h):
    """Return the autoconvolution values at all lags (length 2N-1)."""
    return np.correlate(h, h, mode='full')

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Baseline: 2-element
    h2 = np.zeros(N)
    h2[0] = 2.0/3.0
    h2[1] = 1.0/3.0
    try_candidate(h2)
    
    # Baseline: uniform
    h_uni = np.ones(N) / N
    try_candidate(h_uni)

    # --- Multiplicative update (Sinkhorn-Knopp style) for the min-max game ---
    # The objective is R(h) = ||h||_2^2 / (||h||_1 * max_k c_k) where c_k = (h*h)[k].
    # We formulate a zero-sum game: defender chooses h >= 0, attacker chooses lag k.
    # Multiplicative updates naturally enforce non-negativity.
    
    def multiplicative_update(h_init, n_iter=2000, temperature=1.0):
        """
        Iterative multiplicative updates approximating the saddle point of the min-max game.
        
        Idea: The gradient of log(R) w.r.t. h_i involves:
          - 2*h_i / ||h||_2^2  (from the numerator)
          - 1 / ||h||_1        (from the L1 norm)
          - sum over lags k where c_k is near max: (2*h_i contribution to c_k) / max_k c_k
        
        We use a smoothed version with temperature to handle the max.
        """
        h = np.asarray(h_init, dtype=float).copy()
        h = np.maximum(h, 1e-10)  # ensure positive for multiplicative updates
        
        best_local = 0.0
        best_local_h = h.copy()
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute autoconvolution
            c = _correlation_peaks(h)  # length 2N-1
            c_max = np.max(c)
            if c_max < 1e-15:
                break
            
            # Smoothed "attacker" weights: softmax over lags with temperature
            # This gives a smooth surrogate for the max operation
            c_shifted = c - c_max
            log_weights = c_shifted / (temperature * c_max + 1e-15)
            # Clip to avoid overflow
            log_weights = np.clip(log_weights, -50, 0)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            
            # For each height h_i, compute its contribution to each lag c_k
            # c_k = sum_j h_j * h_{j+k} (with appropriate indexing)
            # Gradient of c_k w.r.t. h_i: 
            #   if k >= 0: c_k = sum_j h_j * h_{j+k}, so dc_k/dh_i = h_{i-k} (if i-k in range) + h_{i+k} (if i+k in range) for k>0, and 2*h_i for k=0
            #   Actually for full correlation: c[k] = sum_j h[j] * h[j - k + N-1] ... let me think more carefully.
            # 
            # np.correlate(h, h, 'full') gives c[m] = sum_j h[j] * h[j - m + (N-1)]
            # So c[m] = sum_j h[j] * h[j - m + N - 1]
            # dc[m]/dh[i] = h[i - m + N - 1] (if 0 <= i-m+N-1 < N) + h[i + m - (N-1)] (if 0 <= i+m-(N-1) < N)
            # For m=0: dc[0]/dh[i] = 2*h[i]
            # For m > 0: dc[m]/dh[i] = h[i - m + N - 1] if valid + h[i + m - (N-1)] if valid
            # For m < 0: symmetric
            
            # Compute the "pressure" on each h_i from the attacker
            # pressure_i = sum_k weights[k] * dc_k/dh_i / c_max
            pressure = np.zeros(N)
            
            # For lag m (index in c array, 0 to 2N-2, corresponding to lag m-(N-1)):
            # dc[m]/dh[i] = h[i - (m-(N-1)) + N - 1] + h[i + (m-(N-1)) - (N-1)]
            #              = h[i - m + 2*(N-1)] + h[i + m - 2*(N-1)]
            # Let lag = m - (N-1), so lag ranges from -(N-1) to (N-1)
            # dc/dh[i] for lag L: h[i-L] (if 0<=i-L<N) + h[i+L] (if 0<=i+L<N), with factor 2 if L=0
            
            for m in range(2*N - 1):
                w = weights[m]
                if w < 1e-10:
                    continue
                L = m - (N - 1)  # lag
                if L == 0:
                    # dc/dh[i] = 2*h[i]
                    pressure += (2.0 * w / c_max) * h
                elif L > 0:
                    # dc/dh[i] = h[i-L] (if i>=L) + h[i+L] (if i<N-L)
                    for i in range(N):
                        contrib = 0.0
                        if i - L >= 0:
                            contrib += h[i - L]
                        if i + L < N:
                            contrib += h[i + L]
                        pressure[i] += w * contrib / c_max
                else:
                    # L < 0, let L' = -L > 0
                    Lp = -L
                    for i in range(N):
                        contrib = 0.0
                        if i + Lp < N:
                            contrib += h[i + Lp]
                        if i - Lp >= 0:
                            contrib += h[i - Lp]
                        pressure[i] += w * contrib / c_max
            
            # Update rule: multiplicative update
            # h_i <- h_i * (gradient_factor)
            # The log-derivative of R w.r.t. log(h_i) is:
            #   2*h_i^2/||h||_2^2 - h_i/||h||_1 - (h_i/c_max) * sum_k weights[k] * (dc_k/dh_i)/h_i
            # Simplified: we want to increase h_i if it helps the ratio.
            # 
            # Use a simpler heuristic: 
            #   h_i <- h_i * (1 + eta * (2*h_i/||h||_2^2 - 1/||h||_1 - pressure_i/h_i))
            # But this can go negative. Use multiplicative form:
            #   h_i <- h_i * exp(eta * (2*h_i/||h||_2^2 - 1/||h||_1 - pressure_i / max(h_i, 1e-10)))
            
            l2_sq = np.sum(h**2)
            l1 = np.sum(h)
            
            # Log-derivative of R w.r.t. log(h_i):
            # d log(R) / d log(h_i) = 2*h_i^2/l2_sq - h_i/l1 - (h_i/c_max) * sum_k weights[k] * (dc_k/dh_i)/h_i
            #                        = 2*h_i^2/l2_sq - h_i/l1 - sum_k weights[k] * (dc_k/dh_i)/c_max
            # The last term is exactly pressure_i.
            
            log_deriv = 2.0 * h**2 / l2_sq - h / l1 - pressure
            
            # Multiplicative update with learning rate
            eta = 0.5
            h = h * np.exp(eta * log_deriv)
            h = np.maximum(h, 1e-10)
            
            # Check score
            s = compute_score(h)
            if s > best_local:
                best_local = s
                best_local_h = h.copy()
        
        return best_local_h, best_local
    
    # Try multiplicative updates from various initializations
    structures = []
    
    # 1. Ramp functions
    for slope in [0.5, 1.0, 2.0]:
        h = np.linspace(0, slope, N)
        structures.append(h)
    
    # 2. Symmetric bump
    center = N // 2
    for width in [5, 10, 15, 20, 25]:
        h = np.zeros(N)
        for i in range(max(0, center-width), min(N, center+width+1)):
            h[i] = np.exp(-((i-center)**2) / (2*width**2))
        structures.append(h)
    
    # 3. Periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    h[i] = 1.0
            structures.append(h)
    
    # 4. Random sparse patterns
    np.random.seed(42)
    for sparsity in [0.1, 0.2, 0.3, 0.5]:
        for _ in range(5):
            h = np.zeros(N)
            n_nonzero = max(1, int(N * sparsity))
            idx = np.random.choice(N, n_nonzero, replace=False)
            h[idx] = np.random.uniform(0.1, 1.0, n_nonzero)
            structures.append(h)
    
    # 5. Power-law decay
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i+1)**(-alpha)
        structures.append(h)
    
    # 6. Two-bump structures
    for pos1 in [5, 10, 15, 20, 25]:
        for pos2 in [30, 35, 40, 45]:
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            structures.append(h)
    
    # 7. Cosine windows
    for freq in [1, 2, 3, 4]:
        h = np.cos(np.pi * np.arange(N) * freq / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            structures.append(h)
    
    # 8. Uniform with small noise (to break symmetry)
    np.random.seed(123)
    for _ in range(5):
        h = np.ones(N) * 0.5 + np.random.uniform(-0.2, 0.2, N)
        h = np.maximum(h, 0)
        structures.append(h)
    
    # Run multiplicative updates on each structure
    for h_init in structures:
        if time.time() > deadline - 15:
            break
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-15:
            continue
        h_refined, s_refined = multiplicative_update(h_init, n_iter=500, temperature=1.0)
        try_candidate(h_refined)
    
    # Coordinate descent polish on the best found
    def coordinate_descent(h, max_iter=200):
        """Cyclic coordinate descent for maximizing the ratio."""
        h = h.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a few perturbations
                for delta in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
                    h[i] = max(0.0, old + delta)
                    s = compute_score(h)
                    if s > best_score:
                        improved = True
                # Also try scaling: h[i] = old * factor
                for factor in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
                    h[i] = max(0.0, old * factor)
                    s = compute_score(h)
                    if s > best_score:
                        improved = True
                if not improved:
                    h[i] = old
            if not improved:
                break
        return h
    
    # Final: try a more intensive coordinate descent on the best found
    if time.time() < deadline - 5:
        h_final = coordinate_descent(best_h, max_iter=500)
        try_candidate(h_final)
    
    return list(best_h)
