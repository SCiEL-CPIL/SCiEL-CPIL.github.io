import numpy as np
import time
import random

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    def compute_autoconv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv
    
    # Parametric families for initialization
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    def make_bimodal(sigma, alpha):
        """Bimodal: two Gaussians at positions alpha*N and (1-alpha)*N"""
        pos1 = alpha * N
        pos2 = (1 - alpha) * N
        g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
        g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
        return g1 + g2
    
    def make_poly(c, d, e):
        """Polynomial: c + d*(x-center)^2 + e*(x-center)^4, clipped to nonneg"""
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        return np.maximum(val, 0.01)
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal with various parameters
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            inits.append(make_bimodal(sigma, alpha))
    
    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        inits.append(make_poly(c, d, e))
    
    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    # W-shape (two bumps)
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # Random smooth
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    best_h = None
    best_val = np.inf
    
    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 10:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Stage 2: Simulated Annealing refinement
    if time_left() > 20:
        current_h = best_h.copy()
        current_val = best_val
        best_h_sa = best_h.copy()
        best_val_sa = best_val
        
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        n_iter = int(min(5000, time_left() * 50))  # Cap iterations
        
        for k in range(n_iter):
            if time_left() < 5:
                break
            
            # Random perturbation: pick a few random indices and modify
            n_perturb = random.randint(1, 5)
            indices = random.sample(range(N), n_perturb)
            
            new_h = current_h.copy()
            for idx in indices:
                # Add random noise
                noise = random.gauss(0, T * 0.1)
                new_h[idx] = max(0, current_h[idx] + noise)
            
            # Normalize
            s = np.sum(new_h)
            if s < 1e-10:
                continue
            new_h /= s
            
            new_val = compute_autoconv_max(new_h)
            
            # Acceptance criterion
            if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / T):
                current_h = new_h
                current_val = new_val
                
                if current_val < best_val_sa:
                    best_val_sa = current_val
                    best_h_sa = current_h.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_val_sa < best_val:
            best_val = best_val_sa
            best_h = best_h_sa
    
    # Stage 3: Local coordinate descent refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(20, time_left() // 2))
        
        for round_num in range(max_rounds):
            if time_left() < 5:
                break
            
            improved = False
            # Try modifying each coordinate
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break
                
                # Try increasing and decreasing this coordinate
                for delta in [0.1, 0.05, 0.02, -0.1, -0.05, -0.02]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] + delta * current_h[idx])
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s
                    
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
