import numpy as np
import time

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
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
        return np.fft.ifft(H * H).real
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal with various parameters
    for sigma in [N/12, N/10, N/8, N/7, N/6, N/5]:
        for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            pos1 = alpha * N
            pos2 = (1 - alpha) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2)
    
    # Trimodal
    for sigma in [N/10, N/8]:
        for a, b in [(0.1, 0.5), (0.15, 0.5), (0.2, 0.5), (0.25, 0.5)]:
            pos1 = a * N
            pos2 = b * N
            pos3 = (1 - a) * N
            g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
            g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
            g3 = np.exp(-((i - pos3) ** 2) / (2 * sigma ** 2))
            inits.append(g1 + g2 + g3)
    
    # U-shaped (mass at edges)
    u_shape = 1.0 + 3.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    # Strong U
    u_shape2 = 0.5 + 5.0 * ((i - center) / (N/2))**2
    inits.append(u_shape2)
    
    # W-shape
    w_shape = 1.0 + 2.0 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # M-shape (inverse W)
    m_shape = 1.0 - 2.0 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(m_shape, 0.01))
    
    # Random smooth
    np.random.seed(42)
    for _ in range(10):
        freqs = np.random.choice([1, 2, 3, 4, 5, 6, 7], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.3, 2.0, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    # Block structures
    for nblocks in [2, 3, 4]:
        block_size = N // nblocks
        for offset in range(block_size):
            block = np.zeros(N)
            for b in range(nblocks):
                start = b * block_size + offset
                end = start + block_size
                if end <= N:
                    block[start:end] = 1.0
                elif start < N:
                    block[start:N] = 1.0
            inits.append(block)
    
    best_h = None
    best_val = np.inf
    
    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 20:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Stage 2: Simulated annealing with multiple move types
    if time_left() > 30:
        current_h = best_h.copy()
        current_val = best_val
        
        T0 = 0.01
        T_end = 0.0001
        cooling_rate = 0.9995
        
        T = T0
        no_improve = 0
        max_no_improve = 200
        
        while time_left() > 25 and T > T_end:
            if time_left() < 10:
                break
            
            # Choose move type
            move_type = np.random.choice(['swap', 'perturb', 'block'], p=[0.4, 0.4, 0.2])
            
            if move_type == 'swap':
                # Swap mass between two random positions
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 == idx2:
                    continue
                
                # Transfer some mass from idx1 to idx2
                amount = np.random.uniform(0.01, 0.1) * current_h[idx1]
                if amount < 1e-10:
                    continue
                
                new_h = current_h.copy()
                new_h[idx1] -= amount
                new_h[idx2] += amount
                
                if new_h[idx1] < 0:
                    continue
                
                s = np.sum(new_h)
                new_h /= s
                
            elif move_type == 'perturb':
                # Perturb a random subset of bins
                n_perturb = np.random.randint(5, 20)
                idxs = np.random.choice(N, n_perturb, replace=False)
                
                new_h = current_h.copy()
                for idx in idxs:
                    factor = np.random.uniform(0.5, 1.5)
                    new_h[idx] *= factor
                
                new_h = np.maximum(new_h, 0)
                s = np.sum(new_h)
                if s < 1e-10:
                    continue
                new_h /= s
                
            else:  # block
                # Move a block of consecutive bins
                block_len = np.random.randint(5, 30)
                start = np.random.randint(0, N - block_len)
                
                new_h = current_h.copy()
                # Zero out the block and redistribute
                block_mass = np.sum(new_h[start:start+block_len])
                new_h[start:start+block_len] = 0
                
                # Add to a random location
                target_start = np.random.randint(0, N - block_len)
                new_h[target_start:target_start+block_len] += block_mass / block_len
                
                s = np.sum(new_h)
                new_h /= s
            
            new_val = compute_autoconv_max(new_h)
            
            # Acceptance criterion
            delta = new_val - current_val
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_h = new_h
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cooling_rate
            
            # Restart if stuck
            if no_improve > max_no_improve:
                T = max(T * 2, 0.005)
                no_improve = 0
    
    # Stage 3: Local refinement with coordinate descent
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        
        # Multiple passes
        for pass_num in range(5):
            if time_left() < 5:
                break
            
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break
                
                # Try increasing and decreasing this bin
                for delta_factor in [1.1, 1.05, 0.95, 0.9]:
                    new_h = current_h.copy()
                    new_h[idx] *= delta_factor
                    new_h = np.maximum(new_h, 0)
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
