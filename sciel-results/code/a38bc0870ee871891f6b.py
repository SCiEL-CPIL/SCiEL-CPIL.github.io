import time
import numpy as np

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    return conv

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks with various configurations
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse with various densities
    for _ in range(10):
        hr = np.zeros(N)
        density = rng.uniform(0.2, 0.8)
        mask = rng.random(N) < density
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)
        
    # 6. Plateau with tapered edges
    for plateau_frac in [0.3, 0.5, 0.7]:
        hp = np.zeros(N)
        p_start = int((1 - plateau_frac) / 2 * N)
        p_end = int((1 + plateau_frac) / 2 * N)
        hp[p_start:p_end] = 1.0
        # Taper
        taper_len = max(1, int(0.1 * N))
        if p_start > taper_len:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            hp[p_start - taper_len:p_start] = ramp
        if p_end < N - taper_len:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            hp[p_end:p_end + taper_len] = ramp[::-1]
        candidates.append(hp)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Multi-Start Simulated Annealing with improved parameters
    num_chains = 8
    steps_per_chain = 1500
    
    for chain_idx in range(num_chains):
        if deadline_reached():
            break
        
        # Initialize this chain
        if chain_idx == 0:
            current_h = best_h.copy()
        elif chain_idx == 1:
            # Best from initial candidates
            current_h = candidates[rng.integers(0, min(10, len(candidates)))].copy()
        else:
            # Mix: 50% from candidates, 50% random perturbation of best
            if rng.random() < 0.5:
                current_h = candidates[rng.integers(0, len(candidates))].copy()
            else:
                current_h = np.maximum(best_h + rng.normal(0, 0.3, N), 0.0)
        
        current_score = compute_score(current_h)
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        # SA Parameters - more aggressive cooling
        T_start = 0.2
        T_end = 0.0005
        step_size = 0.15
        min_step = 0.005
        
        for step in range(steps_per_chain):
            if deadline_reached():
                break
                
            # Annealing schedule
            progress = step / steps_per_chain
            T = T_start * (T_end / T_start) ** progress
            
            # Generate perturbation
            noise = rng.normal(0, step_size, N)
            new_h = current_h + noise
            new_h = np.maximum(new_h, 0.0)
            
            if np.sum(new_h) <= 0:
                continue
                
            new_score = compute_score(new_h)
            
            # Acceptance
            if new_score < current_score:
                accept = True
            else:
                delta = new_score - current_score
                # Accept small deteriorations with probability exp(-delta/T)
                if delta < 0.1: 
                    accept = rng.random() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                # Increase step size slightly on acceptance
                step_size = min(0.3, step_size * 1.02)
            else:
                # Decrease step size on rejection
                step_size = max(min_step, step_size * 0.92)
                
    # Final polish: Coordinate descent + local refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_score = best_score
        
        # Coordinate descent: optimize one coordinate at a time
        coord_passes = 2
        for _ in range(coord_passes):
            if deadline_reached():
                break
            for i in range(N):
                if deadline_reached():
                    break
                # Try small changes to coordinate i
                old_val = current_h[i]
                for delta in [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]:
                    new_h = current_h.copy()
                    new_h[i] = max(0.0, old_val + delta)
                    new_score = compute_score(new_h)
                    if new_score < current_score:
                        current_h = new_h
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_h = current_h.copy()
                        break
        
        # Local random search around best
        if time_left() > 10:
            polish_steps = 800
            current_h = best_h.copy()
            current_score = best_score
            step_size = 0.03
            
            for _ in range(polish_steps):
                if deadline_reached():
                    break
                # Perturb a random subset of coordinates
                num_perturb = rng.integers(5, 20)
                indices = rng.choice(N, num_perturb, replace=False)
                noise = rng.normal(0, step_size, num_perturb)
                new_h = current_h.copy()
                new_h[indices] = np.maximum(current_h[indices] + noise, 0.0)
                new_score = compute_score(new_h)
                
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    step_size = min(0.08, step_size * 1.03)
                else:
                    step_size = max(0.003, step_size * 0.93)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
