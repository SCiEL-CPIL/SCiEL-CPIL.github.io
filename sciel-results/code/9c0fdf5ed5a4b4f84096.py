import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def shape_score(h):
    """Scale-invariant score: minimize ||h*h||_inf * ||h||_1 / ||h||_2^2"""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l1 * conv_peak / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Initial best: uniform distribution
    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)

    # Simulated Annealing Parameters
    # We want to MINIMIZE shape_score.
    # Objective: f(h) = L1 * ConvPeak / L2_sq
    # Since we fix L1=1, f(h) = ConvPeak / L2_sq.
    
    # Start with a random sparse or structured initialization to avoid local minima
    # The uniform distribution is a strong baseline, but let's try a few diverse starts.
    
    starts = []
    starts.append(np.ones(N) / N)
    
    # Sparse starts
    for k in [2, 3, 5, 10]:
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / k
        starts.append(h)
        
    # Symmetric Gaussian-like start
    x = np.arange(N)
    h = np.exp(-0.5 * ((x - N//2) / (N/4))**2)
    h /= np.sum(h)
    starts.append(h)

    best_h = min(starts, key=shape_score)
    best_shape = shape_score(best_h)
    
    current_h = best_h.copy()
    current_score = best_shape
    
    # Annealing schedule
    T_start = 1e-3
    T_end = 1e-12
    alpha = (T_end / T_start) ** (1.0 / 5000) # 5000 steps
    
    # We will perform as many iterations as possible before the deadline.
    # Each iteration involves:
    # 1. Generating a perturbation (moving mass from i to j)
    # 2. Computing the new shape score (FFT based, ~O(N log N) but with overhead)
    
    # To speed up, we can batch evaluate or use a smarter move.
    # Move: Pick i, j randomly. Move small amount delta from i to j.
    # Maintain L1=1.
    
    T = T_start
    step = 0
    
    # Precompute nothing, just loop.
    # We need to be careful with time.
    # Let's estimate time per iteration.
    # FFT of size 100 is fast. Python overhead is the bottleneck.
    # Let's try to run for ~100,000 iterations if possible.
    
    while time.time() < deadline:
        # Generate move
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        # Amount to move
        # We want to explore the simplex.
        # Let's move a fraction of h[i] to h[j].
        # Or add/subtract a small noise.
        # Moving mass is better for maintaining non-negativity and L1=1.
        
        # Pick a fraction
        frac = np.random.uniform(0.01, 0.5)
        delta = current_h[i] * frac
        
        if delta < 1e-15:
            continue
            
        new_h = current_h.copy()
        new_h[i] -= delta
        new_h[j] += delta
        
        # Check non-negativity
        if new_h[i] < -1e-12:
            continue
            
        new_score = shape_score(new_h)
        
        # Metropolis criterion
        # We are MINIMIZING.
        # Accept if new_score < current_score
        # Or accept with probability exp(-(new_score - current_score) / T)
        
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if current_score < best_shape:
                best_shape = current_score
                best_h = current_h.copy()
        else:
            # Compute delta_E = new_score - current_score (positive)
            delta_E = new_score - current_score
            if delta_E < 1e-12:
                prob = 1.0
            else:
                prob = np.exp(-delta_E / T)
                
            if np.random.random() < prob:
                current_h = new_h
                current_score = new_score
        
        # Cool down
        T *= alpha
        step += 1
        
        # Occasionally, if stuck, try a larger jump or reset?
        # Simulated annealing usually handles this with T.
        # But if T gets too low, we might get stuck.
        # Let's add a restart mechanism if we haven't improved in a long time?
        # For now, standard SA.
        
        # Time check inside loop is expensive? time.time() is fast enough.
        # But let's check every 100 steps to reduce overhead?
        # No, time.time() is ~100ns. FFT is ~10us. Overhead is small.

    # Final polish: Local coordinate descent on the best found
    if time.time() < deadline - 5:
        h = best_h.copy()
        current_score = best_shape
        
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                if h[i] < 1e-12:
                    continue
                for j in range(N):
                    if i == j:
                        continue
                    # Try moving small amounts
                    for frac in [0.01, 0.05, 0.1, 0.2]:
                        delta = h[i] * frac
                        h_new = h.copy()
                        h_new[i] -= delta
                        h_new[j] += delta
                        if h_new[i] < -1e-12:
                            continue
                        s_new = shape_score(h_new)
                        if s_new < current_score - 1e-12:
                            h = h_new
                            current_score = s_new
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
                
        if current_score < best_shape:
            best_shape = current_score
            best_h = h

    # Normalize to L1=1
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 1e-15:
        best_h /= np.sum(best_h)
        
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print(f"Ratio: {compute_ratio(h)}")
    print(f"Shape score: {shape_score(h)}")
