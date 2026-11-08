import numpy as np
import time

N = 400
DEADLINE = 140.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # Use FFT to compute the autoconvolution efficiently
    # Pad to at least 2*N to avoid circular convolution effects
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    # Build the signal on a fine grid using vectorized operations
    # Each interval has width h = 1/N
    # We represent the function on a grid of 'size' points
    # We can do this by repeating each value size//N times
    # To be safe and exact, we can use np.repeat if size is divisible by N, 
    # but size is a power of 2 and N=400, so it's not divisible.
    # The original code used a loop. Let's try to vectorize or use a better method.
    # Actually, for N=400, size is 1024. 1024/400 is not integer.
    # The original code:
    # start = i * (size // N)
    # end = (i + 1) * (size // N)
    # This effectively truncates the last bit of the interval for some i?
    # size // N = 1024 // 400 = 2.
    # So each step is 2 samples wide. 400 * 2 = 800 samples used out of 1024.
    # The rest are zeros? 
    # f = np.zeros(size)
    # for i in range(N):
    #    start = i * 2
    #    end = (i+1) * 2
    #    f[start:end] = signed_step[i]
    # This puts the function in the first 800 samples, leaving 224 zeros at the end.
    # This is a valid representation for linear convolution via FFT if we treat the zeros as padding.
    
    # Let's replicate the exact logic of the seed to ensure the metric is identical.
    f = np.zeros(size)
    # Vectorized assignment is tricky with non-divisible sizes if we want to match the loop exactly.
    # However, the loop does:
    # f[i*(size//N) : (i+1)*(size//N)] = signed_step[i]
    # Since size//N is constant (2), we can do:
    step_size = size // N
    # Create indices
    # idx = np.arange(N) * step_size
    # This would be 0, 2, 4, ...
    # We can just do a loop if it's fast enough, or use repeat and slice.
    # np.repeat(signed_step, step_size) gives length N*step_size = 800.
    f[:N*step_size] = np.repeat(signed_step, step_size)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale by the width of each grid cell
    dx = 0.5 / size
    autoconv *= dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with random signs or all ones? 
    # All ones gives C3 = max |int f^2| = 0.5 (since f=1 on [-0.25, 0.25], int is 0.5, conv at 0 is 0.5? No.
    # f(t)=1 on [-0.25, 0.25].
    # (f*f)(t) = int f(s)f(t-s) ds.
    # At t=0, int_{-0.25}^{0.25} 1*1 ds = 0.5.
    # So C3 = 0.5.
    # Random signs might start lower?
    # Let's start with all ones to be consistent with "seed" if possible, but random is often better for SA.
    # The prompt says "seed program" returns all ones? "Initialize with all ones (the seed)".
    # But we are allowed to change the strategy.
    
    np.random.seed(42)
    
    # Start with a random configuration to avoid the trivial all-ones local minimum immediately?
    # Or start with all ones and let SA explore.
    # Starting random is usually better for SA.
    current_step = np.random.choice([-1, 1], size=N).astype(np.int8)
    current_c3 = _compute_c3(current_step)
    
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Simulated Annealing parameters
    # We have ~140 seconds.
    # _compute_c3 is fast (FFT 1024).
    # Let's estimate iterations.
    # FFT 1024 is ~microseconds. Python overhead dominates.
    # Let's say we can do ~50,000 - 100,000 iterations.
    
    # Schedule: T(t) = T0 * alpha^t
    # We want T to drop to near 0 by the end.
    # Or use a time-based schedule.
    
    # Let's use a fixed number of iterations based on time, or just a large number and break on time.
    # A better schedule for SA is often T = T0 * (T_end/T0)^(t/T_max)
    
    T0 = 1.0  # Initial temperature (relative to typical energy diff)
    T_end = 0.001
    
    # Estimate max iterations possible.
    # Let's run a quick benchmark? No, just guess and break.
    # 140s / 0.001s/iter = 140k iters.
    # Let's aim for 100k iters.
    
    max_iter = 100000
    
    # We can also do multiple restarts if time permits.
    # But SA itself explores.
    
    for i in range(max_iter):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Cooling schedule
        # Geometric cooling: T = T0 * (T_end/T0)^(i/max_iter)
        # This is smooth.
        # Or simple: T = T0 * alpha^i
        # Let's use the geometric interpolation to ensure we reach T_end.
        alpha = (T_end / T0) ** (1.0 / max_iter)
        T = T0 * (alpha ** i)
        
        # Generate move: flip a random index
        idx = np.random.randint(N)
        
        # Calculate delta energy
        # We need C3 of new state.
        # Optimization: Can we update FFT incrementally?
        # f_new = f_old - 2*val_old*delta_ei
        # FFT(f_new) = FFT(f_old) - 2*val_old*FFT(delta_ei)
        # FFT(delta_ei) is easy to compute? It's a vector with 2 ones at positions corresponding to interval i.
        # This would save the full FFT of f, but we still need IFFT and max.
        # The bottleneck is likely Python overhead + IFFT + Max.
        # Full recompute is simpler and robust.
        
        old_val = current_step[idx]
        current_step[idx] = -old_val
        
        new_c3 = _compute_c3(current_step)
        
        delta = new_c3 - current_c3
        
        if delta < 0:
            # Improvement
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Accept with probability exp(-delta / T)
            # If T is very small, exp(-large) is 0.
            # If T is large, exp(-small) is ~1.
            # Optimization: if delta > 10*T, accept prob is negligible.
            if T > 1e-9 and delta < 10 * T:
                if np.random.random() < np.exp(-delta / T):
                    current_c3 = new_c3
                else:
                    # Reject
                    current_step[idx] = old_val
            else:
                # Reject
                current_step[idx] = old_val
        
    return best_step
