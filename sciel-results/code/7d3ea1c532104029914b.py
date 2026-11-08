import numpy as np
import time

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 is defined as the maximum absolute value of the autoconvolution.
    """
    # signed_step is an array of N values in {-1, 1}
    # We need to pad to at least 2*N to avoid circular convolution effects
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    # Build the signal on a fine grid
    # Each interval has width h = 1/N
    # We'll represent the function on a grid of 'size' points
    # Vectorized construction for speed
    f = np.zeros(size)
    # Create indices for the fine grid
    # Each bit i corresponds to size//N fine-grid points
    points_per_bit = size // N
    # Repeat each bit value points_per_bit times
    f = np.repeat(signed_step, points_per_bit)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale by the width of each grid cell
    # The integral is approximated by sum * dx, where dx = 0.5 / size
    dx = 0.5 / size
    autoconv *= dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with all ones
    current_step = np.ones(N, dtype=np.int8)
    
    # Pre-compute FFT setup
    size = 1 << (2 * N - 1).bit_length()
    points_per_bit = size // N
    dx = 0.5 / size
    
    # Initial FFT and autoconvolution
    f = np.repeat(current_step, points_per_bit)
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    current_c3 = np.max(np.abs(autoconv))
    
    best_step = current_step.copy()
    best_c3 = current_c3
    
    # Pre-compute the delta effect of flipping each bit on the FFT
    # Flipping bit i changes f by 2 * s_i on the segment i
    # The change in FFT is 2 * s_i * FFT(segment_i)
    # We can pre-compute the FFT of the "segment" basis vectors? 
    # Actually, it's faster to just compute the delta autoconvolution incrementally.
    
    # To make it fast, we note that:
    # f_new = f + 2*s_i * e_i (where e_i is 1 on segment i, 0 elsewhere)
    # fft_f_new = fft_f + 2*s_i * fft_e_i
    # autoconv_new = ifft(fft_f_new * conj(fft_f_new))
    
    # We will use a greedy approach:
    # 1. For each bit i, estimate the change in C3 if we flip it.
    # 2. Pick the bit with the largest reduction.
    # 3. If no reduction, try a random bit (simulated annealing style) or restart.
    
    # However, evaluating all N bits every step is O(N * FFT_size) which is too slow.
    # Instead, we can use a stochastic gradient descent idea or random sampling of candidates.
    
    # Let's try a hybrid:
    # 1. Randomly sample K bits.
    # 2. Compute the exact new C3 for each.
    # 3. Pick the best one.
    
    K = 10  # Number of candidates to test per step
    
    # Pre-compute the FFT of the basis vectors for the segments?
    # fft_e_i is the FFT of a vector that is 1 on segment i and 0 elsewhere.
    # This is a shifted Dirichlet kernel.
    # It might be faster to just update the FFT directly for the candidates.
    
    # Let's optimize the inner loop.
    # We will maintain fft_f.
    # For a candidate flip at index i:
    # delta_f = 2 * s_i * e_i
    # delta_fft = 2 * s_i * fft(e_i)
    
    # We can pre-compute fft(e_i) for all i? 
    # e_i is 1 from i*ppb to (i+1)*ppb.
    # fft(e_i) = sum_{k=i*ppb}^{(i+1)*ppb - 1} exp(-2j pi k / size)
    # This is a geometric series.
    # Let's pre-compute these N complex numbers.
    
    k = np.arange(size)
    # We need fft(e_i) for i in 0..N-1
    # e_i is a rectangular window of width ppb starting at i*ppb.
    # fft(e_i) = exp(-2j pi (i*ppb) / size) * sum_{m=0}^{ppb-1} exp(-2j pi m / size)
    # The sum is a constant (Dirichlet kernel value) for all i.
    # Let S = sum_{m=0}^{ppb-1} exp(-2j pi m / size)
    # Then fft(e_i) = S * exp(-2j pi (i*ppb) / size)
    
    # Compute S
    m = np.arange(ppb)
    S = np.sum(np.exp(-2j * np.pi * m / size))
    
    # Compute the phases for each i
    phases = np.exp(-2j * np.pi * (np.arange(N) * ppb) / size)
    fft_e = S * phases  # Shape (N,)
    
    # Now, for a flip at i:
    # new_fft_f = fft_f + 2 * s_i * fft_e[i]
    # new_autoconv = ifft(new_fft_f * conj(new_fft_f)) * dx
    # new_c3 = max(abs(new_autoconv))
    
    # This is O(size) per candidate. With K=10, it's 10*size per step.
    # size = 1024. 10*1024 = 10240 ops. Very fast.
    
    np.random.seed(42)
    
    steps = 0
    while time.time() - start_time < DEADLINE:
        # Sample K random indices
        candidates = np.random.choice(N, K, replace=False)
        
        best_candidate = -1
        best_new_c3 = current_c3
        
        for idx in candidates:
            s_i = current_step[idx]
            # Flip means s_i -> -s_i, so delta is 2 * s_i
            delta_fft = 2 * s_i * fft_e[idx]
            new_fft_f = fft_f + delta_fft
            new_autoconv = np.fft.ifft(new_fft_f * np.conj(new_fft_f)).real * dx
            new_c3 = np.max(np.abs(new_autoconv))
            
            if new_c3 < best_new_c3:
                best_new_c3 = new_c3
                best_candidate = idx
        
        if best_candidate != -1:
            # Apply the best flip
            i = best_candidate
            s_i = current_step[i]
            delta_fft = 2 * s_i * fft_e[i]
            fft_f += delta_fft
            current_step[i] = -s_i
            current_c3 = best_new_c3
            
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_step = current_step.copy()
        else:
            # No improvement found in candidates.
            # To escape local minima, we can do a random jump or accept a worse move with low probability.
            # Let's try a random move with a small probability of accepting a worse state (SA-like).
            # But since we are greedy, let's just pick a random move and accept it if it's not too bad, 
            # or simply restart the search from the best state if we stagnate for too long.
            
            # For now, let's just accept the best candidate even if it's worse, but with a decay?
            # No, let's stick to greedy but add a random perturbation if we stagnate.
            
            # Check if we've been stuck
            if best_c3 == current_c3:
                 # Try a random flip anyway to explore
                 i = np.random.choice(N)
                 s_i = current_step[i]
                 delta_fft = 2 * s_i * fft_e[i]
                 new_fft_f = fft_f + delta_fft
                 new_autoconv = np.fft.ifft(new_fft_f * np.conj(new_fft_f)).real * dx
                 new_c3 = np.max(np.abs(new_autoconv))
                 
                 # Accept if it's not significantly worse (e.g., within 1%)
                 if new_c3 < best_c3 * 1.01:
                     fft_f = new_fft_f
                     current_step[i] = -s_i
                     current_c3 = new_c3
                     
        steps += 1
        if steps % 1000 == 0:
             # Optional: periodic full recompute to avoid numerical drift
             f = np.repeat(current_step, points_per_bit)
             fft_f = np.fft.fft(f)
             autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
             current_c3 = np.max(np.abs(autoconv))

    return best_step

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Best C3: {_compute_c3(res)}")
