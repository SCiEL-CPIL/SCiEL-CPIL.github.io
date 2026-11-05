import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function (represented as a sequence of +/-1)
    that minimizes the Peak Sidelobe Level (PSL) of its autoconvolution.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initialize with random signs
    h = np.random.choice([-1, 1], N)
    
    # Helper to compute PSL using FFT
    # We pad to next power of 2 >= 2*N to avoid circular convolution artifacts
    fft_len = 1024  # 2^10, sufficient for 2*400=800
    
    def calc_psl(seq):
        # Compute autoconvolution via FFT
        # Zero-pad sequence
        seq_padded = np.zeros(fft_len)
        seq_padded[:N] = seq
        
        H = np.fft.fft(seq_padded)
        C = np.fft.ifft(H * np.conj(H)).real
        
        # The linear autoconvolution lags 0 to N-1 are in C[0:N]
        # We want max absolute value of lags 1 to N-1
        # Note: C is real and symmetric, C[k] = C[-k]
        # Lags 1..N-1 correspond to indices 1..N-1 in the first half
        lags = C[1:N]
        return np.max(np.abs(lags))
    
    current_cost = calc_psl(h)
    best_cost = current_cost
    best_h = h.copy()
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.1
    
    # Main optimization loop
    while time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(N)
        h[idx] *= -1
        
        new_cost = calc_psl(h)
        
        # Acceptance criterion
        if new_cost < best_cost:
            best_cost = new_cost
            best_h = h.copy()
            current_cost = new_cost
        elif np.random.rand() < np.exp(-(new_cost - current_cost) / T):
            # Accept worse solution to escape local minima
            current_cost = new_cost
        else:
            # Reject and revert
            h[idx] *= -1
            
        # Cool down
        if T > min_T:
            T *= cooling_rate
            
    return best_h
# EVOLVE-BLOCK-END
