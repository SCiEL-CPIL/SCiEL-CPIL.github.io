"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400
h = 1.0 / 200.0  # Interval width

# EVOLVE-BLOCK-START
import sys

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _autoconv_bound(f):
    """
    Compute max_t |f*f(t)| for a step function f on [-1/4, 1/4] with N intervals.
    f is an array of N heights (typically +/- 1).
    Uses FFT for efficient convolution.
    """
    # Linear convolution via FFT
    # We need to compute the autocorrelation.
    # Let F be the FFT of the padded signal.
    # Autocorrelation R(k) = ifft(|F|^2)[k]
    # We want the max absolute value of this correlation.
    
    # Pad to next power of 2 >= 2*N to avoid circular convolution artifacts
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    
    # Zero-pad f
    F = np.zeros(nfft, dtype=np.complex128)
    F[:N] = f
    
    # Compute power spectrum
    F_fft = np.fft.rfft(F)
    power = F_fft * np.conj(F_fft)
    
    # Inverse FFT to get autocorrelation
    # Note: The scaling factor for ifft is 1/nfft.
    # The integral approximation involves multiplying by h.
    # conv[k] approximates int f(s)f(t-s)ds.
    # Let's check the scaling carefully.
    # If f is a step function with height 1 on intervals of width h,
    # the continuous autocorrelation at lag 0 is the energy, which is sum(f_i^2 * h) = N * 1 * h = 400 * (1/200) = 2.
    # Let's verify the FFT output scaling.
    # np.fft.irfft(np.fft.rfft(x)**2) * h
    # This should approximate the linear convolution.
    
    conv = np.fft.irfft(power) * h
    
    # The convolution result corresponds to lags.
    # For linear convolution of two length N signals, the result has length 2N-1.
    # The peak (lag 0) is at index N-1 in the standard convolution output.
    # In the FFT-based circular convolution of length nfft, the lag 0 is at index 0 (or nfft-1 depending on shift).
    # Actually, np.fft.irfft returns the unshifted inverse.
    # The circular autocorrelation r[k] = sum_n f[n] f[n-k].
    # For linear autocorrelation, we need to handle the wrap-around.
    # Since we padded with zeros, the circular autocorrelation of the padded signal
    # is equivalent to the linear autocorrelation for lags where no wrap-around occurs.
    # The valid lags for linear autocorrelation of length N are -(N-1) to N-1.
    # In the circular buffer of size nfft, these correspond to indices 0..N-1 and nfft-N+1..nfft-1.
    # However, simply taking the max of the whole array might include artifacts if padding is insufficient, but nfft >= 2N ensures no overlap for the main lobes.
    # The max absolute value will be at the center (lag 0) or near it.
    
    return float(np.max(np.abs(conv)))

def _initial_sequence(seed):
    """Generate a random +/- 1 sequence."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _local_search(seq, time_limit, rng):
    """
    Perform local search (hill climbing with simulated annealing) on the sequence.
    Returns the best sequence found.
    """
    current_seq = seq.copy()
    current_val = _autoconv_bound(current_seq)
    
    best_seq = current_seq.copy()
    best_val = current_val
    
    # Simulated Annealing parameters
    # Start with a higher temperature to allow worse moves, cool down
    T = 1.0 
    cooling_rate = 0.995
    # We will iterate until time runs out or temperature is very low
    # and we've stuck for a while.
    
    # Precompute indices for potential flips
    # We can flip single bits or small blocks. Single bits are faster to evaluate?
    # No, FFT is O(N log N). Flipping one bit changes the FFT significantly.
    # There is an O(N) update for autocorrelation if we flip one bit, but implementation is complex.
    # Given N=400, FFT is very fast. Let's stick to full FFT evaluation per move.
    
    stagnation_counter = 0
    max_stagnation = 500
    
    while time.time() < time_limit:
        # Choose a random index to flip
        idx = rng.randint(N)
        
        # Flip the bit
        old_val = current_seq[idx]
        current_seq[idx] = -old_val
        
        # Evaluate new value
        new_val = _autoconv_bound(current_seq)
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            # Accept
            current_val = new_val
            stagnation_counter = 0
            
            if new_val < best_val:
                best_val = new_val
                best_seq = current_seq.copy()
        else:
            # Reject
            current_seq[idx] = old_val
            stagnation_counter += 1
        
        # Cool down
        T *= cooling_rate
        
        # If we've been stuck for too long and T is low, restart or break
        if stagnation_counter > max_stagnation and T < 0.01:
            break
            
        # Check time frequently
        if time.time() > time_limit:
            break
            
    return best_seq, best_val

def construct():
    """
    Entry point. Constructs a signed step function with 400 equal intervals.
    Returns a numpy array of shape (400,) with values +/- 1.
    """
    # Deadline for the search
    search_deadline = _START_TIME + _DEADLINE - 5.0  # Leave 5s for cleanup/return
    
    # Use multiple restarts to escape local minima
    best_overall_seq = None
    best_overall_val = float('inf')
    
    # Number of restarts is not fixed, but determined by time.
    # Each restart does a local search.
    # We'll try a few random initializations.
    
    # Initial seed for reproducibility of the search process itself
    rng = np.random.RandomState(42)
    
    restart_count = 0
    while time.time() < search_deadline:
        # Generate a random starting sequence
        # Sometimes random is good, sometimes structured is good.
        # Let's mix: 50% random, 50% random with some structure (e.g., alternating blocks)
        if restart_count % 2 == 0:
            start_seq = _initial_sequence(rng.randint(100000))
        else:
            # Random block lengths
            start_seq = np.zeros(N)
            pos = 0
            while pos < N:
                block_len = rng.randint(1, 10)
                sign = 1.0 if rng.random() > 0.5 else -1.0
                start_seq[pos:pos+block_len] = sign
                pos += block_len
            # Ensure length is exactly N
            if pos > N:
                 start_seq = start_seq[:N]
            elif pos < N:
                 start_seq[pos:] = 1.0 if rng.random() > 0.5 else -1.0
        
        # Run local search for a limited time or until convergence
        # We allocate a fraction of the remaining time to this restart
        # Or just run until the global deadline is close.
        # A better strategy: run local search for a fixed number of iterations or a time slice.
        # Let's give each restart a time slice of, say, 10% of remaining time, or a minimum of 1s.
        
        remaining_time = search_deadline - time.time()
        if remaining_time < 1.0:
            break
            
        # Cap the time for this specific restart to allow for multiple restarts
        # If we have lots of time, do longer searches.
        restart_time_limit = min(remaining_time, 10.0) 
        
        best_seq, best_val = _local_search(start_seq, time.time() + restart_time_limit, rng)
        
        if best_val < best_overall_val:
            best_overall_val = best_val
            best_overall_seq = best_seq
        
        restart_count += 1
        
        # Check if we should stop
        if time.time() > search_deadline:
            break
            
    if best_overall_seq is None:
        # Fallback if no search was performed (should not happen)
        return np.ones(N)
        
    return best_overall_seq
# EVOLVE-BLOCK-END
