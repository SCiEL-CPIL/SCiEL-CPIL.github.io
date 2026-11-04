"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_m_sequence(n, seed=12345):
    """Generate a pseudo-random ±1 sequence using a simple LFSR-like approach."""
    # Use a maximal-length LFSR for good autocorrelation properties
    # For N=400, we use a 9-bit LFSR (2^9 - 1 = 511) and take first 400
    # Or use a simpler approach: generate from a good PRNG
    
    # Actually, let's use a well-known approach: 
    # Generate using a linear congruential generator mapped to ±1
    # But better: use a m-sequence based approach
    
    # For N=400, let's use a 9-stage LFSR with primitive polynomial x^9 + x^4 + 1
    # This gives period 511, we take first 400
    
    # LFSR with polynomial x^9 + x^4 + 1 (taps at positions 9 and 4, 0-indexed: 8 and 3)
    state = seed & 0x1FF  # 9-bit state
    if state == 0:
        state = 1
    
    sequence = []
    for i in range(n):
        sequence.append(1 if (state & 1) == 0 else -1)
        # Feedback: bit 8 XOR bit 3
        feedback = ((state >> 8) ^ (state >> 3)) & 1
        state = ((state << 1) | feedback) & 0x1FF
    
    return np.array(sequence, dtype=np.float64)


def _compute_c3_bound(seq):
    """Compute the C3 upper bound for a signed step function."""
    # The C3 bound relates to autoconvolution
    # For a step function with values ±1, the autoconvolution at lag k is:
    # R(k) = sum_{i} seq[i] * seq[i+k]
    # The C3 inequality bound involves the maximum autocorrelation
    # or a specific combination
    
    # Common interpretation: C3 bound = max over lags of |autocorrelation| / N
    # or the peak sidelobe level
    
    # Let's compute the full autocorrelation and take the max absolute value
    # excluding the zero lag (which is always N for ±1 sequences)
    
    # Actually, for the "third autocorrelation inequality", this might be:
    # C3 <= (1/N) * max_k |R(k)| for k != 0
    # or some other specific bound
    
    # Let's use: the bound is the maximum of the normalized autocorrelation
    # at non-zero lags, which for good sequences is ~1/sqrt(N)
    
    # Compute autocorrelation efficiently using FFT
    n = len(seq)
    # Pad to next power of 2 for FFT efficiency
    size = 1
    while size < 2 * n:
        size *= 2
    
    fft_seq = np.fft.rfft(seq, n=size)
    power = fft_seq * np.conj(fft_seq)
    autocorr = np.fft.irfft(power, n=size)[:n]
    
    # autocorr[0] = N (for ±1 sequence)
    # Take max absolute value at non-zero lags
    if n > 1:
        max_side = np.max(np.abs(autocorr[1:]))
    else:
        max_side = 0
    
    # The C3 bound is likely this max sidelobe or related quantity
    # For all-ones: autocorr[k] = N-k for all k, so max at k=1 is N-1=399
    # For random ±1: max sidelobe ~ sqrt(N * log(N)) ~ sqrt(400*6) ~ 49
    
    # The "upper bound on C3" - let's define it as the max sidelobe
    # This makes all-ones give 399, and good sequences give ~50
    
    return max_side


def construct():
    """Construct a signed step function that minimizes the C3 autoconvolution bound."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate a good pseudo-random ±1 sequence using m-sequence
    best_seq = _generate_m_sequence(N, seed=12345)
    best_score = _compute_c3_bound(best_seq)
    
    # Try multiple seeds to find a better sequence
    for seed in range(2, 200):
        if time.time() > deadline:
            break
        
        seq = _generate_m_sequence(N, seed=seed)
        score = _compute_c3_bound(seq)
        
        if score < best_score:
            best_score = score
            best_seq = seq
    
    return best_seq
# EVOLVE-BLOCK-END
