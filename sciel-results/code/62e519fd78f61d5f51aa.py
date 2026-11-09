"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START


def _compute_c3(h):
    """
    Compute the autoconvolution upper bound C3 for a signed step function h.
    C3 is defined as the maximum absolute value of the autocorrelation at non-zero lags,
    normalized by the zero-lag value (which is N for binary +/-1 sequences).
    Lower is better.
    """
    # Compute autocorrelation via FFT
    # Pad to at least 2N to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0 (sum of squares = N)
    # r[1] to r[N-1] are positive lags
    # For real signals, autocorrelation is symmetric, so we only need positive lags
    
    # The zero-lag value
    r0 = r[0]
    
    # Max absolute value at non-zero lags
    # Positive lags: r[1:N]
    max_sidelobe = np.max(np.abs(r[1:N]))
    
    # C3 is the ratio (upper bound on normalized autocorrelation peak)
    return max_sidelobe / r0


def _generate_lfsr_sequence(n, poly, init_state):
    """
    Generate a maximal-length sequence using a linear feedback shift register.
    poly is an integer where bit i is 1 if there's feedback from bit i.
    init_state is the initial register state.
    Returns a sequence of +1/-1 of length n.
    """
    # For N=400, we need a primitive polynomial of degree m such that 2^m - 1 >= 400
    # 2^9 - 1 = 511, so m=9 works
    # We'll use a standard primitive polynomial for m=9: x^9 + x^4 + 1
    # In integer form, the taps are at positions 0, 4, 8 (or 9 depending on convention)
    
    # Let's use a simpler approach: generate a long m-sequence and take the first n bits
    # For m=9, primitive polynomial x^9 + x^4 + 1
    # The LFSR has 9 bits. We'll use the standard Fibonacci LFSR.
    
    # Actually, let's use a well-known primitive polynomial for m=9
    # One such polynomial is x^9 + x^5 + x^4 + x^3 + 1
    # Taps: 9, 5, 4, 3 (1-indexed from LSB)
    
    # Let's just use a simple approach: generate an m-sequence of length 511
    # and repeat/truncate as needed
    
    # For simplicity, let's use a known good m-sequence
    # Generate using a standard LFSR
    
    m = 9  # degree of polynomial
    # Primitive polynomial: x^9 + x^4 + 1
    # This means feedback from bit 8 and bit 4 (0-indexed)
    
    reg = init_state
    if reg == 0:
        reg = 1  # avoid all-zero state
    
    seq = []
    for i in range(n):
        seq.append(1 if (reg & 1) else -1)
        # Shift right and apply feedback
        feedback = ((reg >> 8) ^ (reg >> 4)) & 1
        reg = (reg >> 1) | (feedback << 8)
    
    return np.array(seq, dtype=np.float64)


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Generate LFSR-based initial sequences
    # Try several different primitive polynomials and initial states
    rng = np.random.default_rng(42)
    
    best_h = None
    best_c3 = np.inf
    
    # Try multiple LFSR configurations
    lfsr_configs = []
    
    # m=9, primitive polynomial x^9 + x^4 + 1
    for init in range(1, 32):
        lfsr_configs.append(('poly1', init))
    
    # Also try some other approaches
    # Generate random sequences as fallback
    for i in range(50):
        lfsr_configs.append(('random', i))
    
    # First, evaluate LFSR sequences
    for config_type, param in lfsr_configs:
        if time.time() - start_time > deadline - 10:
            break
        
        if config_type == 'poly1':
            h = _generate_lfsr_sequence(N, 0, param)
        elif config_type == 'random':
            h = rng.choice([-1, 1], size=N).astype(np.float64)
        else:
            continue
        
        c3 = _compute_c3(h)
        if c3 < best_c3:
            best_c3 = c3
            best_h = h.copy()
    
    # Now do local search refinement from the best LFSR sequence
    h = best_h.copy()
    no_improve_count = 0
    max_no_improve = 10000
    
    # Use coordinate descent with periodic random restarts
    while time.time() - start_time < deadline:
        if time.time() - start_time > deadline - 5:
            break
        
        # Try flipping a random position
        idx = rng.integers(0, N)
        old_val = h[idx]
        h[idx] *= -1
        
        new_c3 = _compute_c3(h)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            best_h = h.copy()
            no_improve_count = 0
        else:
            h[idx] = old_val
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Try a few random restarts
            for _ in range(5):
                if time.time() - start_time > deadline - 5:
                    break
                h_restart = rng.choice([-1, 1], size=N).astype(np.float64)
                c3_restart = _compute_c3(h_restart)
                if c3_restart < best_c3:
                    best_c3 = c3_restart
                    best_h = h_restart.copy()
                    h = best_h.copy()
                    no_improve_count = 0
                    break
            else:
                no_improve_count = 0
    
    return best_h
# EVOLVE-BLOCK-END
