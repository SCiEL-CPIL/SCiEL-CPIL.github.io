"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lfsr_m_sequence(length, poly=0x187):
    """Generate an LFSR m-sequence and take first `length` bits as +1/-1."""
    # Use a 9-bit LFSR with polynomial x^9 + x^5 + 1 (0x187 = 1 1000 0111)
    # Actually let's use a known good polynomial for degree 9: x^9 + x^5 + 1
    # Tap positions (0-indexed from LSB): 0, 4, 8
    # Let's use degree 8: x^8 + x^6 + x^5 + x^4 + 1 -> taps at 0, 3, 4, 5, 7
    # Or simpler: use numpy's random with a good seed, but LFSR is better
    
    # For N=400, let's use a 9-bit LFSR (period 511)
    # Polynomial: x^9 + x^5 + 1, taps at bit 0 and bit 4 (feedback = bit0 XOR bit4)
    state = 0x1FF  # non-zero initial state
    seq = []
    for _ in range(length):
        bit = state & 1
        seq.append(1 if bit == 0 else -1)
        # Feedback: XOR of bit 0 and bit 4 (for x^9 + x^5 + 1)
        fb = (state & 1) ^ ((state >> 4) & 1)
        state = (state >> 1) | (fb << 8)
    
    return np.array(seq[:length], dtype=np.float64)


def _autocorr_peak(x):
    """Compute the maximum absolute value of the periodic autocorrelation."""
    n = len(x)
    # Use FFT for fast autocorrelation
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # The peak is the max absolute value excluding lag 0 (or including, depending on definition)
    # For the C3 bound, we typically care about the max off-peak autocorrelation
    # Let's return max of |acf| for all lags (including 0 might be too large)
    # Actually, the "upper bound on C3" likely relates to the peak sidelobe
    # Let's compute max |acf[k]| for k != 0
    peak = np.max(np.abs(acf[1:]))
    return peak


def _autocorr_peak_incl(x):
    """Compute max absolute autocorrelation including lag 0."""
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(acf))


def _flip_and_eval(x, idx):
    """Flip bit at idx and return new peak autocorrelation."""
    x[idx] = -x[idx]
    peak = _autocorr_peak(x)
    x[idx] = -x[idx]  # revert
    return peak


def _local_search(x, max_time=100.0):
    """Local search to minimize peak autocorrelation."""
    start_time = time.time()
    best_x = x.copy()
    best_peak = _autocorr_peak(x)
    n = len(x)
    
    # Multi-pass local search
    while time.time() - start_time < max_time:
        improved = False
        for idx in range(n):
            if time.time() - start_time > max_time:
                break
            new_peak = _flip_and_eval(x, idx)
            if new_peak < best_peak - 1e-12:
                x[idx] = -x[idx]
                best_peak = new_peak
                best_x = x.copy()
                improved = True
        if not improved:
            # Try random restarts of local search from perturbed states
            if time.time() - start_time > max_time - 10:
                break
            # Perturb: flip 5 random bits
            for _ in range(5):
                idx = np.random.randint(n)
                x[idx] = -x[idx]
            x_peak = _autocorr_peak(x)
            if x_peak < best_peak:
                best_peak = x_peak
                best_x = x.copy()
    
    return best_x, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with LFSR m-sequence
    np.random.seed(42)
    x = _lfsr_m_sequence(N)
    
    # Initial peak
    current_peak = _autocorr_peak(x)
    best_x = x.copy()
    best_peak = current_peak
    
    # Local search
    best_x, best_peak = _local_search(x, max_time=100.0)
    
    # Check deadline
    if time.time() > deadline:
        return best_x
    
    # Try a few more local searches from different LFSR initializations
    for seed in range(1, 20):
        if time.time() > deadline - 5.0:
            break
        np.random.seed(seed)
        x2 = _lfsr_m_sequence(N)
        # Small perturbation
        for _ in range(3):
            idx = np.random.randint(N)
            x2[idx] = -x2[idx]
        
        x2, peak2 = _local_search(x2, max_time=min(10.0, deadline - time.time() - 2.0))
        
        if peak2 < best_peak - 1e-12:
            best_peak = peak2
            best_x = x2
    
    return best_x
# EVOLVE-BLOCK-END
