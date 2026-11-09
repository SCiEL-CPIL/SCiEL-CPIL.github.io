"""Rewritten program: LFSR-based signed step function with local refinement."""
import time
import numpy as np

N = 400
DEADLINE = 140.0  # safety margin before 150s limit

def _autoconv_bound(x):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    The metric: we compute the circular autoconvolution of x with itself,
    and the bound is the maximum off-center absolute value divided by the center value.
    Lower is better.
    """
    # Use the signed metric (standard for m-sequences)
    n = len(x)
    fft_x = np.fft.fft(x)
    conv = np.fft.ifft(fft_x * fft_x).real
    # Normalize
    center = conv[0]
    if center <= 0:
        return 1.0
    # Off-center maximum absolute value
    off_vals = np.concatenate([conv[1:], conv[:-1]])
    off_max = np.max(np.abs(off_vals))
    return off_max / center


def _lfsr_sequence(n, poly_bits, init):
    """Generate n bits from an LFSR with given polynomial and initial state."""
    seq = []
    state = init
    deg = poly_bits.bit_length() - 1
    for _ in range(n):
        seq.append(state & 1)
        # Feedback: XOR of bits at positions defined by polynomial
        feedback = 0
        for i in range(deg + 1):
            if (poly_bits >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << deg)
    return seq


def _refine(x, max_time):
    """Local search: flip individual bits to reduce the autoconvolution bound.
    Uses FFT-based incremental updates for speed."""
    x = x.copy()
    n = len(x)
    best_val = _autoconv_bound(x)
    start = time.time()
    
    # Precompute FFT for faster evaluation
    fft_x = np.fft.fft(x)
    
    # Greedy coordinate descent with multiple passes
    improved = True
    passes = 0
    while improved and (time.time() - start) < max_time:
        improved = False
        passes += 1
        for i in range(n):
            if (time.time() - start) > max_time:
                break
            # Flip bit i
            old_val = x[i]
            x[i] = -old_val
            
            # Update FFT incrementally
            delta = x[i] - old_val
            fft_x[i] += delta * np.exp(-2j * np.pi * i / n)
            # Actually need to recompute properly - just use full FFT for correctness
            fft_x = np.fft.fft(x)
            conv = np.fft.ifft(fft_x * fft_x).real
            center = conv[0]
            if center <= 0:
                val = 1.0
            else:
                off_vals = np.concatenate([conv[1:], conv[:-1]])
                off_max = np.max(np.abs(off_vals))
                val = off_max / center
            
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                x[i] = old_val
                fft_x = np.fft.fft(x)
    return x, best_val


def _refine_fast(x, max_time):
    """Faster local search using direct convolution for small N."""
    x = x.copy()
    n = len(x)
    best_val = _autoconv_bound(x)
    start = time.time()
    
    # Precompute base convolution
    fft_x = np.fft.fft(x)
    conv = np.fft.ifft(fft_x * fft_x).real
    
    # Greedy coordinate descent
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        for i in range(n):
            if (time.time() - start) > max_time:
                break
            # Try flipping bit i
            x[i] *= -1
            
            # Recompute convolution
            fft_x = np.fft.fft(x)
            conv = np.fft.ifft(fft_x * fft_x).real
            center = conv[0]
            if center <= 0:
                val = 1.0
            else:
                off_vals = np.concatenate([conv[1:], conv[:-1]])
                off_max = np.max(np.abs(off_vals))
                val = off_max / center
            
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                x[i] *= -1
    return x, best_val


def construct():
    t_start = time.time()
    
    best_x = None
    best_val = float('inf')
    
    # Try several LFSR configurations and pick the best
    # Primitive polynomials for various degrees
    # We need sequences of length 400, so use a degree >= 9 (2^9-1=511 > 400)
    
    # Candidate LFSR configurations (polynomial in binary, initial state)
    # Degree 9 primitive polynomials
    configs = [
        (0b100010001, 0x1FF),  # x^9+x^4+1, all-ones init
        (0b100100001, 0x1FF),  # x^9+x^5+1
        (0b101101001, 0x1FF),  # x^9+x^6+x^5+x^3+1
        (0b100010001, 0x0FF),
        (0b100100001, 0x0FF),
        (0b101101001, 0x0FF),
        (0b100010001, 0x123),
        (0b100100001, 0x123),
        (0b101101001, 0x123),
        (0b100010001, 0x001),
        (0b100100001, 0x001),
        (0b101101001, 0x001),
        (0b100010001, 0x0F0),
        (0b100100001, 0x0F0),
        (0b101101001, 0x0F0),
        # Degree 10
        (0b1000001001, 0x3FF),
        (0b1101010001, 0x3FF),
        (0b1000001001, 0x1FF),
        (0b1101010001, 0x1FF),
        (0b1000001001, 0x0FF),
        (0b1101010001, 0x0FF),
        (0b1000001001, 0x001),
        (0b1101010001, 0x001),
        # Degree 12
        (0b1000110000101, 0xFFF),
        (0b1000000000011, 0xFFF),
        (0b1000110000101, 0x1FF),
        (0b1000000000011, 0x1FF),
    ]
    
    # Also try phase shifts of good LFSR sequences
    candidate_sequences = []
    
    for poly, init in configs:
        if time.time() - t_start > DEADLINE - 40:
            break
        bits = _lfsr_sequence(N, poly, init)
        # Convert bits to signed values: 0 -> -1, 1 -> +1
        x = np.array([1.0 if b == 1 else -1.0 for b in bits])
        val = _autoconv_bound(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        candidate_sequences.append((x.copy(), val))
    
    # Try phase shifts of the best few LFSR sequences
    if time.time() - t_start < DEADLINE - 40:
        # Sort candidates by value
        candidate_sequences.sort(key=lambda t: t[1])
        top_k = min(5, len(candidate_sequences))
        
        for k in range(top_k):
            if time.time() - t_start > DEADLINE - 40:
                break
            base_x = candidate_sequences[k][0]
            # Try different phase shifts (circular shifts)
            for shift in range(0, N, 10):  # Try every 10th shift for speed
                if time.time() - t_start > DEADLINE - 40:
                    break
                shifted_x = np.roll(base_x, shift)
                val = _autoconv_bound(shifted_x)
                if val < best_val:
                    best_val = val
                    best_x = shifted_x.copy()
    
    # Also try random initialization as fallback/comparison
    if time.time() - t_start < DEADLINE - 40:
        rng = np.random.RandomState(42)
        for seed in range(30):
            if time.time() - t_start > DEADLINE - 40:
                break
            rng = np.random.RandomState(seed)
            x = rng.choice([-1.0, 1.0], size=N)
            val = _autoconv_bound(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    if best_x is None:
        return np.ones(N)
    
    # Local refinement with remaining time
    remaining_time = DEADLINE - (time.time() - t_start) - 5  # 5s safety margin
    if remaining_time > 2.0:
        best_x, best_val = _refine_fast(best_x, remaining_time)
    
    return best_x
