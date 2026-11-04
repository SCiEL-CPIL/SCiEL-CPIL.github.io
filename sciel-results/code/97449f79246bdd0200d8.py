"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """
    Compute the upper bound on C3: the maximum absolute value of the
    autoconvolution (f * f) over all non-zero lags.
    """
    n = len(f)
    conv_len = 2 * n - 1
    fft_size = 1
    while fft_size < conv_len:
        fft_size *= 2
    
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:conv_len]
    
    if conv_len <= 1:
        return 0.0
    
    max_val = np.max(np.abs(conv[1:conv_len]))
    return max_val


def _legendre_sequence(n):
    """
    Generate a Legendre sequence of length n.
    A Legendre sequence is defined by the quadratic residue symbol modulo p.
    For p = 2n+1 (prime), the sequence has good autocorrelation properties.
    """
    # Find the smallest prime p >= 2*n+1
    p = 2 * n + 1
    while True:
        # Check if p is prime
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 2 if p > 2 else 1
    
    # Generate Legendre sequence
    seq = np.zeros(n)
    for i in range(n):
        # Legendre symbol (i+1/p)
        val = pow(i + 1, (p - 1) // 2, p)
        if val == 1:
            seq[i] = 1.0
        elif val == p - 1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0  # Should not happen for i+1 < p
    
    return seq


def _m_sequence(n):
    """
    Generate an m-sequence (maximum length sequence) of length n.
    Uses a primitive polynomial over GF(2).
    For n=400, we need a primitive polynomial of degree 9 (since 2^9=512>400).
    """
    # Use a known primitive polynomial for degree 9: x^9 + x^4 + 1
    # This gives a sequence of length 511
    degree = 9
    # Primitive polynomial coefficients (bit representation)
    # x^9 + x^4 + 1 = 0x002 0000 0010 (bit 9, bit 4, bit 0)
    # In binary: 100010001 (bits 9,4,0 set)
    poly = (1 << degree) | (1 << 4) | 1  # 512 + 16 + 1 = 529
    
    # Generate m-sequence
    state = 1
    seq = []
    for _ in range(2**degree - 1):
        seq.append(1.0 if (state & 1) == 1 else -1.0)
        # Shift and feedback
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= poly
    
    # Take first n elements
    return np.array(seq[:n])


def _simulated_annealing(f, deadline, initial_temp=10.0, cooling_rate=0.999):
    """
    Simulated annealing to minimize the autoconvolution upper bound.
    """
    current_score = _autoconv_max_abs(f)
    best_f = f.copy()
    best_score = current_score
    temp = initial_temp
    rng = np.random.RandomState()
    
    while time.time() < deadline and temp > 1e-6:
        # Try flipping a random position
        pos = rng.randint(0, N)
        f[pos] = -f[pos]
        new_score = _autoconv_max_abs(f)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0 or (temp > 0 and np.random.random() < np.exp(-delta / temp)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            f[pos] = -f[pos]  # Revert
        
        temp *= cooling_rate
    
    return best_f, best_score


def construct():
    """
    Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try multiple initializations: random, Legendre, m-sequence
    initializations = []
    
    # Random initialization
    rng = np.random.RandomState(42)
    initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # Legendre sequence initialization
    initializations.append(_legendre_sequence(N))
    
    # M-sequence initialization
    initializations.append(_m_sequence(N))
    
    # Add a few more random initializations
    for _ in range(3):
        initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    best_f = None
    best_score = float('inf')
    
    for f_init in initializations:
        if time.time() > deadline:
            break
        
        # Simulated annealing from this initial point
        f_result, score = _simulated_annealing(f_init.copy(), deadline)
        
        if score < best_score:
            best_score = score
            best_f = f_result.copy()
    
    # If we haven't found anything, use the first initialization
    if best_f is None:
        best_f = initializations[0]
    
    return best_f
# EVOLVE-BLOCK-END
