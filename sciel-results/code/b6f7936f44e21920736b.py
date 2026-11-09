"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # 100s deadline with 20s safety margin
    
    def compute_C3(seq):
        """Compute max normalized autocorrelation (excluding lag 0)."""
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(seq, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft)[:N]
        norms = np.arange(N, 0, -1, dtype=np.float64)
        normalized = np.abs(acf[1:]) / norms[1:]
        return float(np.max(normalized)) if len(normalized) > 0 else 0.0
    
    # Generate multiple initial candidates and pick the best
    rng = np.random.RandomState(42)
    
    candidates = []
    
    # Random initializations
    for _ in range(5):
        candidates.append(rng.choice([-1, 1], size=N).astype(np.float64))
    
    # Thue-Morse-like pattern
    def thue_morse(n):
        tm = np.zeros(n, dtype=np.float64)
        for i in range(n):
            # Count bits
            c = bin(i).count('1')
            tm[i] = 1.0 if c % 2 == 0 else -1.0
        return tm
    
    candidates.append(thue_morse(N))
    
    # Quadratic residue pattern (mod prime)
    p = 401  # prime > N
    qr = np.zeros(N, dtype=np.float64)
    for i in range(N):
        val = (i * i) % p
        # Legendre symbol
        if val == 0:
            qr[i] = 1.0
        else:
            # Check if quadratic residue mod p
            if pow(val, (p-1)//2, p) == 1:
                qr[i] = 1.0
            else:
                qr[i] = -1.0
    candidates.append(qr)
    
    # M-sequence-like: use LFSR
    def m_sequence(n, poly_bits):
        """Generate m-sequence-like ±1 sequence."""
        seq = np.zeros(n, dtype=np.float64)
        state = 1
        for i in range(n):
            seq[i] = 1.0 if (state & 1) == 0 else -1.0
            # LFSR step
            feedback = 0
            for b in poly_bits:
                feedback ^= (state >> b) & 1
            state = (state >> 1) | (feedback << (len(poly_bits) - 1))
        return seq
    
    # 9-bit LFSR (maximal length 511)
    candidates.append(m_sequence(N, [8, 0]))
    
    # Pick best candidate
    best_seq = None
    best_score = float('inf')
    for cand in candidates:
        s = compute_C3(cand)
        if s < best_score:
            best_score = s
            best_seq = cand.copy()
    
    current_seq = best_seq.copy()
    current_score = best_score
    
    # Simulated annealing
    T = 0.03
    T_min = 0.0005
    cooling = 0.9995
    
    while time.time() < deadline - 5:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Random flip
        idx = rng.randint(N)
        new_seq = current_seq.copy()
        new_seq[idx] *= -1
        
        new_score = compute_C3(new_seq)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_seq = new_seq
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = current_seq.copy()
    
    return best_seq
# EVOLVE-BLOCK-END
