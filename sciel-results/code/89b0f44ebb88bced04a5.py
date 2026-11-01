import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    def compute_c3(h):
        """
        Compute the upper bound C3.
        Try: C3 = max(|R[k]| for k != 0) / R[0]
        where R is the full autocorrelation.
        """
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h.astype(float), n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N]
        
        peak = R_pos[0]
        if peak < 1e-10:
            return 1e10
        sidelobes = R_pos[1:]
        max_side = np.max(np.abs(sidelobes))
        return max_side / peak
    
    best_h = None
    best_val = float('inf')
    
    # Generate candidate sequences from various deterministic sources
    
    # 1. Random sequences (baseline)
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        h = np.random.choice([-1, 1], size=N)
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 2. Logistic map chaotic sequences
    for a in [3.5, 3.7, 3.8, 3.9, 3.99]:
        if time.time() > deadline - 10.0:
            break
        x = 0.5
        h = np.zeros(N, dtype=int)
        for i in range(N):
            x = a * x * (1 - x)
            h[i] = 1 if x > 0.5 else -1
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 3. Van der Corput low-discrepancy sequence
    def van_der_corput(n, base=2):
        """Generate van der Corput sequence of length n."""
        seq = []
        for i in range(1, n + 1):
            f = 1.0 / base
            x = 0.0
            num = i
            while num > 0:
                x += f * (num % base)
                num //= base
                f /= base
            seq.append(x)
        return np.array(seq)
    
    for base in [2, 3, 4, 5]:
        if time.time() > deadline - 10.0:
            break
        seq = van_der_corput(N, base)
        # Threshold to binary
        median_val = np.median(seq)
        h = np.where(seq > median_val, 1, -1).astype(int)
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 4. Thue-Morse sequence (known for low autocorrelation properties)
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        tm = [0]
        for i in range(1, n):
            tm.append(tm[i // 2] ^ (i % 2))
        return np.array(tm[:n])
    
    if time.time() < deadline - 10.0:
        tm = thue_morse(N)
        h = np.where(tm == 0, 1, -1).astype(int)
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 5. Try sign of sine wave with various frequencies
    for freq in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        if time.time() > deadline - 10.0:
            break
        t = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        h = np.sign(np.sin(t)).astype(int)
        # Handle zeros
        zeros = np.where(h == 0)[0]
        if len(zeros) > 0:
            h[zeros] = 1
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # 6. Try a simple local search refinement on the best candidate
    if best_h is not None and time.time() < deadline - 5.0:
        h = best_h.copy()
        current_val = best_val
        n_iterations = 2000
        for it in range(n_iterations):
            if time.time() > deadline - 5.0:
                break
            # Flip a random bit
            idx = np.random.randint(0, N)
            old_val = h[idx]
            h[idx] = -h[idx]
            new_val = compute_c3(h)
            if new_val < current_val:
                current_val = new_val
                best_val = current_val
                best_h = h.copy()
            else:
                h[idx] = old_val
    
    if best_h is None:
        return np.random.choice([-1, 1], size=N)
    
    return best_h
