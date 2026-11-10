import numpy as np
import time

def construct():
    # Golay-Rudin-Shapiro sequence of length 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:24], dtype=np.float64)
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # FFT size for evaluation
    N = 1 << 16  # 65536 points
    
    def get_score(c):
        # Compute FFT
        # c is length 24, pad to N
        # We want P(z_j) for z_j = exp(2 pi i j / N)
        # This is the DFT of c padded with zeros to length N
        # But standard FFT computes sum c_k exp(-2 pi i k j / N)
        # So we need to be careful with signs or just use the magnitude
        # |P(z)| = |sum c_k z^k|
        # Let's just compute the FFT of the padded sequence
        padded = np.zeros(N, dtype=np.float64)
        padded[:24] = c
        F = np.fft.fft(padded)
        # The magnitude is |F|
        return np.max(np.abs(F))
    
    best_coeffs = coeffs.copy()
    best_score = get_score(best_coeffs)
    
    # Hill climbing
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(24):
            # Flip coefficient i
            new_coeffs = best_coeffs.copy()
            new_coeffs[i] *= -1
            new_score = get_score(new_coeffs)
            if new_score < best_score:
                best_coeffs = new_coeffs
                best_score = new_score
                improved = True
                # Optional: restart the loop to try other flips from the new state
                # Or continue with the next i. Continuing is fine.
        if not improved:
            # Try a few random restarts or kicks if stuck
            # Since dimension is small, we can try a few random flips
            for _ in range(10):
                kick_coeffs = best_coeffs.copy()
                for _ in range(3):
                    idx = np.random.randint(0, 24)
                    kick_coeffs[idx] *= -1
                kick_score = get_score(kick_coeffs)
                if kick_score < best_score:
                    best_coeffs = kick_coeffs
                    best_score = kick_score
                    improved = True
                    break
            if not improved:
                break
                
    return best_coeffs.astype(int).tolist()
