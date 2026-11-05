"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / sum(h)^2)
    
    Key insight: np.convolve(h,h) is convolution, not correlation.
    For constant h, conv has peak N at center, sum(h)=N, value=2.
    
    Lower bound: sum_k c_k = (sum h)^2, so max(c_k) >= (sum h)^2/(2N-1)
    Thus value >= 2N/(2N-1) ~ 1.0.
    
    Strategy: Use real part of complex Zadoff-Chu * chirp with window to
    flatten the convolution. Search over parameters.
    """
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2.0 * N * m / (s * s))
        return val if np.isfinite(val) else float('inf')
    
    best_h = np.ones(N)
    best_val = evaluate(best_h)
    
    t = np.arange(N, dtype=float)
    
    # Search over Zadoff-Chu root, chirp rate, and window width
    for root in range(1, 16):
        for chirp_rate in np.linspace(-2.0, 2.0, 9):
            for win_width in [0.3, 0.5, 0.7, 1.0]:
                # Complex Zadoff-Chu
                zc = np.exp(-1j * np.pi * root * t * (t + 1) / N)
                # Complex chirp modulation
                chirp = np.exp(1j * chirp_rate * np.pi * t * t / N)
                # Combined complex signal
                z = zc * chirp
                # Real part
                h_real = np.real(z)
                # Apply raised cosine window
                if win_width < 1.0:
                    # Window that is 1 in center and tapers at edges
                    x = np.linspace(-1, 1, N)
                    win = np.ones(N)
                    edge = int((1 - win_width) * N / 2)
                    if edge > 0:
                        # Taper first and last 'edge' samples
                        taper = np.cos(np.pi * np.arange(edge) / (2 * edge)) ** 2
                        win[:edge] = taper
                        win[-edge:] = taper[::-1]
                    h_real = h_real * win
                
                # Also try adding a DC offset to ensure sum != 0
                for dc_offset in [0.0, 0.1, 0.2, 0.5, 1.0]:
                    h_try = h_real + dc_offset
                    val = evaluate(h_try)
                    if val < best_val:
                        best_val = val
                        best_h = h_try.copy()
    
    # Also try pure cosines with DC offset
    for freq in range(1, 30):
        for dc in [0.0, 0.1, 0.2, 0.5, 1.0]:
            h_try = np.cos(2 * np.pi * freq * t / N) + dc
            val = evaluate(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    # Try sine with DC
    for freq in range(1, 30):
        for dc in [0.0, 0.1, 0.2, 0.5, 1.0]:
            h_try = np.sin(2 * np.pi * freq * t / N) + dc
            val = evaluate(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    return best_h
# EVOLVE-BLOCK-END
