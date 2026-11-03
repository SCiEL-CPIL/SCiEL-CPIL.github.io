import numpy as np
import time

N = 600

def _compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    h = np.asarray(heights, dtype=np.float64)
    # Autoconvolution: (h * h)[k] = sum_i h[i] * h[i+k] for valid i
    # Using FFT for speed
    n = len(h)
    # Pad to avoid circular convolution
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)
    # The linear autoconvolution is conv[0:n] (lags 0 to n-1)
    # Actually for autoconvolution, we want the full linear convolution
    # conv[k] for k=0..n-1 gives sum_{i=0}^{n-1-k} h[i]*h[i+k]
    # But we also need negative lags? For nonnegative functions on [0,n), 
    # the autoconvolution is symmetric, so max is at some non-negative lag
    peak = np.max(conv[:n])
    total = np.sum(h)
    if total == 0:
        return float('inf')
    # Normalized maximum: peak / total^2
    return peak / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120 for safety margin
    
    best_score = _compute_score([1.0] * N)
    best_heights = [1.0] * N
    
    # Strategy: try functions with a central "hole" (zero region)
    # and various shapes
    
    # Try different hole sizes
    for hole_frac in np.linspace(0.0, 0.8, 40):
        if time.time() > deadline - 5.0:
            break
        hole_size = int(hole_frac * N)
        if hole_size < 1:
            hole_size = 1
        left_end = (N - hole_size) // 2
        right_start = left_end + hole_size
        
        # Uniform on the non-hole regions
        h = np.zeros(N)
        if left_end > 0:
            h[:left_end] = 1.0
        if right_start < N:
            h[right_start:] = 1.0
        
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_heights = h.tolist()
    
    # Try triangular/tent shapes
    for peak_width in np.linspace(0.1, 0.9, 30):
        if time.time() > deadline - 5.0:
            break
        # Triangle: high in middle, zero at edges
        x = np.linspace(0, 1, N)
        center = 0.5
        half_width = peak_width / 2
        h = np.maximum(0, 1 - np.abs(x - center) / half_width)
        # Scale to have same sum as uniform? No, let's keep as is
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_heights = h.tolist()
        
        # Inverted: zero in middle, high at edges
        h_inv = np.maximum(0, np.abs(x - center) / half_width - (1 - peak_width))
        # Actually let's do: h = max(0, (|x-0.5| - (1-peak_width)/2) / (peak_width/2))
        if peak_width > 0.01:
            edge_width = peak_width / 2
            h_inv = np.zeros(N)
            # Left edge: from 0 to edge_width
            left_mask = x < edge_width
            h_inv[left_mask] = (edge_width - x[left_mask]) / edge_width
            # Right edge: from 1-edge_width to 1
            right_mask = x > (1 - edge_width)
            h_inv[right_mask] = (x[right_mask] - (1 - edge_width)) / edge_width
            score = _compute_score(h_inv)
            if score < best_score:
                best_score = score
                best_heights = h_inv.tolist()
    
    # Try two-bump shapes: two separated peaks
    for sep in np.linspace(0.2, 0.8, 20):
        if time.time() > deadline - 5.0:
            break
        x = np.linspace(0, 1, N)
        center1 = 0.5 - sep / 2
        center2 = 0.5 + sep / 2
        width = 0.15
        h = np.zeros(N)
        h += np.exp(-((x - center1) / width) ** 2)
        h += np.exp(-((x - center2) / width) ** 2)
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_heights = h.tolist()
    
    # Try random perturbations around best found
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        h = np.array(best_heights, dtype=np.float64)
        # Add small noise
        noise = np.random.randn(N) * 0.1
        h_new = np.maximum(0, h + noise)
        score = _compute_score(h_new)
        if score < best_score:
            best_score = score
            best_heights = h_new.tolist()
    
    return best_heights
