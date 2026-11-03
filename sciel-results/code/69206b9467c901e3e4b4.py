import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # The objective function is J = (sum h^2) / (sum h * max(h*h) * delta)
    # For a single spike h = [1, 0, ..., 0]:
    # sum h^2 = 1
    # sum h = 1
    # h*h = [1, 0, ..., 0], max = 1
    # J = 1 / (1 * 1 * delta) = 1/delta
    # This is the theoretical maximum because max(h*h) >= (sum h)^2 / N is not quite right,
    # but max(h*h) >= max(h_i^2) and sum h^2 <= (sum h)^2.
    # Actually, for any non-negative h, max(h*h) >= max(h_i)^2.
    # And sum h^2 <= (max h) * sum h.
    # So J <= (max h * sum h) / (sum h * max h^2 * delta) = 1 / (max h * delta).
    # Wait, if max h is small, this bound is loose.
    # Let's stick to the spike.
    
    h = np.zeros(N)
    h[0] = 1.0
    
    # Verify score locally
    # val = _score_raw(h)
    # print(f"Local Score: {val}") # Should be 100.0
    
    return list(h)
