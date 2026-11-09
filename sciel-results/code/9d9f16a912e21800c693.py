import numpy as np
import time

N = 100

def compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(h, s)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    # Start with uniform
    h = np.full(N, 0.5)
    best_score = compute_score(h)
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try some structured patterns
    # Quadratic residue pattern
    for pattern_name, h_test in [
        ("alternating", np.tile([0.0, 1.0], N//2)),
        ("block", np.concatenate([np.full(N//4, 1.0), np.full(N//2, 0.0), np.full(N//4, 1.0)])),
    ]:
        # Project to have sum N/2
        h_test = h_test.copy()
        h_test = np.clip(h_test, 0, 1)
        # Adjust to have exact sum
        diff = N/2 - np.sum(h_test)
        if abs(diff) > 1e-10:
            # Distribute difference
            h_test += diff / N
            h_test = np.clip(h_test, 0, 1)
            # Re-adjust
            diff2 = N/2 - np.sum(h_test)
            if abs(diff2) > 1e-10:
                h_test += diff2 / N
        
        score = compute_score(h_test)
        if score < best_score:
            best_score = score
            h = h_test.copy()
    
    return h
