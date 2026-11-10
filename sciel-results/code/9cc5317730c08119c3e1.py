# EVOLVE-BLOCK-START

_KNOWN_GOOD = [
    [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
    [1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
    [1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
]

def _fast_sup_norm(coeffs):
    """Fast sup norm using 4096 points."""
    import numpy as np
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(4096) / 4096
    zs = np.exp(1j * angles)
    
    # Evaluate polynomial
    vals = np.zeros(4096, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    
    # Refine top 5 peaks
    top_k = 5
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = 0.0
    for idx in peak_indices:
        center = angles[idx]
        window = 2 * np.pi / 4096
        # Simple golden section or just dense check around peak
        # For speed, just take the max of the sampled values in a small window
        # Or use a few points
        for dtheta in np.linspace(-window, window, 5):
            z = np.exp(1j * (center + dtheta))
            val = 0
            for c in reversed(coeffs):
                val = val * z + c
            m = abs(val)
            if m > best_val:
                best_val = m
    return best_val

def construct():
    best_seq = None
    best_score = float('inf')
    for seq in _KNOWN_GOOD:
        score = _fast_sup_norm(seq)
        if score < best_score:
            best_score = score
            best_seq = seq
    return list(best_seq)

# EVOLVE-BLOCK-END
