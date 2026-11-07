"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Start with the seed
    best_h = np.full(N, 0.5)
    best_val = 1.0  # seed value
    
    # Try a simple alternating pattern: 0, 1, 0, 1, ...
    # This should give very low correlation
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    # Check sum: 50 ones + 50 zeros = 50, which is N/2 = 50. Good.
    val_alt = max(np.correlate(h_alt, 1 - h_alt, mode="full")) / N * 2
    if val_alt < best_val:
        best_h = h_alt.copy()
        best_val = val_alt
    
    # Try a few other simple patterns
    # Pattern: 0, 0, 1, 1, 0, 0, 1, 1, ... (period 4)
    h_p4 = np.zeros(N)
    h_p4[2::4] = 1.0
    h_p4[3::4] = 1.0
    # Sum: 50 ones. Good.
    val_p4 = max(np.correlate(h_p4, 1 - h_p4, mode="full")) / N * 2
    if val_p4 < best_val:
        best_h = h_p4.copy()
        best_val = val_p4
    
    # Pattern: 0, 1, 1, 0, 0, 1, 1, 0, ... (period 4, shifted)
    h_p4b = np.zeros(N)
    h_p4b[1::4] = 1.0
    h_p4b[2::4] = 1.0
    # Sum: 50 ones. Good.
    val_p4b = max(np.correlate(h_p4b, 1 - h_p4b, mode="full")) / N * 2
    if val_p4b < best_val:
        best_h = h_p4b.copy()
        best_val = val_p4b
    
    # Try a sinusoidal pattern
    t = np.linspace(0, 2*np.pi*5, N, endpoint=False)  # 5 cycles
    h_sin = 0.5 + 0.5 * np.sin(t)
    # Adjust to ensure sum is exactly N/2
    diff = N/2 - np.sum(h_sin)
    h_sin += diff / N
    # Clip to [0, 1]
    h_sin = np.clip(h_sin, 0, 1)
    # Re-adjust sum after clipping
    diff = N/2 - np.sum(h_sin)
    if diff > 0:
        # Need to increase some values
        mask = h_sin < 1.0
        if np.sum(mask) > 0:
            h_sin[mask] += diff / np.sum(mask)
            h_sin = np.clip(h_sin, 0, 1)
    else:
        # Need to decrease some values
        mask = h_sin > 0.0
        if np.sum(mask) > 0:
            h_sin[mask] += diff / np.sum(mask)
            h_sin = np.clip(h_sin, 0, 1)
    
    val_sin = max(np.correlate(h_sin, 1 - h_sin, mode="full")) / N * 2
    if val_sin < best_val:
        best_h = h_sin.copy()
        best_val = val_sin
    
    # Try a few more sinusoidal frequencies
    for freq in [3, 7, 10, 13, 17]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h_s = 0.5 + 0.5 * np.sin(t)
        diff = N/2 - np.sum(h_s)
        h_s += diff / N
        h_s = np.clip(h_s, 0, 1)
        diff = N/2 - np.sum(h_s)
        if diff > 0:
            mask = h_s < 1.0
            if np.sum(mask) > 0:
                h_s[mask] += diff / np.sum(mask)
                h_s = np.clip(h_s, 0, 1)
        else:
            mask = h_s > 0.0
            if np.sum(mask) > 0:
                h_s[mask] += diff / np.sum(mask)
                h_s = np.clip(h_s, 0, 1)
        
        val_s = max(np.correlate(h_s, 1 - h_s, mode="full")) / N * 2
        if val_s < best_val:
            best_h = h_s.copy()
            best_val = val_s
        
        if time.time() > deadline:
            break
    
    return best_h
# EVOLVE-BLOCK-END
