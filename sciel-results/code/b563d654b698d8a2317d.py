"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 140  # Safety margin
    
    rng = np.random.default_rng(123)
    
    # Start with random ±1
    h = rng.choice([-1, 1], size=N).astype(float)
    
    # Normalize: sum h = 1
    s = np.sum(h)
    if abs(s) > 1e-10:
        h = h / s
    
    # Compute initial autoconvolution
    ac = np.convolve(h, h, mode='full')
    max_ac = np.max(np.abs(ac))
    
    best_h = h.copy()
    best_max_ac = max_ac
    
    # Local search with efficient updates
    # When flipping h[i] to -h[i], the new h is h' = h - 2*h[i]*e_i
    # ac' = conv(h', h') = conv(h, h) - 4*h[i]*conv(e_i, h) + 4*h[i]²*conv(e_i, e_i)
    # conv(e_i, h) is h shifted by i
    # conv(e_i, e_i) is a delta at 2i
    
    # Precompute h for fast access
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        # Try flipping each element, but use efficient update
        for i in range(N):
            if time.time() > deadline:
                break
            
            hi = h[i]
            if abs(hi) < 1e-15:
                continue
            
            # Compute the change in autoconvolution
            # delta_ac[k] = -4*hi * h[k-i] + 4*hi² * (1 if k==2i else 0)
            # But we need to handle bounds: k-i must be in [0, N-1] for h[k-i] to exist
            
            # The autoconvolution has length 2N-1
            # conv(e_i, h)[k] = h[k-i] if 0 <= k-i < N, else 0
            # So the range of k where conv(e_i, h) is nonzero: k in [i, i+N-1]
            
            k_start = max(0, i)
            k_end = min(2*N-2, i + N - 1)
            
            # Compute delta for the affected range
            # But we only care about the max, so let's compute the new max efficiently
            # This is still O(N) per flip, so O(N²) per iteration, O(N³) total
            
            # For N=400, N²=160K, times 200 iterations = 32M ops, should be OK in numpy
            
            # Actually, let's just do it in numpy for the affected slice
            # The change: ac_new[k] = ac[k] - 4*hi*h[k-i] + 4*hi²*(k==2i)
            
            # Create a copy of ac and update
            ac_new = ac.copy()
            
            # Update the slice
            idx_start = k_start
            idx_end = k_end + 1
            
            if idx_start < idx_end:
                h_slice = h[idx_start - i : idx_end - i]
                ac_new[idx_start:idx_end] -= 4 * hi * h_slice
            
            # Add the delta at k=2i
            if 0 <= 2*i < 2*N-1:
                ac_new[2*i] += 4 * hi * hi
            
            new_max = np.max(np.abs(ac_new))
            
            if new_max < max_ac - 1e-10:
                # Accept
                h[i] = -hi
                ac = ac_new
                max_ac = new_max
                improved = True
            else:
                # Revert (we didn't change h, just ac_new)
                pass
        
        if not improved:
            break
    
    return h
# EVOLVE-BLOCK-END
