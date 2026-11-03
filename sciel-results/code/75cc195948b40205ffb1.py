import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid indices to coarse cell indices
# x in [-0.25, 0.25). Cell i is [x_i, x_{i+1}).
# x_i = -0.25 + i * W
# (x + 0.25) / W = i
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialize with a Gaussian centered at 0
    # x_norm in [-1, 1]
    x_norm = np.linspace(-1.0, 1.0, N)
    best_h = np.exp(-4.0 * x_norm**2)
    # Normalize
    best_h /= np.linalg.norm(best_h)
    best_val = ratio_from_h(best_h)
    
    # Symmetry mask: h[i] == h[N-1-i]
    # We will optimize the first N//2 variables and mirror them
    
    # Subgradient Descent parameters
    n_steps = 3000
    momentum = 0.9
    lr = 0.05
    top_k = 5
    
    # Precompute reflection indices for the fine grid
    # For linear convolution of length 2*FINE_N - 1, the peak is usually near FINE_N - 1 (center)
    # But we just find the global max index.
    
    cur_h = best_h.copy()
    v = best_val
    
    # Velocity for momentum
    v_h = np.zeros(N)
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Symmetrize current h
        cur_h = (cur_h + cur_h[::-1]) / 2.0
        
        # 2. Map to fine grid
        f = cur_h[CELL]
        
        # 3. Compute autoconvolution
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        
        # 4. Find top K peaks
        # conv length is 2*FINE_N - 1
        # The "center" of the convolution (lag 0) is at index FINE_N - 1
        # But the max could be elsewhere.
        # We want to reduce the peaks.
        # Get indices of top K values
        if len(conv) > top_k:
            # Use argpartition for speed if top_k is small, but sorting 8000 is fine
            top_indices = np.argpartition(conv, -top_k)[-top_k:]
            # Average the gradients from these peaks to smooth
            # Gradient for f_j w.r.t conv_k is f_{k-j}
            # We sum f_{k-j} for k in top_indices
            # g_fine[j] = sum_{k in top} f[k-j]
            
            # Efficient computation:
            # f is length FINE_N.
            # We need to sum shifted versions of f.
            # g_fine = sum_{k in top} f[shifted by k]
            
            # Note: f is defined on [0, FINE_N).
            # conv index k corresponds to lag.
            # f[k-j] is valid if 0 <= k-j < FINE_N.
            
            g_fine = np.zeros(FINE_N)
            for k in top_indices:
                # Shift f by k
                # f[k-j] -> index j in g_fine takes value f[k-j]
                # This is a reversed and shifted f.
                # Let's construct the slice.
                # We want g[j] += f[k-j]
                # j ranges 0..FINE_N-1
                # k-j ranges k..k-(FINE_N-1)
                
                # Simple loop for clarity, FINE_N is 8000, top_k is 5. 40k ops.
                # Can be vectorized with np.roll or slicing.
                
                # f_rev = f[::-1]
                # f[k-j] = f_rev[FINE_N-1-(k-j)] = f_rev[FINE_N-1-k+j]
                # So g[j] += f_rev[FINE_N-1-k+j]
                # This is a shift of f_rev.
                
                shift = FINE_N - 1 - k
                if shift >= 0:
                    # g[j] += f_rev[shift + j]
                    # Valid j where shift+j < FINE_N -> j < FINE_N - shift
                    end_j = FINE_N - shift
                    if end_j > 0:
                        g_fine[:end_j] += f[::-1][shift:shift+end_j]
                else:
                    # shift < 0
                    # g[j] += f_rev[shift + j]
                    # Valid j where shift+j >= 0 -> j >= -shift
                    start_j = -shift
                    g_fine[start_j:] += f[::-1][:FINE_N-start_j]
            
            # Map g_fine back to h
            # G_i = sum_{j in cell_i} g_fine[j] * DX ? 
            # Actually, derivative w.r.t h_i is sum of derivatives w.r.t f_j.
            # The factor DX is in the convolution definition.
            # conv = sum f f DX.
            # d conv / d h_i = sum_{j in cell_i} (d conv / d f_j) * (d f_j / d h_i)
            # d f_j / d h_i = 1 if j in cell_i.
            # So G_i = sum_{j in cell_i} g_fine[j] * DX (if g_fine is d conv / d f_j)
            # Wait, conv = sum f f DX.
            # d conv / d f_j = 2 * (f * f)_j DX? No.
            # (f*f)_k = sum_j f_j f_{k-j} DX.
            # d (f*f)_k / d f_j = f_{k-j} DX + f_{j-k} DX?
            # For real f, f_{k-j} = f_{j-k} only if symmetric? No.
            # Actually, the gradient of the scalar conv[k] w.r.t vector f is f shifted.
            # Specifically, grad_k = f[k-j] * DX.
            
            # So G_i = sum_{j in cell_i} f[k-j] * DX.
            # We accumulated g_fine[j] = sum_{k in top} f[k-j].
            # So G_i = DX * sum_{j in cell_i} g_fine[j].
            
            G = np.zeros(N)
            # Sum g_fine over cells
            # np.bincount is efficient
            G = np.bincount(CELL, weights=g_fine * DX, minlength=N)
            
            # Symmetrize Gradient
            G = (G + G[::-1]) / 2.0
            
            # 5. Update with Momentum
            v_h = momentum * v_h - lr * G
            cur_h = cur_h + v_h
            
            # 6. Clip and Normalize
            cur_h = np.clip(cur_h, 0.0, None)
            norm = np.linalg.norm(cur_h)
            if norm < 1e-15:
                cur_h = best_h.copy()
                continue
            cur_h /= norm
            
            # 7. Evaluate
            v = ratio_from_h(cur_h)
            
            if v < best_val:
                best_val = v
                best_h = cur_h.copy()
                
            # Adaptive learning rate?
            # If we are improving, keep lr. If not, reduce.
            # Simple: decay lr slightly
            lr *= 0.999
            
    # Final refinement: try a few specific shapes to see if they beat the result
    if time.time() < deadline - 2.0:
        shapes = [
            np.exp(-4.0 * x_norm**2),
            np.exp(-2.0 * x_norm**2),
            np.exp(-8.0 * x_norm**2),
            np.clip(1.0 - np.abs(x_norm), 0, None),
            np.clip(1.0 - 2.0 * np.abs(x_norm), 0, None),
        ]
        for sh in shapes:
            sh = np.clip(sh, 0, None)
            sh /= np.linalg.norm(sh)
            val = ratio_from_h(sh)
            if val < best_val:
                best_val = val
                best_h = sh

    return list(best_h)
