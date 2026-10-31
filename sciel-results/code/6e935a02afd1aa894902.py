import numpy as np
import time

N = 600
DEADLINE = 115  # seconds, keep safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    def compute_conv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv
    
    # Parameterize h as a low-order cosine series to ensure smoothness and non-negativity
    # h_k = sum_{m=0}^{M} a_m * cos(2*pi*m*k/N)
    # This restricts the search to smooth, symmetric functions which are likely optimal
    M = 20  # Number of cosine terms
    
    i = np.arange(N, dtype=float)
    
    def coeffs_to_h(c):
        """Convert cosine coefficients to heights, ensuring non-negativity."""
        h = np.zeros(N)
        for m in range(len(c)):
            h += c[m] * np.cos(2 * np.pi * m * i / N)
        # Ensure non-negativity by clipping and renormalizing
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s > 0:
            h /= s
        else:
            h = np.ones(N) / N
        return h
    
    def h_to_coeffs(h):
        """Convert heights to cosine coefficients via DFT."""
        H_freq = np.fft.rfft(h)
        c = np.zeros(M + 1)
        c[0] = H_freq[0].real / N
        for m in range(1, M + 1):
            if m < len(H_freq):
                c[m] = 2 * H_freq[m].real / N
        return c
    
    def compute_grad_coeff(c, beta=50.0):
        """Compute gradient of smooth max autoconvolution w.r.t. coefficients."""
        h = coeffs_to_h(c)
        
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)
        
        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        W = np.fft.fft(w_full, n=size)
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        
        # Chain rule: d(conv_max)/dc_m = sum_k grad_h[k] * cos(2*pi*m*k/N)
        grad_c = np.zeros(len(c))
        for m in range(len(c)):
            grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))
        
        smooth_val = m + np.log(np.sum(weights)) / beta
        return grad_c, smooth_val
    
    # Generate diverse initializations in coefficient space
    inits_coeffs = []
    
    # Uniform
    inits_coeffs.append(np.array([1.0/N] + [0.0]*M))
    
    # Raised cosine
    rc = 0.5 * (1 + np.cos(2 * np.pi * (i - N/2) / N))
    rc = np.maximum(rc, 0.01)
    rc = rc / np.sum(rc)
    inits_coeffs.append(h_to_coeffs(rc))
    
    # Gaussians
    for sigma in [N/6, N/5, N/4, N/3]:
        gauss = np.exp(-((i - N/2) ** 2) / (2 * sigma ** 2))
        gauss = gauss / np.sum(gauss)
        inits_coeffs.append(h_to_coeffs(gauss))
    
    # U-shaped
    u_shape = 1.0 + 3.0 * (1 - np.sin(np.pi * i / N))
    u_shape = u_shape / np.sum(u_shape)
    inits_coeffs.append(h_to_coeffs(u_shape))
    
    # Triangular
    tri = 1 - np.abs(i - N/2) / (N/2)
    tri = np.maximum(tri, 0.01)
    tri = tri / np.sum(tri)
    inits_coeffs.append(h_to_coeffs(tri))
    
    # W-shape
    w_shape = 1.0 + 2.0 * np.cos(4 * np.pi * i / N)
    w_shape = np.maximum(w_shape, 0.01)
    w_shape = w_shape / np.sum(w_shape)
    inits_coeffs.append(h_to_coeffs(w_shape))
    
    # Bimodal
    for alpha in [0.2, 0.3]:
        pos1 = alpha * N
        pos2 = (1 - alpha) * N
        sigma = N/8
        g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
        g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
        bimodal = g1 + g2
        bimodal = bimodal / np.sum(bimodal)
        inits_coeffs.append(h_to_coeffs(bimodal))
    
    best_h = None
    best_val = np.inf
    
    # Evaluate all initializations
    for c_init in inits_coeffs:
        if time_left() < 10:
            break
        h = coeffs_to_h(c_init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Optimize using gradient descent in coefficient space
    current_coeffs = h_to_coeffs(best_h)
    current_val = best_val
    
    # Phase 1: Coarse gradient descent
    eta = 0.05
    for k in range(200):
        if time_left() < 20:
            break
        
        h_curr = coeffs_to_h(current_coeffs)
        curr_val = compute_autoconv_max(h_curr)
        
        if curr_val < best_val:
            best_val = curr_val
            best_h = h_curr.copy()
        
        grad_c, smooth_val = compute_grad_coeff(current_coeffs, beta=50.0)
        
        grad_norm = np.linalg.norm(grad_c)
        if grad_norm < 1e-10:
            break
        
        grad_c = grad_c / grad_norm
        new_coeffs = current_coeffs - eta * grad_c
        
        # Check if new heights are reasonable
        new_h = coeffs_to_h(new_coeffs)
        if np.sum(new_h) > 0:
            new_val = compute_autoconv_max(new_h)
            if new_val < curr_val:
                current_coeffs = new_coeffs
                current_val = new_val
            else:
                eta *= 0.8
                if eta < 1e-6:
                    break
        else:
            eta *= 0.5
    
    # Phase 2: Fine-tuning with lower beta for sharper max approximation
    if time_left() > 30:
        eta = 0.01
        for k in range(200):
            if time_left() < 10:
                break
            
            h_curr = coeffs_to_h(current_coeffs)
            curr_val = compute_autoconv_max(h_curr)
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()
            
            grad_c, smooth_val = compute_grad_coeff(current_coeffs, beta=100.0)
            
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm < 1e-10:
                break
            
            grad_c = grad_c / grad_norm
            new_coeffs = current_coeffs - eta * grad_c
            
            new_h = coeffs_to_h(new_coeffs)
            if np.sum(new_h) > 0:
                new_val = compute_autoconv_max(new_h)
                if new_val < curr_val:
                    current_coeffs = new_coeffs
                    current_val = new_val
                else:
                    eta *= 0.8
                    if eta < 1e-7:
                        break
            else:
                eta *= 0.5
    
    # Phase 3: Direct gradient descent in height space for final refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.02
        
        for k in range(100):
            if time_left() < 5:
                break
            
            conv = compute_conv(current_h)
            c_val = np.max(conv)
            idx_max = np.argmax(conv)
            
            idx = np.arange(N)
            pos1 = idx_max - idx
            pos2 = idx - idx_max
            
            grad = np.zeros(N)
            mask1 = (pos1 >= 0) & (pos1 < N)
            if np.any(mask1):
                grad[mask1] += current_h[pos1[mask1]]
            mask2 = (pos2 >= 0) & (pos2 < N)
            if np.any(mask2):
                grad[mask2] += current_h[pos2[mask2]]
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            new_h = current_h - eta * grad
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s > 0:
                new_h /= s
            
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.8
                if eta < 1e-6:
                    break
    
    return best_h.tolist()
