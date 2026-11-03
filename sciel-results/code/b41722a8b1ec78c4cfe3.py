import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize u
    # Use a random initialization with zero mean
    u = np.random.uniform(-1, 1, N)
    u -= np.mean(u)
    u = np.clip(u, -1, 1)
    
    best_score = float('inf')
    best_u = u.copy()
    
    # Precompute FFT of mask? No, mask changes.
    
    max_iters = 20000
    
    for step in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute autocorrelations
        U = np.fft.fft(u)
        C = np.fft.ifft(np.abs(U)**2).real
        
        # We care about k=1..N-1
        C_lags = C[1:]
        c_min = np.min(C_lags)
        
        # Calculate current max overlap
        # Overlap_k = N/4 - C_k/4
        # Max Overlap = N/4 - min(C_k)/4
        current_max_overlap = N/4.0 - c_min/4.0
        
        if current_max_overlap < best_score:
            best_score = current_max_overlap
            best_u = u.copy()
            
        # Determine active mask
        # Use a soft threshold to include near-minimum lags
        # This helps with convergence when multiple lags are tied
        spread = np.max(C_lags) - c_min
        if spread < 1e-6:
             spread = 1.0
             
        # Start with wide window, narrow it
        # Alpha controls the fraction of lags included
        # Anneal alpha from 0.5 to 0.05
        alpha = 0.5 * (1.0 - step / max_iters) + 0.05
        threshold = c_min + alpha * spread
        active_mask_lags = C_lags < threshold
        
        # Ensure at least one is active
        if not np.any(active_mask_lags):
            active_mask_lags = np.zeros(N-1, dtype=bool)
            active_mask_lags[np.argmin(C_lags)] = True
            
        # Construct full mask of length N
        m = np.zeros(N)
        m[1:] = active_mask_lags
        # Symmetry: C[k] == C[N-k]. If k is active, N-k should be active.
        # The mask m is used in correlation.
        # If we use correlation with m, and m is not symmetric, we get one-sided gradient.
        # But C_k is symmetric. The minimum is likely symmetric.
        # Let's enforce symmetry in m
        m_sym = np.zeros(N)
        for k in range(1, N):
            if active_mask_lags[k-1]:
                m_sym[k] = 1.0
                m_sym[N-k] = 1.0
        # m_sym[0] remains 0
        
        # Compute Gradient
        # G_j = sum_{k} m_sym[k] (u_{j+k} + u_{j-k})
        # Since m_sym is symmetric, u_{j+k} term and u_{j-k} term are identical sums.
        # Sum = 2 * sum_{k} m_sym[k] u_{j+k}
        # This is 2 * (u * m_sym) circular convolution?
        # Let's check: (u * m)_j = sum_i u_i m_{j-i}.
        # We want sum_k m_k u_{j+k}. Let i = j+k => k = i-j.
        # sum_i u_i m_{i-j}. This is correlation.
        # corr(u, m)_j = sum_i u_i m_{i-j}.
        # Using FFT: IFFT(FFT(u) * conj(FFT(m)))
        
        M_fft = np.fft.fft(m_sym)
        G = np.fft.ifft(U * np.conj(M_fft)).real
        G *= 2.0
        
        # Normalize
        g_norm = np.linalg.norm(G)
        if g_norm > 1e-9:
            G = G / g_norm
        else:
            G = np.zeros(N)
            
        # Learning rate
        # Anneal LR
        lr = 0.5 * (1.0 - step / max_iters) + 0.01
        
        # Update
        u += lr * G
        
        # Project to sum 0
        u -= np.mean(u)
        
        # Clip
        u = np.clip(u, -1, 1)
        
        # Re-project
        u -= np.mean(u)
        u = np.clip(u, -1, 1)
        
    # Final Polish
    for step in range(5000):
        if time.time() > deadline:
            break
        U = np.fft.fft(u)
        C = np.fft.ifft(np.abs(U)**2).real
        C_lags = C[1:]
        c_min = np.min(C_lags)
        
        current_max_overlap = N/4.0 - c_min/4.0
        if current_max_overlap < best_score:
            best_score = current_max_overlap
            best_u = u.copy()
            
        # Strict minimum
        idx_min = np.argmin(C_lags)
        k_min = idx_min + 1
        
        m_sym = np.zeros(N)
        m_sym[k_min] = 1.0
        m_sym[N-k_min] = 1.0
        
        M_fft = np.fft.fft(m_sym)
        G = np.fft.ifft(U * np.conj(M_fft)).real
        G *= 2.0
        
        g_norm = np.linalg.norm(G)
        if g_norm > 1e-9:
            G = G / g_norm
            
        lr = 0.05
        u += lr * G
        u -= np.mean(u)
        u = np.clip(u, -1, 1)
        u -= np.mean(u)
        u = np.clip(u, -1, 1)
        
    # Convert to h
    h = (best_u + 1) / 2.0
    
    # Enforce sum constraint
    target_sum = N / 2.0
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    if abs(diff) > 1e-12:
        h += diff / N
        h = np.clip(h, 0, 1)
        # Final adjustment
        diff2 = target_sum - np.sum(h)
        if diff2 != 0:
            h[0] += diff2
            h[0] = np.clip(h[0], 0, 1)
            
    return h
