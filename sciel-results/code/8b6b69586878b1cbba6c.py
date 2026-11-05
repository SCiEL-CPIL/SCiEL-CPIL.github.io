import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    N = 24
    # Use a high-resolution grid for accurate sup-norm estimation
    # M=2^16 (65536) is a good balance between accuracy and speed.
    M = 2 ** 16
    
    # Precompute DFT matrix E where E[k, n] = exp(2j * pi * k * n / M)
    # P(z_k) = sum c_n z_k^n = sum c_n exp(2j * pi * k * n / M)
    theta = 2 * np.pi * np.arange(M) / M
    n_idx = np.arange(N)
    E = np.exp(1j * np.outer(theta, n_idx))
    
    def get_supnorm(coeffs):
        """Compute sup norm of polynomial with given coefficients."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))

    def local_search(current_c):
        """
        Fast local search using 1-flip and 2-flip moves.
        """
        current_val = get_supnorm(current_c)
        best_c = current_c.copy()
        best_val = current_val
        
        # Precompute basis functions for each coefficient
        # basis[i] = exp(2j * pi * k * i / M)
        basis = E.T  # Shape (N, M)
        
        # Iterative improvement
        max_iters = 50000
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # --- 1-Flip Search ---
            P_curr = E @ current_c
            best_flip_val = current_val
            best_flip_idx = -1
            
            # We want to find i such that max |P_curr - 2*c_i*basis[i]| is minimized
            for i in range(N):
                delta = 2.0 * current_c[i] * basis[i]
                P_new = P_curr - delta
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                current_c[best_flip_idx] *= -1
                current_val = best_flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # --- 2-Flip Search ---
            # If no 1-flip improves, try 2-flips
            best_2flip_val = current_val
            best_2flip = None
            
            deltas = [2.0 * current_c[i] * basis[i] for i in range(N)]
            
            for i in range(N - 1):
                for j in range(i + 1, N):
                    P_new = P_curr - deltas[i] - deltas[j]
                    val = np.max(np.abs(P_new))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # No improvement, local minimum
            break
        
        return best_c, best_val

    # --- Initialization Strategy: Spectral Synthesis + Golay ---
    initial_candidates = []
    
    # 1. Golay-Rudin-Shapiro (known good starting point)
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial_candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral Synthesis:
    # We want a time-domain sequence that is +/-1.
    # This corresponds to a frequency-domain sequence with specific properties.
    # However, we can use the duality: 
    # If we pick a "flat" or "smooth" frequency spectrum, the time domain is concentrated.
    # If we pick a "random phase" flat magnitude spectrum, the time domain is noise-like.
    # The goal is to minimize max |P(z)|. 
    # P(z) is the DFT of the coefficients.
    # So we want the DFT of the coefficients to be flat.
    # This is a "flat spectrum" problem in the frequency domain of the coefficients.
    
    # Let's try generating candidates by:
    # a) Start with a target spectrum (e.g., constant magnitude with random phases).
    # b) IFFT to get time domain.
    # c) Round to +/-1.
    # d) Check if the resulting DFT is flatter.
    
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        
        # Random phase, constant magnitude spectrum
        phases = np.random.uniform(0, 2 * np.pi, size=N)
        spectrum = np.exp(1j * phases)
        
        # IFFT to get time domain coefficients
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        
        initial_candidates.append(c_bin)
        
        # Also try cosine-tapered magnitude to see if it helps
        taper = np.cos(np.pi * np.arange(N) / (N - 1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        initial_candidates.append(c_bin2)

    # 3. Random starts
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        initial_candidates.append(c_rand)

    # Run local search on all candidates
    best_overall_c = None
    best_overall_val = float('inf')
    
    for c0 in initial_candidates:
        if time.time() > deadline - 10:
            break
        c_best, v_best = local_search(c0)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best

    # If we have time, do a few more random restarts
    while time.time() < deadline - 5:
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        c_best, v_best = local_search(c_rand)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best

    if best_overall_c is None:
        # Fallback
        return [1] * N
        
    return [int(x) for x in best_overall_c]
