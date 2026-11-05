"""
Combined strategy: 
1. Use Differential Evolution to search parameterized families (DFT, Chirp) for initial low-value candidates.
2. Refine the best candidates using local optimization.
3. Return the best finite, valid h found.
"""
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        # The task defines value = abs(2 * N * np.max(conv) / s^2)
        # Note: np.max(conv) is the maximum signed entry.
        return abs(2.0 * N * np.max(conv) / (s * s))

    def dft_objective(params):
        """
        h[i] = dc + sum_{k=1}^{K} r_k * cos(2*pi*k*i/N + p_k)
        params: [dc, r1, p1, r2, p2, ..., rK, pK]
        """
        K = 5
        dc = params[0]
        h = np.full(N, dc)
        i = np.arange(N)
        for k in range(1, K+1):
            r = params[1 + 2*(k-1)]
            p = params[2 + 2*(k-1)]
            h += r * np.cos(2*np.pi*k*i/N + p)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def chirp_objective(params):
        """
        h[i] = A + B * cos(omega*i + beta*i^2 + phase)
        params: [A, B, omega, beta, phase]
        """
        A, B, omega, beta, phase = params
        i = np.arange(N, dtype=float)
        h = A + B * np.cos(omega * i + beta * i**2 + phase)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    best_h = np.ones(N)
    best_val = compute_value(best_h) # Seed value is 2.0

    # Phase 1: Global Search with Differential Evolution
    # 1a: DFT-based search
    if time.time() < deadline - 60.0:
        try:
            K = 5
            bounds = [(-2.0, 2.0)] # DC
            for k in range(1, K+1):
                bounds.append((-1.5, 1.5)) # Amplitude
                bounds.append((0.0, 2.0 * np.pi)) # Phase
            
            de_result = differential_evolution(
                dft_objective,
                bounds,
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                dc = de_result.x[0]
                h_dft = np.full(N, dc)
                i = np.arange(N)
                for k in range(1, K+1):
                    r = de_result.x[1 + 2*(k-1)]
                    p = de_result.x[2 + 2*(k-1)]
                    h_dft += r * np.cos(2*np.pi*k*i/N + p)
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # 1b: Chirp-based search
    if time.time() < deadline - 60.0:
        try:
            bounds = [
                (-2.0, 2.0),      # A
                (-2.0, 2.0),      # B
                (0.0, 2.0*np.pi/N),  # omega
                (-0.01, 0.01),    # beta
                (0.0, 2.0*np.pi)  # phase
            ]
            de_result = differential_evolution(
                chirp_objective,
                bounds,
                maxiter=50,
                popsize=20,
                seed=123,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                A, B, omega, beta, phase = de_result.x
                i = np.arange(N, dtype=float)
                h_chirp = A + B * np.cos(omega * i + beta * i**2 + phase)
                val_chirp = compute_value(h_chirp)
                if val_chirp < best_val and np.all(np.isfinite(h_chirp)) and abs(np.sum(h_chirp)) > 1e-6:
                    best_h = h_chirp.copy()
                    best_val = val_chirp
                    print(f"Chirp DE: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Chirp DE error: {e}", file=sys.stderr)

    # Phase 2: Local Refinement
    # We try to refine the best_h found so far using Nelder-Mead on the raw vector h.
    # This is computationally expensive (400 dims), so we only do it if we have time.
    # A better approach for local refinement in high dimensions is often to refine the parameters
    # of the family that produced the best result. However, since we don't track which family 
    # won, we can try a few random restarts of local optimization on the raw h if time permits,
    # or simply trust the DE polish. 
    # Given the time constraint and high dimensionality of raw h, we will skip raw-h refinement
    # and instead ensure DE had enough iterations. The DE polish=True already does a local step.
    
    # If we still have time, we can try a second DE run with different seeds for DFT
    if time.time() < deadline - 30.0 and best_val > 1.0:
        try:
            K = 5
            bounds = [(-2.0, 2.0)]
            for k in range(1, K+1):
                bounds.append((-1.5, 1.5))
                bounds.append((0.0, 2.0 * np.pi))
            
            de_result2 = differential_evolution(
                dft_objective,
                bounds,
                maxiter=30,
                popsize=15,
                seed=999,
                tol=1e-6,
                polish=True
            )
            if de_result2.fun < best_val:
                dc = de_result2.x[0]
                h_dft2 = np.full(N, dc)
                i = np.arange(N)
                for k in range(1, K+1):
                    r = de_result2.x[1 + 2*(k-1)]
                    p = de_result2.x[2 + 2*(k-1)]
                    h_dft2 += r * np.cos(2*np.pi*k*i/N + p)
                val_dft2 = compute_value(h_dft2)
                if val_dft2 < best_val and np.all(np.isfinite(h_dft2)) and abs(np.sum(h_dft2)) > 1e-6:
                    best_h = h_dft2.copy()
                    best_val = val_dft2
                    print(f"DFT DE 2nd: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE 2nd error: {e}", file=sys.stderr)

    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-6:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
