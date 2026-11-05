"""Optimized step-height constructor for the third autocorrelation inequality task.

Strategy: Use a sinusoidal chirp (linear frequency sweep) with a DC offset,
parameterized by (frequency, chirp_rate, dc_bias). The rapid oscillation causes
destructive interference in autocorrelation sidelobes, potentially reducing the
maximum signed autocorrelation entry below the seed's value.

We perform a multi-stage optimization:
1. Coarse grid search over the three parameters.
2. Fine-grained local refinement around the best grid point.
3. A final polish pass with tighter tolerances.

Time budget: explicit deadline guard to finish well within 150s.
"""

import numpy as np
import time

N = 400
DEADLINE = 145.0  # seconds, safety margin under 150s


def _value(h):
    """Compute the objective: 2*N*max(conv(h,h))/sum(h)^2."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _make_h(freq, chirp, dc):
    """Construct h from parameters.
    
    h[i] = dc + A * sin(2*pi*freq*i/N + 0.5*chirp*(i/N)^2)
    where A is chosen so that sum(h) is nonzero (we fix A=1 and rely on dc).
    """
    t = np.arange(N, dtype=np.float64) / N
    phase = 2.0 * np.pi * freq * t + 0.5 * chirp * (t * t)
    return dc + np.sin(phase)


def construct():
    t0 = time.time()
    
    # Stage 1: Coarse grid search
    # Frequency: number of cycles over [-1/4, 1/4]; try a range
    # Chirp: rate of frequency change
    # DC: bias to ensure sum != 0 and potentially reduce max autocorr
    
    best_val = float('inf')
    best_params = (1.0, 0.0, 1.0)
    
    # Coarse grid
    freqs = np.arange(0.5, 40.5, 1.0)       # 40 values
    chirps = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])  # 7 values
    dcs = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])          # 7 values
    
    # To keep runtime manageable, we subsample: try all freqs with a subset of chirps/dcs
    # Then refine.
    
    # First pass: all freqs, chirps=[0, 10, 50], dcs=[0.3, 1.0, 2.0]
    chirps_coarse = [0.0, 10.0, 50.0]
    dcs_coarse = [0.3, 1.0, 2.0]
    
    for freq in freqs:
        if time.time() - t0 > DEADLINE - 30:
            break
        for chirp in chirps_coarse:
            for dc in dcs_coarse:
                h = _make_h(freq, chirp, dc)
                s = np.sum(h)
                if abs(s) < 1e-10:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_params = (freq, chirp, dc)
    
    # Stage 2: Fine local search around best_params
    freq0, chirp0, dc0 = best_params
    if time.time() - t0 < DEADLINE - 20:
        # Local refinement: 5x5x5 grid around best
        freq_step = max(0.25, freq0 * 0.05)
        chirp_step = max(1.0, chirp0 * 0.2) if chirp0 > 0 else 5.0
        dc_step = max(0.1, dc0 * 0.1)
        
        freqs_fine = np.arange(freq0 - 2*freq_step, freq0 + 2*freq_step + freq_step, freq_step)
        chirps_fine = np.arange(max(0, chirp0 - 2*chirp_step), chirp0 + 2*chirp_step + chirp_step, chirp_step)
        dcs_fine = np.arange(max(0.01, dc0 - 2*dc_step), dc0 + 2*dc_step + dc_step, dc_step)
        
        for freq in freqs_fine:
            if time.time() - t0 > DEADLINE - 15:
                break
            for chirp in chirps_fine:
                if chirp < 0:
                    continue
                for dc in dcs_fine:
                    if dc < 0.01:
                        continue
                    h = _make_h(freq, chirp, dc)
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        continue
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_params = (freq, chirp, dc)
    
    # Stage 3: Even finer polish
    freq0, chirp0, dc0 = best_params
    if time.time() - t0 < DEADLINE - 10:
        freq_step = max(0.05, freq0 * 0.01)
        chirp_step = max(0.5, chirp0 * 0.05) if chirp0 > 0 else 1.0
        dc_step = max(0.02, dc0 * 0.02)
        
        freqs_polish = np.arange(freq0 - freq_step, freq0 + freq_step + freq_step, freq_step/2)
        chirps_polish = np.arange(max(0, chirp0 - chirp_step), chirp0 + chirp_step + chirp_step, chirp_step/2)
        dcs_polish = np.arange(max(0.001, dc0 - dc_step), dc0 + dc_step + dc_step, dc_step/2)
        
        for freq in freqs_polish:
            if time.time() - t0 > DEADLINE - 5:
                break
            for chirp in chirps_polish:
                if chirp < 0:
                    continue
                for dc in dcs_polish:
                    if dc < 0.001:
                        continue
                    h = _make_h(freq, chirp, dc)
                    s = np.sum(h)
                    if abs(s) < 1e-10:
                        continue
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_params = (freq, chirp, dc)
    
    # Final: also check the seed in case all sinusoidal attempts are worse
    seed_val = _value(np.ones(N))
    if seed_val < best_val:
        return np.ones(N)
    
    freq0, chirp0, dc0 = best_params
    h_final = _make_h(freq0, chirp0, dc0)
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(h_final)) or abs(np.sum(h_final)) < 1e-10:
        return np.ones(N)
    
    return h_final
