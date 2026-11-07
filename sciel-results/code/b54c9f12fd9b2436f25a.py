"""Program for Erdos minimum overlap task using low-dimensional parametric projection."""
import numpy as np
from itertools import product

N = 100


def _evaluate_h(h):
    """Compute the objective value for a given height sequence."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return float('inf')
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return float('inf')
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_and_clip(h):
    """Project h onto sum=N/2 and clip to [0,1], then re-project."""
    h = h.copy()
    # First clip
    h = np.clip(h, 0.0, 1.0)
    # Project onto sum = N/2
    h = h + (N / 2.0 - np.sum(h)) / N
    # Clip again
    h = np.clip(h, 0.0, 1.0)
    # Final projection
    h = h + (N / 2.0 - np.sum(h)) / N
    return h


def construct():
    """
    Use a sum of three sinusoids with variable frequencies, amplitudes, and phases.
    Grid search over 9 parameters (3 freqs, 3 amps, 3 phases) to minimize
    max(np.correlate(h, 1-h, mode="full")) / N * 2.
    """
    i = np.arange(N)
    
    # Define search ranges
    # Frequencies: use rational multiples that create interesting interference patterns
    # We want frequencies that are not too close to avoid constructive interference
    freq_options = [1.0 / 8, 1.0 / 12, 1.0 / 16, 1.0 / 20, 1.0 / 24, 1.0 / 30, 1.0 / 40, 1.0 / 50]
    amp_options = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    phase_options = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
    
    best_value = float('inf')
    best_h = np.full(N, 0.5)
    
    # We'll do a coarse-to-fine search
    # First: try combinations of 3 distinct frequencies from a smaller set
    # to keep computation manageable
    
    # Precompute sine waves for each frequency
    sin_waves = {}
    for f in freq_options:
        for p in phase_options:
            key = (f, p)
            sin_waves[key] = np.sin(2 * np.pi * f * i + p)
    
    # Search over triples of (freq, phase) pairs with amplitudes
    # To keep it fast, we'll limit the search space
    # Use a subset of frequencies and phases for the coarse search
    
    # Coarse search: try a subset
    coarse_freqs = [1.0 / 10, 1.0 / 15, 1.0 / 20, 1.0 / 25, 1.0 / 35]
    coarse_phases = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    coarse_amps = [0.15, 0.25, 0.35]
    
    for f1, f2, f3 in product(coarse_freqs, repeat=3):
        for p1, p2, p3 in product(coarse_phases, repeat=3):
            for a1, a2, a3 in product(coarse_amps, repeat=3):
                h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i + p1) \
                         + a2 * np.sin(2 * np.pi * f2 * i + p2) \
                         + a3 * np.sin(2 * np.pi * f3 * i + p3)
                h = _project_and_clip(h_raw)
                val = _evaluate_h(h)
                if val < best_value:
                    best_value = val
                    best_h = h.copy()
    
    # Fine-tune around the best solution
    if best_value < float('inf'):
        # Extract approximate parameters from best_h by fitting
        # For now, do a local refinement by perturbing amplitudes and phases
        # We'll try small perturbations
        
        # Get the best configuration's parameters
        # We need to track which parameters gave the best result
        # Let's redo the coarse search tracking the best params
        best_value = float('inf')
        best_h = np.full(N, 0.5)
        best_params = None
        
        for f1, f2, f3 in product(coarse_freqs, repeat=3):
            for p1, p2, p3 in product(coarse_phases, repeat=3):
                for a1, a2, a3 in product(coarse_amps, repeat=3):
                    h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i + p1) \
                             + a2 * np.sin(2 * np.pi * f2 * i + p2) \
                             + a3 * np.sin(2 * np.pi * f3 * i + p3)
                    h = _project_and_clip(h_raw)
                    val = _evaluate_h(h)
                    if val < best_value:
                        best_value = val
                        best_h = h.copy()
                        best_params = (f1, f2, f3, p1, p2, p3, a1, a2, a3)
        
        # Local refinement: perturb each parameter slightly
        if best_params is not None:
            f1, f2, f3, p1, p2, p3, a1, a2, a3 = best_params
            
            # Try small perturbations
            freq_pert = [0.0, 0.005, -0.005, 0.01, -0.01]
            phase_pert = [0.0, 0.1, -0.1, 0.2, -0.2]
            amp_pert = [0.0, 0.01, -0.01, 0.02, -0.02]
            
            for df1 in freq_pert:
                for df2 in freq_pert:
                    for df3 in freq_pert:
                        for dp1 in phase_pert:
                            for dp2 in phase_pert:
                                for dp3 in phase_pert:
                                    for da1 in amp_pert:
                                        for da2 in amp_pert:
                                            for da3 in amp_pert:
                                                h_raw = 0.5 + (a1 + da1) * np.sin(2 * np.pi * (f1 + df1) * i + p1 + dp1) \
                                                         + (a2 + da2) * np.sin(2 * np.pi * (f2 + df2) * i + p2 + dp2) \
                                                         + (a3 + da3) * np.sin(2 * np.pi * (f3 + df3) * i + p3 + dp3)
                                                h = _project_and_clip(h_raw)
                                                val = _evaluate_h(h)
                                                if val < best_value:
                                                    best_value = val
                                                    best_h = h.copy()
    
    # Final verification
    assert np.all(np.isfinite(best_h)), "Heights must be finite"
    assert np.all(best_h >= -1e-9) and np.all(best_h <= 1 + 1e-9), "Heights out of bounds"
    assert abs(np.sum(best_h) - N / 2) <= 1e-9 * N, "Sum constraint violated"
    
    return best_h
