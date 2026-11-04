"""
Program to minimize the peak autoconvolution of a signed step function.
Uses Rule 90 cellular automaton (Sierpinski gasket) structure for initial
construction, then applies targeted local search for refinement.
"""

import numpy as np
import time

N = 400

def _autoconv_peak(f):
    """
    Compute the maximum absolute value of the autoconvolution (f * f)
    for non-zero lags.
    """
    n = len(f)
    if n == 0:
        return 0.0
    
    # Use FFT for efficient convolution
    fft_len = 1
    while fft_len < 2 * n - 1:
        fft_len *= 2
        
    F = np.fft.rfft(f, n=fft_len)
    conv = np.fft.irfft(F * F, n=fft_len)
    
    center = n - 1
    left = conv[:center]
    right = conv[center+1:]
    
    max_val = 0.0
    if len(left) > 0:
        max_val = max(max_val, np.max(np.abs(left)))
    if len(right) > 0:
        max_val = max(max_val, np.max(np.abs(right)))
        
    return float(max_val)

def _rule90_sequence(n, seed=1):
    """
    Generate a signed sequence from Rule 90 cellular automaton.
    Rule 90: new_cell = left XOR right
    Starting from a single seed, this creates a Sierpinski gasket pattern.
    We run for enough steps to fill n positions, then map to signs.
    """
    # Determine number of rows needed to get at least n cells
    # After k steps, we have 2k+1 cells centered
    # We need to collect n cells from the automaton evolution
    
    # Start with a single 1 in the center of a wide array
    width = 2 * n + 100
    state = np.zeros(width, dtype=np.uint8)
    center_idx = width // 2
    state[center_idx] = 1
    
    # Collect cells as we evolve
    collected = []
    max_steps = n + 10
    
    for step in range(max_steps):
        # Collect cells from current state (skip leading/trailing zeros)
        nonzero_indices = np.where(state > 0)[0]
        if len(nonzero_indices) > 0:
            # Take cells from the non-zero region
            for idx in nonzero_indices:
                if len(collected) >= n:
                    break
                collected.append(1 if state[idx] == 1 else -1)
        
        if len(collected) >= n:
            break
            
        # Apply Rule 90: new[i] = state[i-1] XOR state[i+1]
        new_state = np.zeros(width, dtype=np.uint8)
        new_state[1:-1] = np.bitwise_xor(state[:-2], state[2:])
        state = new_state
    
    # If we didn't collect enough, pad with alternating signs
    while len(collected) < n:
        collected.append(1 if len(collected) % 2 == 0 else -1)
    
    return np.array(collected[:n], dtype=np.float64)

def _thue_morse_signs(n):
    """
    Generate a Thue-Morse sequence mapped to signs.
    """
    signs = np.zeros(n)
    for i in range(n):
        count = bin(i).count('1')
        signs[i] = 1.0 if count % 2 == 0 else -1.0
    return signs

def _local_search(f, deadline, max_iter_per_call=50000):
    """
    Perform local search (Hill Climbing) to minimize the peak autoconvolution.
    Uses a more aggressive approach with multiple restarts within the function.
    """
    current_f = f.copy()
    current_score = _autoconv_peak(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    # Simple hill climbing with occasional random restarts
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Try a random flip
        pos = rng.randint(0, N)
        old_val = current_f[pos]
        current_f[pos] = -old_val
        
        new_score = _autoconv_peak(current_f)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
        else:
            # Revert if not improving
            current_f[pos] = old_val
            no_improve_count += 1
            
            # If stuck for too long, do a random restart from best
            if no_improve_count > max_no_improve:
                current_f = best_f.copy()
                current_score = best_score
                no_improve_count = 0
                # Add some random perturbations
                for _ in range(5):
                    pos2 = rng.randint(0, N)
                    current_f[pos2] = -current_f[pos2]
                current_score = _autoconv_peak(current_f)
        
    return best_f, best_score

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound.
    """
    start_time = time.time()
    # Safety margin: 150s limit, use 130s for search
    deadline = start_time + 130.0
    
    # Generate candidate initializations
    candidates = []
    
    # 1. Rule 90 cellular automaton (primary structural choice)
    candidates.append(_rule90_sequence(N, seed=1))
    
    # 2. Rule 90 with different seed
    candidates.append(_rule90_sequence(N, seed=3))
    
    # 3. Thue-Morse (known good baseline)
    candidates.append(_thue_morse_signs(N))
    
    # 4. Random initialization
    rng = np.random.RandomState(123)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    best_f = None
    best_score = float('inf')
    
    # Allocate time: spend most time on the best candidates
    # First pass: quick evaluation of all candidates
    for i, f_init in enumerate(candidates):
        if time.time() > deadline:
            break
            
        score = _autoconv_peak(f_init)
        if score < best_score:
            best_score = score
            best_f = f_init.copy()
    
    # Second pass: run local search on the best initializations
    # Sort candidates by their initial score
    scored_candidates = []
    for f_init in candidates:
        score = _autoconv_peak(f_init)
        scored_candidates.append((score, f_init))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on top candidates
    for i, (init_score, f_init) in enumerate(scored_candidates):
        if time.time() > deadline:
            break
            
        # Allocate more time to better initializations
        if i == 0:
            sub_deadline = deadline
        elif i == 1:
            sub_deadline = deadline - 10.0
        else:
            sub_deadline = deadline - 20.0
        
        if sub_deadline <= time.time():
            break
            
        f_opt, score = _local_search(f_init, sub_deadline)
        
        if score < best_score:
            best_score = score
            best_f = f_opt
    
    if best_f is None:
        # Fallback
        best_f = candidates[0]
        
    return best_f

# EVOLVE-BLOCK-END
