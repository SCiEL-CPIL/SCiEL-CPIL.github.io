"""Signed step function for third autocorrelation inequality - CA-inspired search with off-peak optimization."""
import time
import numpy as np
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START

def _get_offpeak_autocorr(signs):
    """
    Compute the off-peak linear autocorrelation of the sign sequence.
    Returns the array of autocorrelation values for lags 1 to N-1.
    """
    n = len(signs)
    # Use a size that is a power of 2 for FFT efficiency if possible, 
    # or just 2*n. 2*400=800. Next power of 2 is 1024.
    size = 1024 
    f = np.zeros(size)
    f[:n] = signs
    F = rfft(f)
    # Power spectrum
    power = F * np.conj(F)
    corr = irfft(power, size)
    # The linear autocorrelation is in corr[0:n]
    # corr[0] is the sum of squares (N)
    # corr[1:n] are the off-peak autocorrelations
    return corr[1:n]

def _objective(signs):
    """
    Minimize the maximum absolute value of the off-peak autocorrelation.
    """
    corr = _get_offpeak_autocorr(signs)
    return np.max(np.abs(corr))

def _local_search(signs, time_limit=5.0):
    """
    Greedy local search with block flips to minimize off-peak autocorrelation.
    """
    start = time.time()
    n = len(signs)
    current = signs.copy()
    current_obj = _objective(current)
    best = current.copy()
    best_obj = current_obj
    
    # Pre-compute the FFT size and window for speed? 
    # _objective is fast enough for N=400.
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Single bit flips
        for i in range(n):
            current[i] = -current[i]
            new_obj = _objective(current)
            if new_obj < current_obj:
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
                improved = True
            else:
                current[i] = -current[i]
                
        # Occasional block flips to escape local minima
        if (time.time() - start) < time_limit:
             # Try a few random block flips
             for _ in range(5):
                 if (time.time() - start) > time_limit: break
                 bs = np.random.choice([4, 8, 16])
                 if bs >= n: continue
                 start_idx = np.random.randint(0, n - bs)
                 old_block = current[start_idx:start_idx+bs].copy()
                 current[start_idx:start_idx+bs] = -old_block
                 new_obj = _objective(current)
                 if new_obj < current_obj:
                     current_obj = new_obj
                     if current_obj < best_obj:
                         best_obj = current_obj
                         best = current.copy()
                     improved = True
                 else:
                     current[start_idx:start_idx+bs] = old_block

    return best, best_obj

def _generate_ca_sequence(rule, initial_state, steps):
    """
    Generate a sequence by evolving a 1D Cellular Automaton.
    """
    n = len(initial_state)
    current = initial_state.copy()
    for _ in range(steps):
        # Periodic boundary conditions
        left = np.roll(current, 1)
        right = np.roll(current, -1)
        # Convert to 0/1 for rule application? 
        # Signs are -1/1. Let's map -1->0, 1->1 for standard CA rules.
        # But we can define a custom update rule directly on signs.
        # Standard Elementary CA: new[i] = rule applied to (left, current, right)
        # Let's stick to 0/1 internally for CA generation, then map to -1/1.
        # Map: 0 -> -1, 1 -> 1
        
        # Convert current to 0/1
        bits = (current + 1) // 2
        l_bits = np.roll(bits, 1)
        r_bits = np.roll(bits, -1)
        
        # Index into rule
        # Pattern is 3 bits: l, c, r -> index 4*l + 2*c + r
        indices = 4 * l_bits + 2 * bits + r_bits
        new_bits = (rule >> indices) & 1
        
        # Convert back to signs
        current = 2 * new_bits - 1
        
    return current

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound C3.
    Uses CA-generated initializations and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    np.random.seed(42)
    
    best_signs = None
    best_obj = np.inf
    
    # List of CA rules to try
    # 30 (chaotic), 110 (complex), 90 (Sierpinski), 150, 184 (traffic), 225
    ca_rules = [30, 90, 110, 150, 184, 225, 15, 255, 0, 51, 102, 204]
    
    # Number of CA seeds to try
    num_ca_seeds = 50
    
    # Time budget per local search
    # Total time 115s. 
    # If we do 50 CA seeds, each gets ~2s for local search.
    # Plus generation time.
    
    for i in range(num_ca_seeds):
        if time.time() > deadline - 5.0:
            break
            
        rule = ca_rules[i % len(ca_rules)]
        
        # Generate random initial state for CA
        init_state = np.random.choice([-1, 1], size=N)
        
        # Choose number of steps
        # Vary steps to get different structures
        steps = (i % 20) + 1
        
        # Generate sequence
        ca_seq = _generate_ca_sequence(rule, init_state, steps)
        
        # Refine with local search
        # Allocate time based on remaining budget
        time_left = deadline - time.time() - 5.0
        if time_left <= 0: break
        
        # Give each CA seed a proportional share of the remaining time
        # But cap it to ensure we finish
        alloc_time = min(2.0, time_left / (num_ca_seeds - i))
        
        refined_signs, obj = _local_search(ca_seq, time_limit=alloc_time)
        
        if obj < best_obj:
            best_obj = obj
            best_signs = refined_signs.copy()
            
    # If we still have time, do a final polish on the best found
    if best_signs is not None and time.time() < deadline - 5.0:
        final_time = deadline - time.time() - 2.0
        if final_time > 0:
            final_signs, final_obj = _local_search(best_signs, time_limit=final_time)
            if final_obj < best_obj:
                best_signs = final_signs
                best_obj = final_obj

    # Fallback: if something went wrong, return random
    if best_signs is None:
        best_signs = np.random.choice([-1, 1], size=N)
        
    return best_signs

# EVOLVE-BLOCK-END
