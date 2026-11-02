import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    if len(A) > 1:
        return float(np.max(np.abs(A[1:])))
    return 0.0

def _update_acf(A, s, i):
    """
    Efficiently compute the new A array if s[i] is flipped.
    Returns the new A array.
    """
    # Delta calculation for flipping s[i]
    # A[k] = sum_{j=0}^{N-1-k} s[j] s[j+k]
    # Flipping s[i] changes terms where s[i] is involved.
    # s[i] is involved in A[k] if:
    # 1. j = i -> term s[i]s[i+k] exists if i+k < N
    # 2. j+k = i -> term s[i-k]s[i] exists if i-k >= 0
    
    # delta[k] = -2 * s[i] * ( (s[i+k] if i+k<N else 0) + (s[i-k] if i-k>=0 else 0) )
    
    delta = np.zeros(N, dtype=np.float64)
    old_si = s[i]
    
    # Part 1: k such that i+k < N => k < N-i
    # delta[k] += -2 * old_si * s[i+k]
    # s[i+k] for k=0..N-i-1 is s[i], s[i+1], ..., s[N-1]
    if N - i > 0:
        delta[:N-i] += -2 * old_si * s[i:]
        
    # Part 2: k such that i-k >= 0 => k <= i
    # delta[k] += -2 * old_si * s[i-k]
    # s[i-k] for k=0..i is s[i], s[i-1], ..., s[0]
    if i + 1 > 0:
        delta[:i+1] += -2 * old_si * s[i::-1]
        
    return A + delta

def _greedy_polish(s, A, deadline):
    """
    Perform strict greedy coordinate descent until no improvement is found
    or deadline is reached.
    """
    current_val = _max_abs_acf_from_array(A)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Random permutation to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() >= deadline:
                return s, A, current_val
            
            new_si = -s[i]
            A_new = _update_acf(A, s, i)
            new_val = _max_abs_acf_from_array(A_new)
            
            if new_val < current_val:
                s[i] = new_si
                A = A_new
                current_val = new_val
                improved = True
                
    return s, A, current_val

def _simulated_annealing(s, A, start_time, deadline, T_start=1.0, T_end=0.01, alpha=0.999):
    """
    Perform simulated annealing.
    """
    current_val = _max_abs_acf_from_array(A)
    best_val = current_val
    best_s = s.copy()
    best_A = A.copy()
    
    T = T_start
    # Estimate steps per second to adjust cooling if necessary? 
    # For now, use a fixed cooling schedule based on time or steps.
    # Let's use time-based cooling or step-based.
    # Given N=400, ops are fast. Let's do a fixed number of steps or time-based.
    
    # We will run until deadline.
    # To ensure we cover the space, we'll just loop.
    
    # Adaptive T reduction
    # If we are stuck, lower T faster.
    
    while time.time() < deadline:
        i = np.random.randint(N)
        A_new = _update_acf(A, s, i)
        new_val = _max_abs_acf_from_array(A_new)
        
        delta = new_val - current_val
        
        if delta < 0 or (T > 0 and np.random.random() < math.exp(-delta / T)):
            # Accept move
            s[i] = -s[i]
            A = A_new
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
                best_A = A.copy()
                
        # Cool down
        # Simple exponential decay
        T *= alpha
        
        # If T is too small, we are effectively doing greedy. 
        # We can break early if T is very small and we haven't improved in a while?
        # Or just let it run.
        
    return best_s, best_A, best_val

def construct():
    start_time = time.time()
    # Safety margin: finish 5 seconds before 120s limit
    deadline = start_time + 115.0
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # Try a few restarts
    # Each restart: Random init -> Anneal -> Polish
    # We don't know how many restarts fit, so we loop while time remains.
    
    while time.time() < deadline - 1.0: # Leave 1s for final polish of best
        # Initialize
        s = np.random.randint(0, 2, size=N).astype(np.int8) * 2 - 1
        A = _calc_acf(s)
        
        # Phase 1: Simulated Annealing
        # Allocate 50% of remaining time to annealing? 
        # Or just run annealing for a fixed duration, e.g., 2s, then polish.
        # Let's try: Anneal for min(remaining_time/2, 3.0) seconds.
        anneal_time = min(3.0, (deadline - time.time()) * 0.5)
        if anneal_time < 0.5:
            break
            
        s_annealed, A_annealed, val_annealed = _simulated_annealing(
            s, A, start_time, time.time() + anneal_time
        )
        
        # Phase 2: Greedy Polish
        # Use remaining time for this restart's polish
        s_polished, A_polished, val_polished = _greedy_polish(
            s_annealed, A_annealed, deadline
        )
        
        if val_polished < best_val:
            best_val = val_polished
            best_seq = s_polished.copy()
            
    # Final Polish on the global best to squeeze out more
    if time.time() < deadline:
        A_best = _calc_acf(best_seq)
        best_seq, A_best, best_val = _greedy_polish(best_seq, A_best, deadline)
        
    return best_seq
