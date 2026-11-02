import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT for initialization."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _update_acf_delta(A, s, i):
    """
    Update autocorrelation A in-place after flipping s[i].
    Returns the change in cost (new_cost - old_cost) is not directly returned, 
    but A is updated. We need to recompute cost from A.
    
    Delta A_k = -2 * s[i] * (s[(i+k)%N] + s[(i-k)%N])
    """
    si = s[i]
    # We need to update A for all k from 1 to N-1
    # A_k_new = A_k_old - 2 * si * (s[i+k] + s[i-k])
    
    # Vectorized update
    # k = 0..N-1
    k = np.arange(N)
    # s[i+k]
    idx_plus = (i + k) % N
    # s[i-k]
    idx_minus = (i - k) % N
    
    # Note: For k=0, the term is -2*si*(s[i]+s[i]) = -4*si*s[i] = -4*si^2 = -4.
    # But we only care about k >= 1 for the cost.
    # However, A[0] is the energy, which is constant N. 
    # Wait, A[0] = sum s_j^2 = N. It doesn't change.
    # Let's verify the formula for k=0:
    # A_0 = sum s_j s_j = N.
    # Flip s_i: new A_0 = N - 2*s_i^2 + 2*s_i^2 = N. Correct.
    # The formula A_k = A_k - 2 s_i (s_{i+k} + s_{i-k}) works for all k.
    
    delta = -2.0 * si * (s[idx_plus] + s[idx_minus])
    A += delta
    # A[0] should remain N, but floating point errors might creep in.
    # We can reset A[0] = N if needed, but it's not used in cost.

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Heuristic initialization: Try to find a good starting point
    # Random initialization is fine, but we can try a few random starts and keep the best.
    
    # Simulated Annealing parameters
    # We will run multiple SA runs or one long SA with restarts.
    # Given N=400, O(N) update is very fast.
    
    for run_id in range(20):  # Max 20 restarts if time permits
        if time.time() > deadline:
            break
            
        # Initialize sequence
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # Simulated Annealing
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling = 0.9999
        max_iter = 100000  # Max iterations per restart
        
        # We track the current cost. 
        # To avoid O(N) max calculation every step, we can maintain a histogram or 
        # just accept the O(N) cost. N=400 is small. 100k iterations * 400 ops = 40M ops.
        # In Python, this might be slow if not vectorized well.
        # np.max on 400 elements is fast.
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Pick random index
            i = np.random.randint(N)
            
            # Flip
            old_si = s[i]
            s[i] = -old_si
            
            # Update A
            _update_acf_delta(A, s, i)
            
            # Calculate new cost
            new_val = _max_abs_acf_from_array(A)
            
            # Acceptance criterion
            delta_cost = new_val - current_val
            if delta_cost < 0:
                # Improvement
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_seq = s.copy()
            else:
                # Deterioration
                if T > 0:
                    prob = np.exp(-delta_cost / T)
                    if np.random.rand() < prob:
                        current_val = new_val
                    else:
                        # Reject: Revert
                        s[i] = old_si
                        _update_acf_delta(A, s, i) # Revert A
                        # Note: _update_acf_delta uses s[i] which is now old_si again.
                        # Wait, the delta formula is linear. 
                        # If we flip back, we apply the same delta again?
                        # A_new = A_old - 2 s_i (s_{i+k} + s_{i-k})
                        # If we flip s_i to -s_i, the new s_i is -old_si.
                        # Let's trace:
                        # Start: s_i = 1. A = A0.
                        # Flip to -1. Delta = -2(1)(...). A1 = A0 + Delta.
                        # Reject. Flip back to 1.
                        # Current s_i = -1.
                        # Delta_revert = -2(-1)(...) = 2(...).
                        # A2 = A1 + Delta_revert = A0 + Delta - Delta = A0.
                        # So yes, calling _update_acf_delta with the *new* s[i] (which is -old_si) 
                        # will calculate the correct delta to revert.
                        # Wait, my code above does:
                        # s[i] = -old_si (flip)
                        # _update_acf_delta(A, s, i) -> uses s[i] = -old_si.
                        # If reject:
                        # s[i] = old_si (revert)
                        # _update_acf_delta(A, s, i) -> uses s[i] = old_si.
                        # This is correct.
                        
                # Cool down
                T *= cooling
                
        # Optional: Final polish with greedy local search
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                # Try all flips, keep the best one
                best_i = -1
                best_v = current_val
                for i in range(N):
                    old_si = s[i]
                    s[i] = -old_si
                    _update_acf_delta(A, s, i)
                    v = _max_abs_acf_from_array(A)
                    if v < best_v:
                        best_v = v
                        best_i = i
                    # Revert for next trial
                    s[i] = old_si
                    _update_acf_delta(A, s, i)
                
                if best_i != -1:
                    # Apply the best flip
                    old_si = s[best_i]
                    s[best_i] = -old_si
                    _update_acf_delta(A, s, best_i)
                    current_val = best_v
                    if current_val < best_val:
                        best_val = current_val
                        best_seq = s.copy()
                    improved = True
                else:
                    break

    return best_seq
