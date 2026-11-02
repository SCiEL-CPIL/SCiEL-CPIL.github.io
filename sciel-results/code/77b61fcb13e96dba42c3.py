import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 110.0

def _compute_C(h):
    """
    Compute C_k = sum_i h[i] h[(i+k)%N] for binary h.
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return autocorr

def _max_overlap_from_C(C):
    """
    O_k = N/2 - C_k.
    Max overlap is N/2 - Min(C).
    """
    return HALF - np.min(C)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_val = float('inf')
    best_h = None
    
    # Try multiple random restarts
    while time.time() - t0 < DEADLINE:
        # Initialize random binary sequence with N/2 ones
        h = np.zeros(N)
        indices = rng.permutation(N)
        h[indices[:HALF]] = 1.0
        
        C = _compute_C(h)
        cur_val = _max_overlap_from_C(C)
        
        if cur_val < best_val:
            best_val = cur_val
            best_h = h.copy()
            
        # Local search
        improved = True
        while improved and (time.time() - t0 < DEADLINE):
            improved = False
            C = _compute_C(h)
            # We want to maximize min(C).
            # Find indices where C is minimal.
            min_C = np.min(C)
            # If all C are equal, we might be at a local optimum or a symmetric point.
            # Find the k with minimum C.
            k_min = np.argmin(C)
            
            # Search for a swap (a, b) with h[a]=1, h[b]=0
            # that increases C[k_min].
            # Delta C[k_min] = h[b+k_min] + h[b-k_min] - h[a+k_min] - h[a-k_min]
            # We want Delta > 0.
            
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            
            # To speed up, we can precompute or just iterate.
            # N=100, so 50*50 = 2500 checks per step. Very fast.
            found_swap = False
            best_delta = 0
            best_a = -1
            best_b = -1
            
            # We can prioritize swaps that increase C[k_min] the most.
            # Max possible delta is 2.
            
            for a in ones:
                # h[a] is 1
                # Terms subtracted: h[a+k_min] + h[a-k_min]
                sub_val = h[(a + k_min) % N] + h[(a - k_min) % N]
                for b in zeros:
                    # h[b] is 0
                    # Terms added: h[b+k_min] + h[b-k_min]
                    add_val = h[(b + k_min) % N] + h[(b - k_min) % N]
                    delta = add_val - sub_val
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
                        if best_delta == 2:
                            break
                if best_delta == 2:
                    break
                    
            if best_delta > 0:
                # Perform swap
                h[best_a] = 0
                h[best_b] = 1
                improved = True
                
                # Check if this improved the global min C
                C = _compute_C(h)
                new_val = _max_overlap_from_C(C)
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
            else:
                # No swap improves C[k_min].
                # Maybe we should try to improve the second minimum?
                # Or accept that we are stuck in a local minimum for this restart.
                # For now, break the local search loop for this restart.
                pass
                
    # Ensure best_h is valid
    if best_h is None:
        # Fallback to constant
        best_h = np.full(N, 0.5)
        
    return best_h
