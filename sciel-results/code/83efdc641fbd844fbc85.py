import numpy as np
import time

N = 100

def _compute_value(h):
    A = np.correlate(h, h, mode='full')
    min_A = np.min(A)
    max_C = 50 - min_A
    return max_C / N * 2

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_h = None
    best_val = float('inf')
    
    # Initial candidates
    candidates = []
    
    # Random
    for _ in range(20):
        h = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1
        candidates.append(h)
        
    # Structured
    h = np.zeros(N)
    for i in range(10):
        start = i * 10
        h[start:start+5] = 1
    candidates.append(h)
    
    h = np.zeros(N)
    h[::2] = 1
    candidates.append(h)
    
    h = np.zeros(N)
    h[::4] = 1
    h[1::4] = 1
    candidates.append(h)
    
    for h in candidates:
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    current_h = best_h.copy()
    current_A = np.correlate(current_h, current_h, mode='full')
    current_min_A = np.min(current_A)
    current_val = _compute_value(current_h)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        ones = np.where(current_h == 1)[0]
        zeros = np.where(current_h == 0)[0]
        
        best_swap = None
        best_new_min_A = current_min_A
        
        # Shuffle to avoid bias
        ones_list = list(ones)
        zeros_list = list(zeros)
        np.random.shuffle(ones_list)
        np.random.shuffle(zeros_list)
        
        # Check swaps
        # To save time, we can limit the number of swaps checked per iteration
        # or break early if we find a good one.
        # But finding the BEST swap requires checking all.
        # Let's check all.
        
        for i in zeros_list:
            # Pre-compute shift for i?
            # h[i+k] for k in [-99, 99]
            # np.correlate output index m corresponds to lag k = m - (N-1)
            # We need to align Delta A with current_A.
            
            # Let's use a helper to get the shifted vector aligned with A
            # A has length 2N-1 = 199.
            # Lag k is at index N-1+k.
            
            # Vector Vi: Vi[m] = h[i + (m - (N-1))]
            # Vector Vj: Vj[m] = h[j + (m - (N-1))]
            
            # This is equivalent to:
            # Vi = np.zeros(2*N-1)
            # for m in range(2*N-1):
            #    idx = i + m - (N-1)
            #    if 0 <= idx < N: Vi[m] = h[idx]
            
            # We can do this with slicing.
            
            # Let's just do the slicing in the loop.
            
            # Shift i
            # The vector of h shifted by i relative to the correlation center?
            # Actually, simpler:
            # Delta A[k] = h[i+k] - h[j+k]
            # We can construct the full Delta A vector.
            
            # Let's create an array of lags
            # lags = np.arange(-(N-1), N)
            # Delta_A = h[i+lags] - h[j+lags] (with 0 padding)
            
            # This is efficient with numpy.
            
            lags = np.arange(-(N-1), N)
            
            # h[i+lags]
            idx_i = i + lags
            valid_i = (idx_i >= 0) & (idx_i < N)
            Vi = np.zeros(2*N-1)
            Vi[valid_i] = current_h[idx_i[valid_i]]
            
            idx_j = j + lags
            valid_j = (idx_j >= 0) & (idx_j < N)
            Vj = np.zeros(2*N-1)
            Vj[valid_j] = current_h[idx_j[valid_j]]
            
            Delta_A = Vi - Vj
            A_new = current_A + Delta_A
            min_A_new = np.min(A_new)
            
            if min_A_new > best_new_min_A:
                best_new_min_A = min_A_new
                best_swap = (i, j)
                
        if best_swap is not None:
            i, j = best_swap
            current_h[i] = 1
            current_h[j] = 0
            current_A = np.correlate(current_h, current_h, mode='full') # Recalc to be safe
            current_min_A = np.min(current_A)
            current_val = _compute_value(current_h)
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 20:
                # Perturb
                for _ in range(2):
                    i = np.random.choice(zeros)
                    j = np.random.choice(ones)
                    current_h[i], current_h[j] = current_h[j], current_h[i]
                current_A = np.correlate(current_h, current_h, mode='full')
                current_min_A = np.min(current_A)
                current_val = _compute_value(current_h)
                no_improve_count = 0

    return current_h
