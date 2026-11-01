import numpy as np
import time

N = 400

def _compute_ac(h):
    """
    Compute aperiodic autocorrelation for lags 0 to N-1.
    h: np.array of length N with values +/- 1.
    Returns ac: np.array of length N.
    """
    # np.correlate(h, h, 'full')
    # Index N-1 is lag 0.
    # Index N-1+k is lag k.
    corr = np.correlate(h, h, mode='full')
    return corr[N-1 : 2*N-1]

def _local_search_steepest(h, time_limit=5.0, use_sa=False, T0=0.01):
    """
    Local search using incremental autocorrelation updates.
    Tries all single-bit flips and picks the best (steepest descent).
    If use_sa, allows worse moves to escape local minima.
    """
    start_time = time.time()
    h = h.astype(int).copy()
    ac = _compute_ac(h)
    current_val = np.max(np.abs(ac[1:]))
    best_val = current_val
    best_h = h.copy()
    
    T = T0 if use_sa else 0.0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        best_j = -1
        best_new_val = current_val
        best_new_ac_1 = None
        best_new_h = None
        
        # Try all possible flips
        for j in range(N):
            new_ac_1 = ac[1:].copy()
            
            # Update ac[1:] based on flip at j
            # Delta for lag k (index k-1 in ac[1:])
            # Term 1: h[j] h[j+k] for k < N-j
            # k ranges 1 to N-j-1
            # Indices 0 to N-j-2 in new_ac_1
            if j < N - 1:
                new_ac_1[0 : N-j-1] -= 2 * h[j] * h[j+1 : N]
                
            # Term 2: h[j-k] h[j] for k <= j
            # k ranges 1 to j
            # Indices 0 to j-1 in new_ac_1
            if j > 0:
                new_ac_1[0 : j] -= 2 * h[j] * h[j-1 : -1 : -1]
                
            new_val = np.max(np.abs(new_ac_1))
            
            # Acceptance criteria
            accept = False
            if new_val < best_new_val:
                accept = True
            elif use_sa and T > 1e-9:
                delta = new_val - current_val
                if delta < 0:
                    accept = True
                else:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        accept = True
            
            if accept:
                best_new_val = new_val
                best_j = j
                best_new_ac_1 = new_ac_1
                best_new_h = h.copy()
                best_new_h[j] = -h[j]
                
        if best_j == -1:
            # No move accepted
            if use_sa:
                T *= 0.95
                if T < 1e-9:
                    break
            else:
                break
        else:
            # Apply move
            h = best_new_h
            ac[1:] = best_new_ac_1
            current_val = best_new_val
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If SA, we might have accepted a worse move, so no_improve_count increases
                # But we track best_h separately.
                
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Initializations
    inits = []
    inits.append(np.ones(N))
    inits.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    inits.append(np.sign(np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))))
    
    # Add some random inits
    rng = np.random.default_rng(42)
    for _ in range(3):
        inits.append(rng.choice([-1, 1], N))
        
    for h0 in inits:
        if time.time() >= deadline - 10.0:
            break
            
        # Phase 1: Steepest Descent
        h_sd, val_sd = _local_search_steepest(h0, time_limit=5.0)
        
        if val_sd < best_val:
            best_val = val_sd
            best_h = h_sd.copy()
            
        # Phase 2: Simulated Annealing on the result of SD
        if time.time() < deadline - 5.0:
            h_sa, val_sa = _local_search_steepest(h_sd, time_limit=5.0, use_sa=True, T0=max(0.01, best_val*0.1))
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
                
    if best_h is None:
        return np.ones(N)
        
    return best_h
