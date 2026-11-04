import numpy as np
import time

N = 400
TIME_LIMIT = 110.0 # Safety margin

def construct():
    start_time = time.time()
    
    # Initialize with random signs
    # Using a fixed seed for reproducibility in debugging, but random is fine for search
    # Actually, using random is better to explore different basins.
    # We will run a few restarts if time permits.
    
    best_score = float('inf')
    best_seq = None
    
    # Pre-compute indices for autocorrelation updates?
    # A_k = sum_{i=0}^{N-1-k} s_i s_{i+k}
    
    # We will minimize S = sum_{k=1}^{N-1} A_k^2
    # But we want to return the sequence with min max|A_k|/ (N-1)
    
    def calc_metrics(s):
        # s is array of +1/-1
        # Compute autocorrelations A_k for k=1..N-1
        # A_k = np.dot(s[:-k], s[k:])
        # This is O(N^2) if done naively for all k.
        # For N=400, N^2=160k. This is fast.
        A = np.zeros(N)
        for k in range(1, N):
            A[k] = np.dot(s[:-k], s[k:])
        max_sidelobe = np.max(np.abs(A[1:]))
        sum_sq = np.sum(A[1:]**2)
        return max_sidelobe, sum_sq, A

    # Local Search
    # We use sum_sq as the objective for the search, but track max_sidelobe for the final score.
    
    for restart in range(10): # Max 10 restarts
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        s = np.random.choice([1, -1], N)
        
        # Initial metrics
        max_sl, sum_sq, A = calc_metrics(s)
        best_local_sum_sq = sum_sq
        best_local_max_sl = max_sl
        best_local_s = s.copy()
        
        # Greedy local search
        improved = True
        while improved:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            improved = False
            # Shuffle indices to avoid bias? Or just iterate 0..N-1.
            # Iterating 0..N-1 is deterministic and fast.
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                    
                # Calculate change in sum_sq if we flip s[i]
                # s_new = s.copy(); s_new[i] = -s[i]
                # Instead of full recalc, use delta.
                # A_k changes by -2*s[i]*(s[i+k] + s[i-k])
                # Let's compute the delta for all k.
                
                # s is 1D array
                # k ranges 1..N-1
                # term_k = s[i+k] if i+k < N else 0
                # term_km = s[i-k] if i-k >= 0 else 0
                # delta_A_k = -2 * s[i] * (term_k + term_km)
                
                # We can vectorize this.
                # k = np.arange(1, N)
                # idx_plus = i + k
                # valid_plus = idx_plus < N
                # idx_minus = i - k
                # valid_minus = idx_minus >= 0
                
                # s_plus = np.where(valid_plus, s[idx_plus[valid_plus]], 0) # Need to handle sizes
                # This is getting complex to vectorize perfectly with variable bounds.
                # For N=400, a simple loop for k is fast enough.
                
                si = s[i]
                delta_sum_sq = 0
                # We need to update A as well if we accept, but for now just check delta.
                # Actually, we need A to compute delta.
                # So we maintain A.
                
                # Let's re-evaluate the cost of maintaining A.
                # If we accept a flip, we update A in O(N).
                # If we reject, we don't.
                
                # Compute delta for all k
                # A is array of size N, A[0] unused or 0.
                # k from 1 to N-1
                
                # Optimization:
                # delta_A[k] = -2 * si * (s[i+k] if i+k<N else 0 + s[i-k] if i-k>=0 else 0)
                
                # Let's just do the loop.
                for k in range(1, N):
                    val_plus = 0.0
                    if i + k < N:
                        val_plus = s[i + k]
                    val_minus = 0.0
                    if i - k >= 0:
                        val_minus = s[i - k]
                    
                    delta_A_k = -2.0 * si * (val_plus + val_minus)
                    if delta_A_k != 0:
                        # new_A_k = A[k] + delta_A_k
                        # delta_sum_sq += 2 * A[k] * delta_A_k + delta_A_k**2
                        delta_sum_sq += 2.0 * A[k] * delta_A_k + delta_A_k**2
                        
                if delta_sum_sq < -1e-9: # Strict improvement
                    # Accept flip
                    s[i] = -si
                    # Update A
                    for k in range(1, N):
                        val_plus = 0.0
                        if i + k < N:
                            val_plus = s[i + k] # Note: s[i] changed, but s[i+k] is the neighbor
                            # Wait, the formula for A_k involves s[i] and s[i+k].
                            # If we flip s[i], the term s[i]s[i+k] becomes -s[i]s[i+k].
                            # So A_k decreases by 2 * old_s[i] * s[i+k].
                            # This matches delta_A_k calculated above.
                        val_minus = 0.0
                        if i - k >= 0:
                            val_minus = s[i - k]
                        
                        # We already calculated delta_A_k in the loop above, but we didn't store it.
                        # Let's just recalculate or store it.
                        # To avoid double loop, we should store delta_A.
                        # But for N=400, this is fine.
                        pass
                    
                    # Recalculate A properly to avoid errors
                    # Actually, updating A in place is safer.
                    # Let's just recalculate A from scratch if we accept?
                    # Cost O(N^2).
                    # If we accept often, this is expensive.
                    # Better to update A.
                    
                    # Let's refactor the inner loop to store delta_A.
                    # This is getting messy. Let's use a simpler approach:
                    # Recalculate A after each accepted flip.
                    # If we do ~1000 accepted flips, 1000 * 160k ops = 1.6e8 ops.
                    # This might be too slow in Python (10-20s).
                    # So we MUST update A incrementally.
                    
                    # Let's redo the flip acceptance logic with stored deltas.
                    # This restart loop is getting complicated.
                    # Let's break and write a cleaner version.
                    break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
        # After local search, evaluate best
        # The logic above was flawed (breaks out of loops).
        # Let's restart with a cleaner implementation.

    # Clean Implementation
    # We will use a function to compute A and metrics.
    # We will use a simple greedy search that recalculates A only when a flip is accepted.
    # To speed up, we'll use the delta method but carefully.
    
    best_global_score = float('inf')
    best_global_seq = None
    
    for restart in range(20):
        if time.time() - start_time > TIME_LIMIT - 2:
            break
            
        s = np.random.choice([1, -1], N)
        
        # Initialize A
        A = np.zeros(N)
        for k in range(1, N):
            A[k] = np.dot(s[:-k], s[k:])
            
        current_sum_sq = np.sum(A[1:]**2)
        
        # Local Search
        for iteration in range(1000): # Max 1000 passes
            if time.time() - start_time > TIME_LIMIT - 2:
                break
                
            improved = False
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                    
                si = s[i]
                
                # Calculate delta_A for all k
                # We need to check if flipping s[i] reduces sum_sq
                delta_sum_sq = 0.0
                deltas = np.zeros(N) # Store to update A if accepted
                
                for k in range(1, N):
                    vp = 0.0
                    if i + k < N:
                        vp = s[i + k]
                    vm = 0.0
                    if i - k >= 0:
                        vm = s[i - k]
                    
                    dAk = -2.0 * si * (vp + vm)
                    if dAk != 0.0:
                        deltas[k] = dAk
                        delta_sum_sq += 2.0 * A[k] * dAk + dAk**2
                        
                if delta_sum_sq < -1e-9:
                    # Accept
                    s[i] = -si
                    A += deltas
                    current_sum_sq += delta_sum_sq
                    improved = True
                    
            if not improved:
                break
                
        # Evaluate final sequence of this restart
        max_sl = np.max(np.abs(A[1:]))
        score = max_sl / (N - 1)
        
        if score < best_global_score:
            best_global_score = score
            best_global_seq = s.copy()
            
    if best_global_seq is None:
        return np.ones(N)
        
    return best_global_seq
