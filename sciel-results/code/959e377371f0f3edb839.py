import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit = 138.0  # Safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs for a binary coloring (0/1)."""
        c = coloring[aps_arr]  # shape (total_aps, 4)
        return int(np.all(c == c[:, 0:1], axis=1).sum())
    
    def compute_deltas(coloring):
        """Compute delta for flipping each position. Vectorized."""
        c = coloring[aps_arr]  # (total_aps, 4)
        n0 = (c == 0).sum(axis=1)  # (total_aps,)
        n1 = 4 - n0
        
        # mono_mask: APs that are currently mono
        mono_mask = (n0 == 4) | (n1 == 4)
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Process mono APs: for each mono AP, subtract 1 from all 4 positions
        if mono_mask.any():
            mono_aps = aps_arr[mono_mask]  # (k, 4)
            mono_flat = mono_aps.flatten()
            deltas -= np.bincount(mono_flat, minlength=n).astype(np.int32)
        
        # Process 3-1 APs (n0==3, n1==1): the odd one out is the position with color 1
        mask_31 = (n0 == 3) & (n1 == 1)
        if mask_31.any():
            ap_indices_31 = np.where(mask_31)[0]
            for idx in ap_indices_31:
                ap = aps_arr[idx]
                colors = c[idx]
                odd_pos = ap[colors == 1][0]
                deltas[odd_pos] += 1
        
        # Process 1-3 APs (n0==1, n1==3): the odd one out is the position with color 0
        mask_13 = (n0 == 1) & (n1 == 3)
        if mask_13.any():
            ap_indices_13 = np.where(mask_13)[0]
            for idx in ap_indices_13:
                ap = aps_arr[idx]
                colors = c[idx]
                odd_pos = ap[colors == 0][0]
                deltas[odd_pos] += 1
        
        return deltas
    
    def greedy_improve(coloring, max_sweeps=200, time_limit=None):
        """Greedy local search: repeatedly flip the position with the most negative delta."""
        current_score = count_mono(coloring)
        
        for sweep in range(max_sweeps):
            if time_limit is not None and time.time() > time_limit:
                break
            
            deltas = compute_deltas(coloring)
            min_delta = deltas.min()
            
            if min_delta >= 0:
                break  # No improvement possible
            
            # Flip all positions with delta == min_delta (most negative)
            flip_mask = (deltas == min_delta)
            flip_positions = np.where(flip_mask)[0]
            
            coloring[flip_positions] = 1 - coloring[flip_positions]
            current_score += min_delta * len(flip_positions)
        
        return coloring, current_score
    
    def spectral_objective(signal):
        """
        Compute the spectral objective: sum over all 4-APs of (s_i1*s_i2*s_i3*s_i4)^2
        where signal is a ±1 vector. This equals the number of monochromatic 4-APs
        when signal[i] = 1 if color[i]=1, -1 if color[i]=0.
        
        Actually, for ±1 signals, s_i1*s_i2*s_i3*s_i4 = 1 iff all four are same sign.
        So sum of (product)^2 = total_aps always. We need sum of products instead.
        
        For ±1: product = 1 means mono (all same), product = -1 means not mono.
        Mono count = (total_aps + sum_of_products) / 2
        """
        # For each AP (i1,i2,i3,i4), compute product s[i1]*s[i2]*s[i3]*s[i4]
        # Sum these products
        prods = signal[aps_arr[:,0]] * signal[aps_arr[:,1]] * signal[aps_arr[:,2]] * signal[aps_arr[:,3]]
        return prods.sum()
    
    def spectral_gradient(signal):
        """
        Compute gradient of the spectral objective with respect to each signal element.
        grad[i] = sum over APs containing i of (product of other 3 elements)
        """
        s = signal
        grad = np.zeros(n, dtype=np.float64)
        
        # For each position, sum contributions from all APs containing it
        # AP (i1,i2,i3,i4): contribution to grad[i1] = s[i2]*s[i3]*s[i4]
        # etc.
        
        # Vectorized approach:
        # For each AP, compute the product of all 4, then divide by each element
        # But since elements are ±1, division = multiplication
        prods = s[aps_arr[:,0]] * s[aps_arr[:,1]] * s[aps_arr[:,2]] * s[aps_arr[:,3]]
        
        # grad[i1] += prods / s[i1] = prods * s[i1] (since s[i1] is ±1)
        # Actually: if AP product = p, and we flip s[i1], new product = -p
        # So delta = -p - p = -2p for flipping... wait, let me think more carefully.
        
        # Objective = sum over APs of prod(s)
        # If we flip s[i], for each AP containing i, the product changes sign.
        # So dObj/ds[i] (in the sense of "what's the change if we flip s[i]")
        # = -2 * sum over APs containing i of (product of that AP)
        
        # But for gradient descent on the continuous relaxation, we use:
        # grad[i] = sum over APs containing i of (product of the other 3 elements in that AP)
        
        # Let's compute it properly:
        # For AP (a,b,c,d): 
        #   grad[a] += s[b]*s[c]*s[d]
        #   grad[b] += s[a]*s[c]*s[d]
        #   grad[c] += s[a]*s[b]*s[d]
        #   grad[d] += s[a]*s[b]*s[c]
        
        sb = s[aps_arr[:,1]]
        sc = s[aps_arr[:,2]]
        sd = s[aps_arr[:,3]]
        sa = s[aps_arr[:,0]]
        
        contrib_a = sb * sc * sd
        contrib_b = sa * sc * sd
        contrib_c = sa * sb * sd
        contrib_d = sa * sb * sc
        
        np.add.at(grad, aps_arr[:,0], contrib_a)
        np.add.at(grad, aps_arr[:,1], contrib_b)
        np.add.at(grad, aps_arr[:,2], contrib_c)
        np.add.at(grad, aps_arr[:,3], contrib_d)
        
        return grad
    
    def spectral_optimize(signal, time_limit, max_iters=5000, lr=0.5):
        """
        Optimize the ±1 signal using projected gradient descent on the continuous
        relaxation, then round back to ±1.
        """
        s = signal.astype(np.float64).copy()
        best_signal = signal.copy()
        best_obj = spectral_objective(signal)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            grad = spectral_gradient(s)
            
            # Gradient step
            s -= lr * grad
            
            # Project to hypercube [-1, 1]
            s = np.clip(s, -1.0, 1.0)
            
            # Check if converged (all elements are close to ±1)
            if np.all(np.abs(np.abs(s) - 1.0) < 0.01):
                break
            
            # Evaluate the rounded solution
            rounded = np.sign(s)
            if np.any(rounded == 0):
                rounded[rounded == 0] = 1
            
            obj = spectral_objective(rounded)
            if obj < best_obj:
                best_obj = obj
                best_signal = rounded.copy()
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.95
        
        return best_signal, best_obj
    
    def multi_start_search(time_limit):
        """
        Multi-strategy search:
        1. FFT-based spectral optimization from random starts
        2. Greedy local search from spectral results
        3. Final greedy polish
        """
        best_coloring = None
        best_score = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Spectral optimization from multiple random starts
        n_restarts = 8
        time_per_restart = (time_limit - 30.0) / n_restarts
        
        for r in range(n_restarts):
            if time.time() > time_limit - 30.0:
                break
            
            # Random ±1 signal
            signal = rng.choice([-1.0, 1.0], size=n)
            
            # Spectral optimization
            opt_signal, opt_obj = spectral_optimize(signal, time_limit - 28.0, max_iters=3000, lr=0.8)
            
            # Convert to binary coloring
            coloring = (opt_signal > 0).astype(np.int32)
            
            # Greedy improvement
            coloring, score = greedy_improve(coloring, max_sweeps=100, time_limit=time_limit - 25.0)
            
            if score < best_score:
                best_score = score
                best_coloring = coloring.copy()
            
            # Also try the opposite coloring
            coloring_inv = 1 - coloring
            coloring_inv, score_inv = greedy_improve(coloring_inv, max_sweeps=100, time_limit=time_limit - 25.0)
            
            if score_inv < best_score:
                best_score = score_inv
                best_coloring = coloring_inv.copy()
        
        # Phase 2: More spectral restarts with different parameters if time allows
        if best_coloring is not None and time.time() < time_limit - 15.0:
            for r in range(4):
                if time.time() > time_limit - 15.0:
                    break
                
                signal = rng.choice([-1.0, 1.0], size=n)
                opt_signal, opt_obj = spectral_optimize(signal, time_limit - 12.0, max_iters=2000, lr=0.5)
                coloring = (opt_signal > 0).astype(np.int32)
                coloring, score = greedy_improve(coloring, max_sweeps=100, time_limit=time_limit - 10.0)
                
                if score < best_score:
                    best_score = score
                    best_coloring = coloring.copy()
        
        # Phase 3: Final greedy polish from best solution
        if best_coloring is not None and time.time() < time_limit - 2.0:
            final_coloring, final_score = greedy_improve(best_coloring.copy(), max_sweeps=500, time_limit=time_limit - 1.0)
            if final_score < best_score:
                best_score = final_score
                best_coloring = final_coloring.copy()
        
        if best_coloring is None:
            return [int(x) for x in np.random.randint(0, 2, n)]
        return [int(x) for x in best_coloring]
    
    result = multi_start_search(limit)
    return result
# EVOLVE-BLOCK-END
