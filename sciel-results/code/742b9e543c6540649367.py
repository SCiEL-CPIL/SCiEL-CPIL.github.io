import numpy as np
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses linear (non-circular) correlation: C(k) = sum_i h[i]*(1-h[i+k])
    Optimized via log-sum-exp smooth max + simulated annealing + coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    N = 100
    target_sum = N / 2.0
    
    def get_linear_correlations(h):
        """
        Returns array of length 2*(N-1) with:
          indices 0..N-2: C_neg(k) = sum_i h[i]*(1-h[i-k]) for k=1..N-1 (negative shifts)
          indices N-1..2N-3: C_pos(k) = sum_i h[i]*(1-h[i+k]) for k=1..N-1 (positive shifts)
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is shift 0
        # c[N-1-k] for k=1..N-1 gives negative shifts
        # c[N-1+k] for k=1..N-1 gives positive shifts
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def max_overlap(h):
        c = get_linear_correlations(h)
        return np.max(c)
    
    def smooth_max(c, temp=0.03):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def project(h):
        """Project to 0<=h<=1, sum(h)=N/2 using bisection for exact feasibility."""
        h = np.clip(h, 0, 1)
        
        # If sum is already correct, return
        if abs(np.sum(h) - target_sum) < 1e-12:
            return h
        
        # Bisection on offset to find exact feasible point
        # We need to find offset such that sum(clip(h + offset, 0, 1)) = target_sum
        # The function f(offset) = sum(clip(h + offset, 0, 1)) is monotone in offset
        
        def f(offset):
            return np.sum(np.clip(h + offset, 0, 1)) - target_sum
        
        # Find bracket
        lo, hi = -1.0, 1.0
        flo, fhi = f(lo), f(hi)
        
        if flo * fhi > 0:
            # No bracket, fall back to iterative
            for _ in range(50):
                diff = target_sum - np.sum(h)
                if abs(diff) < 1e-14:
                    break
                mask = (h > 1e-10) & (h < 1.0 - 1e-10)
                if not np.any(mask):
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
            return h
        
        # Bisection
        for _ in range(60):
            mid = (lo + hi) / 2
            fm = f(mid)
            if abs(fm) < 1e-14:
                return np.clip(h + mid, 0, 1)
            if flo * fm <= 0:
                hi = mid
                fhi = fm
            else:
                lo = mid
                flo = fm
        
        return np.clip(h + (lo + hi) / 2, 0, 1)
    
    # Initialize with multiple random starts
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = np.inf
    
    def find_clusters(h, threshold=0.08):
        clusters = []
        i = 0
        while i < N:
            j = i
            base_val = h[i]
            while j < N and abs(h[j] - base_val) < threshold:
                j += 1
            if j > i:
                clusters.append((i, j))
            i = j
        return clusters
    
    def simulated_annealing(h_init, n_iter, T_init, T_final, rng):
        h_curr = h_init.copy()
        local_best_h = h_curr.copy()
        local_best_score = max_overlap(local_best_h)
        
        for it in range(n_iter):
            if time.time() > deadline - 20:
                break
            
            progress = it / n_iter
            T = T_init * (T_final / T_init) ** progress
            
            move_type = rng.random()
            if move_type < 0.30:
                # Cluster shift
                clusters = find_clusters(h_curr, threshold=0.06)
                if clusters:
                    idx = rng.integers(0, len(clusters))
                    i, j = clusters[idx]
                    h_prop = h_curr.copy()
                    shift = rng.uniform(-0.2, 0.2)
                    h_prop[i:j] += shift
                    h_prop = project(h_prop)
                else:
                    h_prop = h_curr.copy()
                    i = rng.integers(0, N)
                    h_prop[i] += rng.uniform(-0.15, 0.15)
                    h_prop = project(h_prop)
            elif move_type < 0.55:
                # Antiferromagnetic flip on a block
                i = rng.integers(0, N - 3)
                length = rng.integers(2, 20)
                j = min(i + length, N)
                h_prop = h_curr.copy()
                h_prop[i:j] = 1.0 - h_curr[i:j]
                h_prop = project(h_prop)
            elif move_type < 0.80:
                # Pair swap
                i, j = rng.integers(0, N, size=2)
                h_prop = h_curr.copy()
                delta = rng.uniform(-0.12, 0.12)
                h_prop[i] += delta
                h_prop[j] -= delta
                h_prop = project(h_prop)
            elif move_type < 0.95:
                # Random restart of a segment
                i = rng.integers(0, N - 5)
                length = rng.integers(3, 25)
                j = min(i + length, N)
                h_prop = h_curr.copy()
                h_prop[i:j] = rng.uniform(0.1, 0.9, size=j-i)
                h_prop = project(h_prop)
            else:
                # Large perturbation
                i = rng.integers(0, N - 10)
                length = rng.integers(10, 30)
                j = min(i + length, N)
                h_prop = h_curr.copy()
                h_prop[i:j] = rng.uniform(0.0, 1.0, size=j-i)
                h_prop = project(h_prop)
            
            score_curr = max_overlap(h_curr)
            score_prop = max_overlap(h_prop)
            
            delta = score_prop - score_curr
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                h_curr = h_prop
            
            curr_score = max_overlap(h_curr)
            if curr_score < local_best_score:
                local_best_score = curr_score
                local_best_h = h_curr.copy()
        
        return local_best_h, local_best_score
    
    # Multiple restarts with different initializations
    n_restarts = 8
    iters_per_restart = 500
    
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Different initialization strategies
        if r == 0:
            h_init = np.full(N, 0.5)
        elif r == 1:
            h_init = np.tile([0.3, 0.7], N // 2)
            if len(h_init) < N:
                h_init = np.append(h_init, 0.5)
            h_init = project(h_init)
        elif r == 2:
            h_init = np.linspace(0.2, 0.8, N)
            h_init = project(h_init)
        elif r == 3:
            h_init = np.sin(np.linspace(0, 2*np.pi, N)) * 0.3 + 0.5
            h_init = project(h_init)
        elif r == 4:
            h_init = np.cos(np.linspace(0, 4*np.pi, N)) * 0.4 + 0.5
            h_init = project(h_init)
        elif r == 5:
            h_init = rng.uniform(0.1, 0.9, N)
            h_init = project(h_init)
        elif r == 6:
            # Block pattern
            h_init = np.zeros(N)
            h_init[::2] = 0.7
            h_init[1::2] = 0.3
            h_init = project(h_init)
        else:
            h_init = rng.uniform(0.0, 1.0, N)
            h_init = project(h_init)
        
        T_init = 0.2 - 0.02 * r  # Decreasing initial temperature
        T_final = 0.0003
        
        h_local, score_local = simulated_annealing(h_init, iters_per_restart, T_init, T_final, rng)
        
        if score_local < best_score:
            best_score = score_local
            best_h = h_local.copy()
    
    # Stage 2: Coordinate descent polish
    if time.time() < deadline - 10 and best_h is not None:
        h = best_h.copy()
        current_score = max_overlap(h)
        
        # Coordinate descent with line search
        for epoch in range(5):
            if time.time() > deadline - 5:
                break
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Try to optimize h[i] while keeping others fixed
                # The overlap depends on h[i] through terms involving shifts that include index i
                # For each shift k, h[i] appears in:
                #   C_pos(k): h[i] * (1 - h[i+k]) if i+k < N
                #   C_neg(k): h[i] * (1 - h[i-k]) if i-k >= 0
                #   Also as the second factor: (1-h[i]) * h[i+k] and (1-h[i]) * h[i-k]
                
                # Find optimal h[i] by checking all shifts where i is involved
                # C(k) = sum_j h[j]*(1-h[j+k])
                # dC/dh[i] involves all k where i or i±k is in range
                
                # Simpler: just do a 1D search over h[i]
                old_val = h[i]
                best_local = old_val
                best_local_score = current_score
                
                # Try a range of values
                for trial in np.linspace(max(0, old_val - 0.3), min(1, old_val + 0.3), 11):
                    h_trial = h.copy()
                    h_trial[i] = trial
                    h_trial = project(h_trial)
                    s = max_overlap(h_trial)
                    if s < best_local_score:
                        best_local_score = s
                        best_local = trial
                
                if best_local != old_val:
                    h[i] = best_local
                    h = project(h)
                    current_score = max_overlap(h)
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
    
    # Final: ensure validity
    best_h = project(best_h)
    
    # Verify sum constraint with high precision
    diff = target_sum - best_h.sum()
    if abs(diff) > 1e-10:
        # Use bisection for exact correction
        def f(offset):
            return np.sum(np.clip(best_h + offset, 0, 1)) - target_sum
        
        lo, hi = -0.1, 0.1
        for _ in range(50):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        
        best_h = np.clip(best_h + (lo + hi) / 2, 0, 1)
    
    # Final clip to ensure bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
