import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Initialize with random signs
    # Using a fixed seed might be reproducible, but random is fine for search
    # We want to beat the all-ones score.
    h = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial autocorrelation
    # R(tau) for tau = 0 to N-1
    # We only care about tau > 0 for PSL, but we need R(0) for completeness?
    # The score is "upper bound on C3".
    # Usually C3 involves the max of |R(tau)| for tau != 0.
    # Let's assume value = max(|R(tau)| for tau in 1..N-1).
    
    def compute_autocorr(h):
        # Using FFT for speed if N is large, but N=400 is small.
        # np.convolve is O(N^2).
        # Let's use np.correlate or manual loop for clarity and ease of update.
        # Actually, for initial calculation, FFT is fast.
        # But for updates, direct calculation is better.
        # Let's use a direct method for the initial state to ensure consistency with updates.
        R = np.zeros(N)
        for tau in range(N):
            # R(tau) = sum_{j=0}^{N-1-tau} h[j] h[j+tau]
            if tau == 0:
                R[tau] = np.sum(h*h)
            else:
                R[tau] = np.sum(h[:N-tau] * h[tau:])
        return R

    R = compute_autocorr(h)
    current_psl = np.max(np.abs(R[1:]))
    best_psl = current_psl
    best_h = h.copy()
    
    # Simulated Annealing / Local Search
    # Temperature schedule
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Calculate change in R if we flip h[idx]
        # We need to update R(tau) for all tau
        # Delta R(tau) = -2 * h[idx] * (h[idx+tau] if valid + h[idx-tau] if valid)
        
        # Create a copy of R to test
        # Instead of copying, we can apply, check, and revert if needed.
        
        old_h_idx = h[idx]
        
        # Prepare the update vector
        # We need to update R[tau] for tau = 1 to N-1
        # And R[0] stays N (since sum h^2 = N)
        
        # Let's just compute the new R values for the affected lags?
        # No, the flip affects ALL lags tau where the index is involved.
        # h[idx] is involved in R(tau) if:
        # 1. idx = j, so tau = j - (j-tau) ... 
        # R(tau) = sum h[j]h[j+tau].
        # h[idx] appears as h[j] if j=idx, so term is h[idx]h[idx+tau]. Valid if idx+tau < N.
        # h[idx] appears as h[j+tau] if j+tau=idx => j=idx-tau. Term is h[idx-tau]h[idx]. Valid if idx-tau >= 0.
        
        # So for a specific tau:
        # change = 0
        # if idx + tau < N: change -= 2 * h[idx] * h[idx+tau]
        # if idx - tau >= 0: change -= 2 * h[idx-tau] * h[idx]
        
        # This is O(1) per tau. Total O(N) to compute all changes.
        
        # Let's apply the flip to h
        h[idx] = -h[idx]
        
        # Update R
        # We need to update R[1:N]
        # We can vectorize the update?
        # Let's just loop tau.
        
        # To optimize, we can precompute the "neighbor" contributions?
        # No, h changes.
        
        # Loop over tau
        # We need to know the OLD h[idx] to calculate the change.
        # Let old_val = old_h_idx
        
        # For each tau in 1..N-1:
        # delta = 0
        # if idx+tau < N: delta -= 2 * old_val * h[idx+tau]  (Note: h[idx+tau] is current, which is unchanged by the flip of h[idx] unless idx+tau==idx, which implies tau=0)
        # if idx-tau >= 0: delta -= 2 * h[idx-tau] * old_val
        
        # So delta = -2 * old_val * (h[idx+tau] (if valid) + h[idx-tau] (if valid))
        
        # We can compute this for all tau at once?
        # Let's try.
        
        # h_plus = h[idx+1:N]  # h[idx+tau] for tau=1..N-1-idx
        # h_minus = h[:idx]    # h[idx-tau] for tau=1..idx
        
        # This is getting complex to vectorize perfectly without padding.
        # Given N=400, a Python loop over 400 items is ~0.1ms.
        # 400 flips per sweep -> 40ms per sweep.
        # 120s / 0.04s = 3000 sweeps.
        # This is plenty.
        
        # Let's do the loop.
        
        # We need to store the old R to revert if we reject.
        # Or just calculate new_psl, and if we reject, revert h and R.
        
        # To avoid copying R (size 400), we can just re-compute R from h if we reject?
        # Re-computing R is O(N^2) = 160,000 ops. ~10-20ms.
        # If we reject often, this is expensive.
        # Better to store old_R = R.copy() (size 400, fast) and revert.
        
        old_R = R.copy()
        
        # Apply changes to R
        for tau in range(1, N):
            delta = 0.0
            if idx + tau < N:
                delta -= 2.0 * old_h_idx * h[idx + tau]
            if idx - tau >= 0:
                delta -= 2.0 * h[idx - tau] * old_h_idx
            R[tau] += delta
            
        # R[0] is always N
        # R[tau] for tau > 0 are updated.
        
        new_psl = np.max(np.abs(R[1:]))
        
        # Acceptance criterion
        if new_psl < current_psl:
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_h = h.copy()
        else:
            # Simulated Annealing acceptance
            delta_e = new_psl - current_psl
            if delta_e > 0:
                prob = np.exp(-delta_e / T)
                if np.random.rand() < prob:
                    current_psl = new_psl
                else:
                    # Revert
                    h[idx] = old_h_idx
                    R = old_R
            else:
                # Should not happen if new_psl >= current_psl
                current_psl = new_psl
                if new_psl < best_psl:
                    best_psl = new_psl
                    best_h = h.copy()
        
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_h
