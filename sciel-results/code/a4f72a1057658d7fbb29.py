import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Initialize with a balanced random sequence
    # 200 ones, 200 minus ones
    x = np.random.choice([1, -1], size=N)
    # Ensure exact balance if possible (N is even)
    if np.sum(x) != 0:
        # Swap signs to balance
        pos_indices = np.where(x == 1)[0]
        neg_indices = np.where(x == -1)[0]
        if len(pos_indices) > N // 2:
            # Flip some 1s to -1
            excess = len(pos_indices) - N // 2
            to_flip = np.random.choice(pos_indices, size=excess, replace=False)
            x[to_flip] = -1
        elif len(neg_indices) > N // 2:
            excess = len(neg_indices) - N // 2
            to_flip = np.random.choice(neg_indices, size=excess, replace=False)
            x[to_flip] = 1

    # Calculate initial autocorrelation (linear)
    # c[k] = sum(x[i] * x[i+k]) for k=0..N-1
    def calc_autocorr(x):
        # Use FFT for initial calculation, then incremental updates
        # Zero pad to 2*N for linear convolution
        f_x = np.fft.fft(x, 2*N)
        acf = np.fft.ifft(f_x * np.conj(f_x))[:N]
        return np.real(acf)

    corr = calc_autocorr(x)
    # We care about max absolute value for k > 0
    # c[0] is N, which is constant for +/-1.
    # We minimize max(c[1:])
    current_max = np.max(np.abs(corr[1:]))
    
    best_x = x.copy()
    best_corr = corr.copy()
    best_max = current_max
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 1e-6
    
    # Precompute indices for fast access?
    # For each flip i, we need to update corr[k] for all k where i is involved.
    # i is involved in c[k] if:
    # 1. i < N-k (i is the first element of the pair) -> pair (i, i+k)
    # 2. i >= k (i is the second element of the pair) -> pair (i-k, i)
    
    while time.time() < deadline:
        # Pick random index to flip
        i = np.random.randint(N)
        old_val = x[i]
        new_val = -old_val
        
        # Calculate delta for all k
        # c[k] changes by:
        # - 2 * old_val * x[i+k] if i+k < N
        # - 2 * old_val * x[i-k] if i-k >= 0
        
        # We update corr in place for evaluation, then revert if rejected
        # But we need to know the NEW max.
        
        # Store old values of corr that will change
        # The indices k that change are those where i is in range.
        # k ranges from 0 to N-1.
        # For a fixed i:
        # k such that i+k < N => k < N-i
        # k such that i-k >= 0 => k <= i
        # So k is in [0, max(i, N-i-1)]? No.
        # k=0: c[0] = sum x^2 = N. Always constant. Skip k=0.
        
        # Let's just compute the new corr for the affected k's.
        # Affected k's:
        # k1: 1 <= k < N-i  (pair i, i+k)
        # k2: 1 <= k <= i   (pair i-k, i)
        
        # To avoid complex logic, since N=400 is small, we can just
        # update the whole corr array? No, O(N^2) per step is too slow for 110s.
        # O(N) per step is needed.
        
        # List of k's to update:
        ks = []
        # k < N-i
        if N - i > 1:
            ks.extend(range(1, N - i))
        # k <= i
        if i >= 1:
            ks.extend(range(1, i + 1))
        
        # Note: Some k might be duplicated if ranges overlap?
        # Range 1: 1 .. N-i-1
        # Range 2: 1 .. i
        # They are distinct sets of k values?
        # For a specific k, the change is:
        # delta_c[k] = -2 * old_val * ( (x[i+k] if i+k < N else 0) + (x[i-k] if i-k >= 0 else 0) )
        
        # We need to find the new max.
        # We can keep a histogram or just re-scan the affected k's and the current max.
        # If the current max k is affected, we need to find the next max.
        
        # Optimization: Maintain a sorted list of (abs_c, k)?
        # Or just keep an array and if max changes, re-scan.
        # Given N=400, re-scanning 400 elements is fast (microseconds).
        # So: Update corr[ks], then max(corr[1:]) is O(N).
        # Total O(N) per step. 110s / (400 ops) is huge.
        
        # Apply flip to x temporarily? No, just calculate delta.
        # x[i] is old_val.
        # x[i] becomes new_val.
        
        deltas = np.zeros(len(ks))
        for idx, k in enumerate(ks):
            term1 = 0
            term2 = 0
            if i + k < N:
                term1 = x[i + k]
            if i - k >= 0:
                term2 = x[i - k]
            deltas[idx] = -2 * old_val * (term1 + term2)
        
        # Apply deltas to corr
        old_corr_vals = corr[ks].copy()
        corr[ks] += deltas
        
        # Calculate new max
        # We only need to check the affected k's and the old max if it wasn't affected
        # But simple max(corr[1:]) is safe and fast enough for N=400.
        new_max = np.max(np.abs(corr[1:]))
        
        # Accept/Reject
        if new_max < current_max:
            # Always accept improvement
            x[i] = new_val
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_x = x.copy()
                best_corr = corr.copy()
        else:
            # Metropolis
            delta_E = new_max - current_max
            if T > min_T:
                prob = np.exp(-delta_E / T)
                if np.random.random() < prob:
                    x[i] = new_val
                    current_max = new_max
                else:
                    # Revert corr
                    corr[ks] = old_corr_vals
            else:
                # Revert
                corr[ks] = old_corr_vals
        
        # Cool down
        T *= cooling
        
        # Optional: Restart if stuck?
        # For now, simple SA.

    return best_x
# EVOLVE-BLOCK-END
