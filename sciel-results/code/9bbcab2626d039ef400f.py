import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random +/- 1 sequence
    x = np.random.choice([1, -1], size=N)
    
    # Precompute autocorrelation
    # r[k] = sum(x[i] * x[i+k]) for k in 0..N-1 (positive lags)
    # We care about |r[k]| for k != 0
    r = np.correlate(x, x, mode='full')
    # r is length 2N-1. Center is N-1.
    # Positive lags are r[N-1+1 : 2N-1]
    # Negative lags are r[0 : N-1]
    # Due to symmetry for real signals, r[k] = r[-k].
    # So we only need to look at r[N-1+1 : 2N-1]
    # r[N-1] is r[0] = N
    
    def get_side_lobe_peak(r_pos):
        # r_pos corresponds to lags 1 to N-1
        if len(r_pos) == 0:
            return 0
        return np.max(np.abs(r_pos))
    
    def get_objective(r_pos):
        # Let's try minimizing the peak, but maybe a mix of peak and sum is better?
        # The task says "upper bound on C3". 
        # In the context of the 3rd autocorrelation inequality, it might be related to 
        # ||r||_3. Let's minimize the peak as a proxy, or the L3 norm.
        # L3 norm: (sum |r_k|^3)^(1/3)
        # Let's minimize the sum of cubes of the side lobes.
        # This is differentiable (sort of) and smooths the landscape.
        return np.sum(np.abs(r_pos)**3)
    
    best_obj = get_objective(r[N-1+1:])
    best_x = x.copy()
    
    # Simulated Annealing
    T = 100.0
    T_min = 1e-3
    alpha = 0.999
    
    # Precompute indices for faster access? 
    # Flipping x[i] affects r[k] for all k.
    # r_new[k] = r_old[k] - 2 * x[i] * (x[i+k] if i+k < N else 0) ... 
    # Wait, r[k] = sum_{j=0}^{N-k-1} x[j] x[j+k].
    # If we flip x[i], it affects terms where j=i or j+k=i.
    # j=i -> term x[i]x[i+k] is in r[k].
    # j+k=i -> j=i-k -> term x[i-k]x[i] is in r[k].
    # So r[k] changes by -2 x[i] x[i+k] (if i+k < N) and -2 x[i] x[i-k] (if i-k >= 0).
    
    # Let's use a simpler approach: recompute r for the candidate? 
    # O(N^2) per step is too slow for 400 length if we do many steps.
    # 400^2 = 160,000 ops. 1000 steps = 160M ops. Doable in Python? Maybe slow.
    # Incremental update is O(N).
    
    # Let's implement incremental update carefully.
    # r_pos[k] corresponds to lag k+1.
    # r[k] = sum_{j=0}^{N-k-1} x[j] x[j+k]
    
    current_obj = best_obj
    current_x = x.copy()
    current_r = r.copy()
    
    # r array in numpy correlate 'full' mode:
    # index 0 corresponds to lag -(N-1)
    # index N-1 corresponds to lag 0
    # index N-1+k corresponds to lag k
    
    # We will maintain current_r
    # Objective is based on lags 1..N-1
    # Indices in current_r: N-1+1 to 2N-2
    
    def compute_obj(r_arr):
        side = r_arr[N-1+1 : 2*N-1]
        return np.sum(np.abs(side)**3)
    
    for step in range(1000000):
        if time.time() > deadline:
            break
            
        # Choose a random index to flip
        i = np.random.randint(0, N)
        
        # Calculate delta in objective
        # We need to update r_arr
        # r[k] for k in 1..N-1
        # Change in r[k] when flipping x[i]:
        # delta_r[k] = -2 * x[i] * (x[i+k] if i+k < N else 0) - 2 * x[i] * (x[i-k] if i-k >= 0 else 0)
        
        # This is O(N) to calculate all deltas.
        # Let's compute the new r values for the affected lags.
        
        # Actually, let's just compute the new objective directly for the candidate?
        # If we flip x[i], we can compute the new side lobes?
        # No, it affects all lags.
        
        # Let's try a simpler heuristic: Random restarts + Local Search on a smaller subset?
        # Or just use the incremental update.
        
        # Update r
        # k ranges from 1 to N-1
        # For a given k:
        # term1: if i+k < N, x[i]x[i+k] is in r[k]. Flipping x[i] changes it to -x[i]x[i+k]. Diff: -2 x[i] x[i+k].
        # term2: if i-k >= 0, x[i-k]x[i] is in r[k]. Flipping x[i] changes it to x[i-k](-x[i]). Diff: -2 x[i] x[i-k].
        
        # We can vectorize this update.
        # k = np.arange(1, N)
        # idx1 = i + k
        # valid1 = idx1 < N
        # idx2 = i - k
        # valid2 = idx2 >= 0
        
        # delta_r = -2 * current_x[i] * (
        #     np.where(valid1, current_x[idx1[valid1]], 0) + 
        #     np.where(valid2, current_x[idx2[valid2]], 0)
        # )
        # This is tricky to vectorize cleanly without creating large temp arrays each step.
        
        # Alternative: Just recompute the objective from scratch for the candidate?
        # Correlation is O(N log N) with FFT.
        # 1000 steps * O(N log N) might be okay.
        # But we need the best result.
        
        # Let's stick to a simpler local search:
        # Try flipping each bit, keep the best one (Hill Climbing).
        # If stuck, do a random restart or simulated annealing move.
        
        # Given the complexity of incremental update in pure Python loops, 
        # let's use FFT for the objective evaluation which is fast in NumPy.
        
        # Candidate: flip bit i
        old_val = current_x[i]
        current_x[i] = -old_val
        
        # Compute new objective
        # Use FFT to compute autocorrelation
        # Zero pad to 2*N
        # F = fft(current_x, 2*N)
        # R = ifft(F * conj(F))
        # Real part is r
        # This is O(N log N).
        
        # Is it faster than O(N) loop? 
        # For N=400, O(N) is 400 ops. O(N log N) is ~4000 ops.
        # Python overhead for FFT call might dominate.
        # Let's try incremental update with pre-allocated arrays.
        
        # Revert for now, let's try a different strategy:
        # Use a known good sequence structure?
        # M-sequence? Barker?
        # For N=400, we can't have a perfect Barker code.
        # Random sequences have PSL ~ sqrt(N log N) ~ 20 * 6 = 120?
        # Seed PSL is 399.
        # So random is already much better.
        
        # Let's just do random restarts and simple hill climbing with FFT.
        
        # Actually, let's just return a random sequence if we can't do complex search?
        # No, we should try to improve.
        
        # Let's implement a simple Hill Climbing with FFT.
        
        # Revert
        current_x[i] = old_val
        
        # Calculate current obj
        curr_obj = compute_obj(current_r)
        
        # Try flipping a random bit
        i = np.random.randint(0, N)
        old_val = current_x[i]
        current_x[i] = -old_val
        
        # Compute new r
        # Using FFT
        F = np.fft.fft(current_x, 2*N)
        R = np.fft.ifft(F * np.conj(F)).real
        # R is the circular autocorrelation? No, zero padding makes it linear.
        # R[0:N] are lags 0..N-1?
        # Check:
        # fft(x, 2N) -> F
        # ifft(F*conj(F)) -> r
        # r[k] for k=0..N-1 are the positive lags.
        
        new_obj = np.sum(np.abs(R[1:N])**3)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_obj - curr_obj
        if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
            # Accept
            current_obj = new_obj
            current_r = R # Store full r? We only need side lobes for obj, but for next step?
            # We need to update current_r for the next step's baseline?
            # If we use FFT every time, we don't need current_r.
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = current_x.copy()
        else:
            # Reject
            current_x[i] = old_val
            
        # Cool down
        T *= alpha
        
    return best_x

# EVOLVE-BLOCK-END
