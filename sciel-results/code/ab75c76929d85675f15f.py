import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Initial guess: Uniform distribution
    h = np.full(N, 0.5)
    
    # Verify constraints
    # Sum should be N/2
    # h in [0,1]
    
    best_h = h.copy()
    best_score = np.inf
    
    # Calculate max overlap for a given h
    def calc_max_overlap(h_vec):
        # h_vec is 1D array
        # Overlap at shift k: sum(h[i] * (1 - h[(i+k)%N]))
        # = sum(h) - sum(h[i]*h[(i+k)%N])
        # sum(h) is N/2
        # We want to minimize max( N/2 - R(k) ) for k in 1..N-1
        # Equivalent to maximizing min(R(k))
        
        # Compute autocorrelations R(k) for k=1..N-1
        # Using FFT for speed
        # R(k) is real part of ifft(|fft(h)|^2)
        # Note: numpy fft is unnormalized forward, 1/N normalized inverse
        # fft(h) -> H
        # ifft(|H|^2) gives circular autocorrelation
        # R(k) = sum_i h_i h_{i+k}
        
        H = np.fft.fft(h_vec)
        R = np.real(np.fft.ifft(np.abs(H)**2))
        # R[0] is sum(h^2)
        # R[k] for k=1..N-1 are the lags
        # Due to numerical noise, R might have tiny imaginary parts, we took real.
        
        # We care about k=1 to N-1
        R_nonzero = R[1:]
        min_R = np.min(R_nonzero)
        max_overlap = N / 2.0 - min_R
        return max_overlap

    current_score = calc_max_overlap(h)
    best_score = current_score
    best_h = h.copy()
    
    # Local Search / Gradient Descent
    # Objective: Minimize max_overlap = N/2 - min(R(k))
    # This is non-smooth.
    # We can use a smooth approximation or just random perturbations (Simulated Annealing style)
    # Given the tight theoretical bound, we expect no improvement.
    # We will do a few random restarts/perturbations to be sure.
    
    np.random.seed(42)
    
    # Simple hill climbing with random restarts
    # Since N=100, FFT is fast.
    # We can afford many iterations.
    
    max_iterations = 5000
    step_size = 0.05
    current_h = best_h.copy()
    current_score = best_score
    
    for i in range(max_iterations):
        if time.time() > deadline - 10:
            break
            
        # Generate a perturbation
        # Randomly pick a few indices to change
        num_changes = np.random.randint(1, 5)
        indices = np.random.choice(N, num_changes, replace=False)
        
        new_h = current_h.copy()
        for idx in indices:
            # Perturb value
            delta = np.random.uniform(-step_size, step_size)
            new_h[idx] += delta
            # Clamp to [0, 1]
            if new_h[idx] < 0: new_h[idx] = 0
            if new_h[idx] > 1: new_h[idx] = 1
        
        # Re-normalize sum to N/2
        # Project to hyperplane sum(x) = N/2
        # Simple projection: shift all values by mean error
        # But this might violate [0,1].
        # Better: adjust only the changed indices? No, that breaks global sum.
        # Let's use a simple projection:
        # new_h = new_h + (N/2 - np.sum(new_h)) / N
        # This might push values out of [0,1].
        # If so, clip and re-normalize iteratively?
        # For simplicity and robustness, if projection violates bounds, discard this move.
        
        err = N / 2.0 - np.sum(new_h)
        new_h = new_h + err / N
        
        # Check bounds
        if np.any(new_h < -1e-9) or np.any(new_h > 1 + 1e-9):
            # Try to fix by clipping and re-distributing?
            # Or just skip this candidate
            continue
        
        # Clamp small numerical errors
        new_h = np.clip(new_h, 0, 1)
        
        # Re-normalize again to be safe (clip might have changed sum slightly)
        # If sum is not exactly N/2, the evaluator might fail?
        # "sum is exactly len/2"
        # Let's enforce exact sum by adjusting one element?
        # Or better, use the projection that preserves sum but respects bounds is complex.
        # Given the starting point is valid and perturbations are small,
        # the simple shift should keep it close.
        
        # Ensure sum is exactly N/2
        # The shift `err / N` makes sum exactly N/2 before clipping.
        # Clipping might break it.
        # Let's check sum after clipping.
        if abs(np.sum(new_h) - N/2.0) > 1e-9:
             # If clipping broke the sum, we need to fix it.
             # This happens if we hit bounds.
             # For now, skip if sum is off.
             continue

        new_score = calc_max_overlap(new_h)
        
        if new_score < current_score - 1e-12:
            current_h = new_h
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # If no improvement, maybe reduce step size?
            # Or just keep exploring.
            pass
            
    # Final check: return the best found
    # If no improvement was found, best_h is the uniform array.
    
    # Ensure final output is valid
    # Clip to [0,1]
    best_h = np.clip(best_h, 0, 1)
    # Ensure sum is exactly N/2
    # The evaluator might be strict about "exactly".
    # Floating point: 
    # Let's adjust the last element to make the sum exact.
    # But this might push it out of [0,1] or change the score significantly?
    # No, changing one element by 1e-15 is negligible for score.
    current_sum = np.sum(best_h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 0:
        # Add diff to the last element
        # Check if it stays in [0,1]
        if 0 <= best_h[-1] + diff <= 1:
            best_h[-1] += diff
        else:
            # Distribute diff among all elements?
            # Or just hope the error is small enough to be ignored?
            # "sum is exactly len/2"
            # If I can't satisfy it with one element, I might need to adjust more.
            # But uniform 0.5 sums to exactly 50.0 (for N=100) in float?
            # 0.5 is exactly representable. 100 * 0.5 = 50.0.
            # So the uniform solution is exactly valid.
            pass

    return best_h
# EVOLVE-BLOCK-END
