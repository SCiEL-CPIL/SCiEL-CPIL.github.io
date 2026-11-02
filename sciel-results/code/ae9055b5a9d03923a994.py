import numpy as np
import time

N = 400

def _autocorr_max_abs(x):
    """Compute max absolute value of circular autocorrelation using FFT."""
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    return np.max(np.abs(acf))

def _compute_score(x):
    """
    Compute the score: max|autoconvolution| / (integral)^2.
    """
    acf_max = _autocorr_max_abs(x)
    total_sum = np.sum(x)
    norm_sq = total_sum * total_sum
    if norm_sq < 1e-12:
        return np.inf
    return acf_max / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_x = np.ones(N)
    best_score = _compute_score(best_x)
    
    rng = np.random.RandomState(42)
    
    # Strategy: Discrete Local Search on Hypercube
    # We treat the problem as minimizing the raw max absolute autoconvolution.
    # Note: For binary sequences with sum != 0, minimizing max|acf| is the primary driver.
    # The denominator (sum)^2 is constant for a fixed pattern of +1/-1 counts? 
    # No, flipping changes sum. 
    # However, usually for these "balanced" sequences, sum is small. 
    # If sum is 0, score is inf. We must avoid sum=0.
    # Actually, the task says "minimizing the maximum absolute autoconvolution divided by squared integral".
    # If we stick to binary {-1, 1}, sum can be anything from -N to N.
    # The seed program uses binary {-1, 1}.
    
    # Heuristic: "Max-Pooling" Greedy Descent
    # 1. Start with a random or structured initialization.
    # 2. Compute autocorrelation.
    # 3. Find the lag k where |acf[k]| is maximum.
    # 4. Determine which single flip (index i) would most reduce this peak.
    #    Flipping x[i] changes acf by -2 * x[i] * (x shifted by i) + 2 * x[i] * (x shifted by -i)?
    #    Actually, acf_new = acf_old - 2*x[i]*correlation_with_i.
    #    This is complex to compute for all i at once efficiently without O(N^2) or careful FFT tricks.
    #    
    # Alternative Efficient Discent:
    # Instead of finding the *global* best flip, we can use a "steepest descent" approximation 
    # or a "coordinate descent" that picks the index i that minimizes the resulting max|acf|.
    # 
    # Let's use a simpler, robust discrete local search:
    # 1. Compute current acf.
    # 2. Identify the peak lag k*.
    # 3. The peak value is P = acf[k*].
    # 4. To reduce P, we want to flip an index i such that the change in acf[k*] is large and negative (if P>0) or positive (if P<0).
    #    Change in acf[k] when flipping x[i]:
    #    acf_new[k] = acf_old[k] - 2 * x[i] * (x_{k-i} - x_{-k-i})? No.
    #    Let's derive:
    #    acf[k] = sum_j x[j] x[j+k]
    #    Flip x[i] -> x[i]' = -x[i]
    #    Terms involving x[i]:
    #    j=i: x[i] x[i+k] -> -x[i] x[i+k] (change -2 x[i] x[i+k])
    #    j=i-k: x[i-k] x[i] -> x[i-k] (-x[i]) (change -2 x[i-k] x[i])
    #    So delta_acf[k] = -2 * x[i] * (x[i+k] + x[i-k])
    #    
    #    We want to minimize max_j |acf_new[j]|.
    #    A good heuristic is to target the current maximum lag k*.
    #    We want delta_acf[k*] to have the opposite sign of acf[k*] and large magnitude.
    #    Let sign_target = -np.sign(acf[k*])
    #    We want sign_target * delta_acf[k*] > 0 (i.e., delta reduces the magnitude).
    #    delta_acf[k*] = -2 * x[i] * (x[i+k*] + x[i-k*])
    #    We want -sign(acf[k*]) * (-2 * x[i] * (x[i+k*] + x[i-k*])) > 0
    #    => sign(acf[k*]) * x[i] * (x[i+k*] + x[i-k*]) > 0
    
    # Let's implement a loop that:
    # 1. Finds k* (argmax |acf|)
    # 2. Computes the "potential" for each i to reduce the peak at k*.
    # 3. Picks the i with the highest potential (or a random top-k to avoid cycles).
    # 4. Flips i.
    # 5. Recomputes score.
    
    # To speed this up, we can do batches of flips.
    
    def _local_search(x_start, max_iters, rng_local):
        x = x_start.copy()
        cur_acf = None
        cur_score = np.inf
        
        # Initial computation
        cur_acf = np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(x)), N)
        total_sum = np.sum(x)
        norm_sq = total_sum * total_sum
        if norm_sq < 1e-12:
             # If sum is 0, we have a problem. Force a flip to break symmetry?
             # Or just return inf.
             # Let's try to move away from sum=0 if we start there.
             # But random start usually has non-zero sum.
             pass
        cur_score = np.max(np.abs(cur_acf)) / norm_sq if norm_sq > 1e-12 else np.inf
        
        best_x_local = x.copy()
        best_score_local = cur_score
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find peak lag
            abs_acf = np.abs(cur_acf)
            k_star = np.argmax(abs_acf)
            
            # Compute potential for each flip i
            # delta_acf[k] for all k is not needed, just for k_star.
            # But flipping i affects ALL acf[k].
            # The heuristic "reduce the current peak" is myopic.
            # A better discrete heuristic: "Minimize the sum of squares of acf" (energy)
            # or "Minimize the max" via a surrogate.
            
            # Let's try a different structural approach:
            # Use the "potential" of flipping i to reduce the *global* max.
            # This is hard.
            
            # Instead, let's use a "steepest descent" on the discrete domain:
            # Try flipping each i? O(N^2) per step. Too slow for 100s if N=400.
            # 400 flips * 400 cost = 160k ops. FFT is O(N log N).
            # If we do 1000 iterations, 160M ops. Might be okay in Python? No, Python loops are slow.
            # We need vectorized evaluation of flips.
            
            # Vectorized Flip Evaluation:
            # We want to evaluate score(x with x[i] flipped) for all i simultaneously.
            # x_new_i = x.copy(); x_new_i[i] = -x[i]
            # This is O(N) memory per i, or O(N^2) total.
            
            # Let's stick to a single best flip per iteration, but find it smartly.
            # Heuristic: The flip that reduces the peak at k* the most.
            # delta_acf[k_star] = -2 * x[i] * (x[(i+k_star)%N] + x[(i-k_star)%N])
            # We want this delta to oppose the current peak.
            # current_peak_val = cur_acf[k_star]
            # We want delta_acf[k_star] * current_peak_val < 0
            # => -2 * x[i] * (x[i+k] + x[i-k]) * cur_acf[k] < 0
            # => x[i] * (x[i+k] + x[i-k]) * cur_acf[k] > 0
            
            peak_val = cur_acf[k_star]
            if peak_val == 0:
                break
                
            # Vectorized calculation of "benefit"
            # x_shifted_plus = np.roll(x, -k_star)  # x[i+k]
            # x_shifted_minus = np.roll(x, k_star)  # x[i-k]
            # benefit = x * (x_shifted_plus + x_shifted_minus) * peak_val
            # We want benefit > 0.
            # To minimize the peak, we want the largest positive benefit.
            
            x_plus = np.roll(x, -k_star)
            x_minus = np.roll(x, k_star)
            benefit = x * (x_plus + x_minus) * peak_val
            
            # Pick the index with max benefit
            i_flip = np.argmax(benefit)
            
            # Check if it actually helps
            if benefit[i_flip] <= 0:
                # No beneficial flip found for this peak?
                # Try a random flip to escape local minima
                i_flip = rng_local.randint(N)
            
            # Perform flip
            x[i_flip] = -x[i_flip]
            
            # Recompute acf
            X = np.fft.rfft(x)
            new_acf = np.fft.irfft(X * np.conj(X), N)
            
            # Update score
            total_sum = np.sum(x)
            norm_sq = total_sum * total_sum
            if norm_sq < 1e-12:
                 new_score = np.inf
            else:
                 new_score = np.max(np.abs(new_acf)) / norm_sq
            
            if new_score < best_score_local:
                best_score_local = new_score
                best_x_local = x.copy()
                
            # Stagnation check: if we didn't improve in last 50 steps, restart?
            # For now, just continue.
            
            cur_acf = new_acf
            cur_score = new_score
            
        return best_x_local, best_score_local

    # Multi-start
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            x_start = np.ones(N)
        elif r == 1:
            x_start = np.sign(np.sin(np.arange(N) * 0.1))
            x_start[x_start == 0] = 1
        else:
            x_start = rng.choice([-1, 1], N)
            
        # Ensure sum is not 0 for the first few steps? 
        # The local search handles it by returning inf if sum=0.
        # But if we start with sum=0, we are stuck?
        # If sum=0, score=inf. Any flip changes sum by +/-2.
        # So we can escape.
        
        x_opt, score_opt = _local_search(x_start, 500, rng)
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt

    return best_x
