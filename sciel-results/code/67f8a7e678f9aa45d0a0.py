"""
Program for the first autocorrelation inequality.
Strategy: Use a multi-block "comb" structure (many thin separated rectangles) 
optimized via FFT-accelerated scoring and Simulated Annealing.
This explores a different design family (high-frequency oscillation/separation)
compared to the current two-block approach, using a different search method
(SA) enabled by faster evaluation (FFT).
"""
import time
import numpy as np

N = 600
dx = 0.5 / N

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    # Use FFT for faster convolution if N is large enough, 
    # though for N=600 direct conv is also fast. 
    # np.convolve is O(N^2). FFT is O(N log N).
    # For N=600, N^2 = 360,000. FFT size ~ 1024 or 2048.
    # Let's use np.convolve for simplicity and correctness first, 
    # but if we do many iterations, FFT might help.
    # Actually, for N=600, np.convolve is very fast (<1ms).
    # The bottleneck is the number of evaluations.
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Strategy: Multi-block "comb"
    # Represent h as a sequence of blocks.
    # Since the domain is fixed [-0.25, 0.25], we can define a periodic comb.
    # Let the comb have period T and pulse width W.
    # h(x) = A * sum_{k} rect((x - k*T)/W)
    # Due to symmetry and the interval boundaries, we need to be careful.
    # However, we can just define the array directly based on parameters.
    
    # Parameters for the comb:
    # 1. Number of peaks (K)
    # 2. Peak width (in number of intervals)
    # 3. Peak height (amplitude) - actually scale doesn't matter for normalized score, 
    #    but relative heights might if we allow non-uniform peaks.
    #    Let's start with uniform peaks.
    # 4. Phase / Offset
    
    # Let's define the state as a vector of heights h of length N.
    # But to guide the search, we'll use a structured initialization and 
    # perturbations that respect the "comb" nature or just general perturbations.
    
    # Better: Directly optimize the array h using Simulated Annealing.
    # The "comb" insight suggests that local minima for two blocks might be 
    # better than global minima found by simple local search on arbitrary h.
    # But SA on the full h vector is expensive (600 dims).
    
    # Alternative: Parameterized Comb.
    # h[i] = H if (i mod P) in [0, W) else 0.
    # This forces a periodic structure.
    # Let's try optimizing P (period), W (width), and H (height, irrelevant for norm) 
    # and maybe an offset.
    # Also, we can allow the comb to be centered or offset.
    
    # Let's try a few different comb configurations and then refine with local search.
    
    def generate_comb(period, width, offset, N):
        """
        Generate a comb-like step function.
        period: number of intervals between start of pulses.
        width: number of intervals the pulse is active.
        offset: starting index of the first pulse.
        """
        h = np.zeros(N)
        if period <= 0:
            return h
        # Indices where the pulse is active
        # The pulse is active at indices i such that (i - offset) % period < width
        # Be careful with negative modulo in Python, but it works correctly for non-negative i.
        i = np.arange(N)
        active = ((i - offset) % period) < width
        # Ensure indices are within bounds (they are, 0..N-1)
        h[active] = 1.0
        return h

    # We will search over (period, width, offset).
    # period is integer 1..N
    # width is integer 1..period
    # offset is integer 0..period-1 (effectively)
    
    # Since the score is invariant under translation (circular? No, linear convolution on finite domain),
    # the position matters because of the boundaries.
    # However, for a comb, the "center" of the comb matters.
    
    # Let's do a coarse grid search over period and width, and optimize offset.
    
    # To speed up, we can precompute or just evaluate quickly.
    # N=600. 
    # Periods to try: 2, 3, 4, ..., 100? 
    # Widths: 1, 2, ... up to period/2?
    
    # Let's use a random search with SA-like acceptance for the parameters.
    
    best_params = None
    best_h_comb = None
    
    # Initial guess: try a few manual combs
    initial_configs = [
        (10, 1, 0),   # Fine comb
        (20, 2, 0),
        (50, 5, 0),
        (100, 10, 0),
        (30, 3, 0),
        (7, 1, 0),
    ]
    
    for p, w, o in initial_configs:
        if time.time() > deadline:
            break
        h = generate_comb(p, w, o, N)
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h
            best_params = (p, w, o)
            
    # Now, let's do a local search on the parameters (p, w, o)
    # We can treat p and w as integers, o as integer.
    # Or we can allow "soft" combs where the edges are not hard cutoffs?
    # No, let's stick to hard step functions for now.
    
    # Let's perform a Simulated Annealing on the parameter space (p, w, o)
    # But p and w are discrete.
    # We can also mix in "broken" combs where some pulses have different heights.
    
    # Let's define a more flexible representation:
    # h is an array.
    # We initialize with a comb.
    # Then we apply random perturbations to individual bins.
    # This allows the comb to break up if beneficial.
    
    # Reset best_h to the best comb found so far.
    if best_params:
        best_h = generate_comb(*best_params, N)
    
    # Simulated Annealing on the vector h
    # T0 = 1.0, T_end = 0.01, cooling rate = 0.99
    # But 600 dimensions is a lot.
    # Let's limit perturbations to a few indices per step.
    
    T = 1.0
    T_min = 1e-4
    alpha = 0.995
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Precompute for speed? 
    # compute_score is O(N^2). 
    # We can use FFT to speed up convolution.
    # Let's implement an FFT-based score for the inner loop.
    
    def compute_score_fft(h):
        s = np.sum(h)
        if s < 1e-12:
            return 0.0
        # Pad to power of 2 for FFT efficiency? 
        # N=600. Convolution length 1199. 
        # Next power of 2 is 2048.
        n_fft = 2048
        H = np.fft.rfft(h, n=n_fft)
        Conv = np.fft.irfft(H * H, n=n_fft)
        # The linear convolution is in the first 2*N-1 elements.
        # Note: irfft returns the full symmetric spectrum if not careful, 
        # but H*H is the power spectrum of the real signal? 
        # Wait. Convolution in time is product in frequency.
        # If h is real, H is complex. H*H is not |H|^2.
        # We need H * H (element-wise multiplication of the FFT coefficients).
        # This corresponds to linear convolution.
        conv = Conv[:2*N-1]
        max_conv = np.max(conv)
        return max_conv / (s * s)
        
    # Verify FFT score matches direct score
    # h_test = np.random.rand(N)
    # assert abs(compute_score(h_test) - compute_score_fft(h_test)) < 1e-5
    
    # Use FFT score for the loop
    current_score = compute_score_fft(current_h)
    
    # SA Loop
    max_iters = 50000
    iter_count = 0
    
    while T > T_min and iter_count < max_iters:
        if time.time() > deadline:
            break
            
        # Generate perturbation
        # Pick 1-5 random indices to modify
        num_changes = np.random.randint(1, 6)
        indices = np.random.choice(N, num_changes, replace=False)
        
        new_h = current_h.copy()
        for idx in indices:
            # Multiplicative perturbation
            # Or additive? 
            # Multiplicative keeps scale somewhat stable.
            # Let's try setting to random value or scaling.
            # Since h >= 0, let's scale by a factor.
            factor = np.random.uniform(0.5, 1.5)
            new_h[idx] *= factor
            if new_h[idx] < 0:
                new_h[idx] = 0
                
        new_score = compute_score_fft(new_h)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = new_h
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
                
        T *= alpha
        iter_count += 1
        
    # Final check with direct computation to ensure precision
    final_score = compute_score(best_h)
    if final_score < best_score:
        best_score = final_score
    else:
        # Sometimes FFT might have slight numerical differences, 
        # but we return the array, so it's fine.
        pass
        
    return best_h.tolist()

# Quick self-test if run directly
if __name__ == "__main__":
    h = construct()
    print("Constructed h with score:", compute_score(np.array(h)))
