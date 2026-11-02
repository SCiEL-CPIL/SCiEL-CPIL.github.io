import numpy as np
import time
import itertools

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_ratio(h):
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)
    
    # The ratio is scale invariant. If we scale h by c, numerator scales by c^2, denominator by c^3? 
    # Wait. 
    # Num: ||ch||_2^2 = c^2 ||h||_2^2
    # Den: ||ch||_1 ||ch||_inf ||ch*ch||_inf = (c ||h||_1) (c ||h||_inf) (c^2 ||h*h||_inf) = c^4 ...
    # No, autoconvolution of ch is c^2 (h*h).
    # So Den = c ||h||_1 * c ||h||_inf * c^2 ||h*h||_inf = c^4 ...
    # Let's re-check.
    # h_i -> c h_i
    # L2^2 -> c^2 sum h_i^2
    # L1 -> c sum h_i
    # L_inf -> c max h_i
    # Conv h*h -> c^2 conv(h,h)
    # L_inf of conv -> c^2 max conv
    # Denominator product: (c L1) * (c L_inf) * (c^2 MaxConv) = c^4 L1 L_inf MaxConv
    # Ratio: c^2 / c^4 = 1/c^2.
    # So larger heights give smaller ratio. We should keep max height = 1.
    # Actually, if we restrict to {0, 1}, the ratio is fixed for a given support set S.
    # Let's verify: if h is 0/1, L2^2 = |S|, L1 = |S|, L_inf = 1.
    # Ratio = |S| / (|S| * 1 * MaxConv) = 1 / MaxConv.
    # So for 0/1 heights, maximizing the ratio is equivalent to MINIMIZING the maximum autoconvolution peak.
    # This is a classic problem: finding a set with minimal additive energy or minimal peak in its autocorrelation.
    # This is related to Sidon sets (B2 sets) where all pairwise sums are distinct (or mostly distinct).
    # If all sums are distinct, MaxConv is 1 (only the diagonal terms contribute, i.e., h_i * h_i = 1).
    # If there are collisions, MaxConv > 1.
    # So the goal is to find a subset S of {0..49} such that the number of solutions to i+j = k is minimized for all k.
    # Ideally, we want a Sidon set in [0, 49].
    # The maximum size of a Sidon set in [1, N] is approx sqrt(N). Here N=50, so size approx 7-8.
    # Let's search for sparse sets with low autoconvolution peaks.

    best_ratio = 0.0
    best_h = None

    # Initialize with a known good Sidon set or random sparse sets
    # A Sidon set in [0, 49]:
    # Example: 0, 1, 4, 10, 18, 23, 30
    candidates = []
    
    # Generate random sparse sets of various sizes
    import random
    random.seed(42)
    
    for size in range(3, 15):
        for _ in range(200):
            s = set(random.sample(range(N), size))
            h = np.zeros(N)
            for idx in s:
                h[idx] = 1.0
            candidates.append(h)
            
    # Also include some structured sets
    # Arithmetic progressions (bad for Sidon, but good baseline)
    for step in range(2, 10):
        h = np.zeros(N)
        for i in range(0, N, step):
            h[i] = 1.0
        candidates.append(h)
        
    # Evaluate initial candidates
    for h in candidates:
        if time.time() > deadline:
            break
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()

    # Local Search: Add, Remove, Swap
    # We represent the state as a set of indices
    current_h = best_h.copy() if best_h is not None else np.zeros(N)
    current_ratio = best_ratio
    
    # If we started with 0, initialize with a single 1
    if current_ratio == 0:
        current_h[0] = 1.0
        current_ratio = compute_ratio(current_h)
        best_h = current_h.copy()
        best_ratio = current_ratio

    # Perform local search
    # We will iterate through moves: Add a 0->1, Remove a 1->0, Swap a 1->0 and 0->1
    # To speed up, we can precompute the autoconvolution and update it incrementally, 
    # but N=50 is small, so O(N^2) per eval is fine.
    # 50^2 = 2500 ops. 10000 evals = 25M ops. Very fast.
    
    max_iterations = 10000
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Add
            zeros = np.where(current_h == 0)[0]
            if len(zeros) == 0: continue
            idx = random.choice(zeros)
            new_h = current_h.copy()
            new_h[idx] = 1.0
        elif move_type < 0.6:
            # Remove
            ones = np.where(current_h == 1)[0]
            if len(ones) <= 1: continue
            idx = random.choice(ones)
            new_h = current_h.copy()
            new_h[idx] = 0.0
        else:
            # Swap
            ones = np.where(current_h == 1)[0]
            zeros = np.where(current_h == 0)[0]
            if len(ones) == 0 or len(zeros) == 0: continue
            idx_out = random.choice(ones)
            idx_in = random.choice(zeros)
            new_h = current_h.copy()
            new_h[idx_out] = 0.0
            new_h[idx_in] = 1.0
            
        new_ratio = compute_ratio(new_h)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_ratio > current_ratio:
            current_h = new_h
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = current_h.copy()
        else:
            delta = current_ratio - new_ratio
            if delta < 1e-15:
                current_h = new_h
                current_ratio = new_ratio
            else:
                if random.random() < np.exp(-delta / T):
                    current_h = new_h
                    current_ratio = new_ratio
        
        T = max(T * cooling, T_min)
        
        # Occasionally, try to "repair" or restart from best
        if i % 500 == 0 and i > 0:
             # With small prob, jump to best
             if random.random() < 0.1:
                 current_h = best_h.copy()
                 current_ratio = best_ratio
                 T = 0.5 # Reheat slightly

    # Final check: try to optimize heights slightly around the best 0/1 solution?
    # Since the ratio is 1/MaxConv for 0/1, and scaling down reduces ratio, 0/1 is locally optimal 
    # for fixed support. But maybe non-binary heights help?
    # Let's do a quick coordinate descent on the best binary solution to see if non-binary helps.
    
    if best_h is not None:
        h_opt = best_h.copy()
        # Simple coordinate ascent on h_i in [0, 1]
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if h_opt[i] == 0:
                    # Try small positive
                    for val in [0.1, 0.5, 1.0]:
                        h_test = h_opt.copy()
                        h_test[i] = val
                        r_test = compute_ratio(h_test)
                        if r_test > compute_ratio(h_opt):
                            h_opt[i] = val
                            improved = True
                            break
                else:
                    # Try changing value
                    for val in [0.0, 0.1, 0.5, 1.0]:
                        if val == h_opt[i]: continue
                        h_test = h_opt.copy()
                        h_test[i] = val
                        r_test = compute_ratio(h_test)
                        if r_test > compute_ratio(h_opt):
                            h_opt[i] = val
                            improved = True
                            break
            if not improved:
                break
        
        final_ratio = compute_ratio(h_opt)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_h = h_opt

    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
