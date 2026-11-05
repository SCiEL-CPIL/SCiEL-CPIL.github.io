import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def normalize_sum(h, target=1.0):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h
        return h * (target / s)

    h_best = np.ones(N)
    val_best = compute_value(h_best)

    rng = np.random.default_rng(42)

    # Strategy: Minimize the L-infinity norm of the autoconvolution.
    # We use a "soft" objective: f(h) = sum(c_k^2) which is a proxy for max(c_k) but is smooth and convex (in h? No, quartic).
    # Actually, sum(c_k^2) is related to the energy of the spectrum.
    # Parseval's theorem: sum |c_k|^2 = sum |H_k|^4 (up to scaling).
    # Minimizing the L4 norm of the spectrum might help flatten the time-domain peaks?
    # Alternatively, use a direct local search on the max term with a randomized restart and larger steps.
    
    # Let's try a different structural approach:
    # The objective is scale invariant. Fix sum(h) = 1.
    # We want to minimize max_k c_k.
    # Note that c_0 = sum h_i^2 is always one of the terms.
    # By Cauchy-Schwarz, sum h_i^2 >= (sum h_i)^2 / N = 1/N.
    # So max(c) >= 1/N.
    # The value V = 2 * N * max(c) / 1^2 = 2 * N * max(c).
    # So V >= 2.
    # Wait, the seed value is 2.0.
    # If max(c) >= 1/N, then V >= 2.
    # This suggests we cannot beat 2.0?
    # Let's re-read the seed calculation.
    # h = ones(400). sum = 400.
    # conv(ones, ones) max is 400.
    # V = 2 * 400 * 400 / 400^2 = 2.
    # My bound: sum h_i^2 >= (sum h_i)^2 / N.
    # If sum h_i = S, sum h_i^2 >= S^2/N.
    # c_0 = sum h_i^2.
    # max(c) >= c_0 >= S^2/N.
    # V = 2 N max(c) / S^2 >= 2 N (S^2/N) / S^2 = 2.
    # So the theoretical lower bound is 2.0?
    # But the inspirations score > 1.0, meaning V < 2.0.
    # Where is the flaw?
    # "max is taken over its signed entries".
    # If max(c) is negative? No, c_0 = sum h_i^2 is always positive.
    # So max(c) is always >= c_0 > 0.
    # So V >= 2.
    # Why do inspirations score 1.125 (V ~ 1.77)?
    # Maybe I am misunderstanding np.convolve or the bound.
    # Let's check h = [1, -1, 1]. N=3.
    # sum = 1.
    # conv:
    # 1*1 = 1
    # 1*(-1) + (-1)*1 = -2
    # 1*1 + (-1)*(-1) + 1*1 = 3
    # (-1)*1 + 1*(-1) = -2
    # 1*1 = 1
    # conv = [1, -2, 3, -2, 1].
    # max = 3.
    # V = 2 * 3 * 3 / 1 = 18.
    #
    # Let's check h = [1, 0, -1]. N=3.
    # sum = 0. Invalid.
    #
    # Let's check h = [1, -0.5, 1]. N=3.
    # sum = 1.5.
    # conv:
    # 1
    # -0.5 -0.5 = -1
    # 1 + 0.25 + 1 = 2.25
    # -0.5 -0.5 = -1
    # 1
    # max = 2.25.
    # V = 2 * 3 * 2.25 / 2.25 = 6.
    #
    # Is it possible to get V < 2?
    # The bound V >= 2 seems robust if max(c) >= c_0.
    # c_0 is the sum of squares.
    # Is c_0 always the maximum?
    # For h=[1, -1], c=[1, -2, 1]. max=1. c_0=2.
    # Here max(c) < c_0.
    # So my assumption "max(c) >= c_0" is WRONG.
    # c_0 is the sum of squares.
    # In np.convolve, the center index is N-1.
    # For h=[1, -1], center is index 1. Value -2.
    # The sum of squares is 2.
    # Where is 2 in the conv?
    # It is NOT in the conv.
    # The conv contains cross terms.
    # c_k = sum h_i h_{k-i}.
    # The sum of squares is sum h_i h_i.
    # This corresponds to the terms where i = k-i => 2i=k.
    # These terms are distributed across different k values.
    # Specifically, the term h_i^2 appears in c_{2i}.
    # So the "energy" of the squares is spread out.
    # Therefore, max(c) can be much smaller than sum h_i^2.
    #
    # So we want to minimize max(c).
    # This is equivalent to designing a sequence with low peak-to-average power ratio in its autoconvolution?
    # Actually, we want to minimize the peak of the autoconvolution.
    # This is related to "low autocorrelation" sequences.
    # Barker codes have low sidelobes, but the main lobe (lag 0) is large.
    # But here, the "main lobe" (sum of squares) is split.
    #
    # Let's try a random search with a focus on making the conv entries small.
    # We can use a greedy approach or simulated annealing.
    
    h_cur = np.ones(N)
    h_cur = normalize_sum(h_cur, 1.0)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    for step in range(50000):
        if time.time() > deadline:
            break
            
        val_cur = compute_value(h_cur)
        
        # Perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, 0.1)
        h_new = h_cur.copy()
        h_new[idx] += delta
        h_new = normalize_sum(h_new, 1.0)
        
        val_new = compute_value(h_new)
        
        if val_new < val_cur:
            h_cur = h_new
            if val_new < val_best:
                val_best = val_new
                h_best = h_cur.copy()
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-(val_new - val_cur) / T):
                h_cur = h_new
        
        T = max(T * cooling, T_min)
        
    return h_best
