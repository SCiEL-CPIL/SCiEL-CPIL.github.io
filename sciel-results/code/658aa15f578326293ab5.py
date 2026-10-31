import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # The maximum ratio is 1.0, achieved by a delta function (spike).
    # Any spread reduces the ratio because it increases the convolution peak
    # relative to the squared L2 norm (or rather, the L2 norm decreases while
    # convolution peak stays high or increases).
    # Specifically, for h >= 0, ||h||_1 = 1:
    # Ratio = ||h||_2^2 / ||h*h||_infty
    # By Cauchy-Schwarz or simple logic:
    # If h is a spike at i, ||h||_2^2 = 1, (h*h)[2i] = 1, ratio = 1.
    # If h has two non-zero entries a, b (a+b=1), ||h||_2^2 = a^2+b^2 < 1.
    # (h*h) has a^2 at 2i, b^2 at 2j, and 2ab at i+j.
    # If i != j, max conv is max(a^2, b^2, 2ab).
    # Since (a+b)^2 = a^2+b^2+2ab = 1, and a^2+b^2 = 1-2ab.
    # If a=b=0.5, ||h||_2^2 = 0.5. Conv peaks: 0.25, 0.25, 0.5. Max = 0.5. Ratio = 1.
    # Wait. If a=b=0.5, ratio = 0.5 / 0.5 = 1.
    # Let's check a=0.6, b=0.4. ||h||_2^2 = 0.36+0.16=0.52.
    # Conv peaks: 0.36, 0.16, 0.48. Max = 0.48. Ratio = 0.52/0.48 > 1?
    # 0.52 / 0.48 = 13/12 = 1.0833.
    # So spreading can INCREASE the ratio above 1?
    # Let's re-evaluate.
    # h = [0.6, 0.4] at indices 0, 1.
    # h*h = [0.36, 0.48, 0.16]. Max = 0.48.
    # ||h||_2^2 = 0.52.
    # Ratio = 0.52 / 0.48 = 1.0833.
    #
    # Let's try 3 points.
    # The goal is to maximize ||h||_2^2 / max(h*h).
    # This is equivalent to minimizing max(h*h) / ||h||_2^2.
    # Let h be non-negative.
    #
    # Let's search for the optimal distribution.
    # The problem is convex in some sense? No, it's a ratio of quadratic forms.
    # But we can fix ||h||_2^2 = 1 and minimize max(h*h).
    # Or fix ||h||_1 = 1 and maximize ||h||_2^2 / max(h*h).
    #
    # Let's try a simple local optimization on the full vector h >= 0, sum(h)=1.
    # We can use gradient ascent on the log ratio or similar.
    # Or just simple coordinate descent / hill climbing.
    #
    # Given the time constraint, a robust approach is to start with a good candidate
    # and refine it.
    #
    # Let's try to find the optimal 2-point distribution first.
    # h = [a, 1-a] at positions i, j.
    # If j-i = 1 (adjacent):
    # Conv: a^2, 2a(1-a), (1-a)^2.
    # Max conv = max(a^2, (1-a)^2, 2a(1-a)).
    # ||h||_2^2 = a^2 + (1-a)^2.
    # Ratio R(a) = (a^2 + (1-a)^2) / max(a^2, (1-a)^2, 2a(1-a)).
    # By symmetry, assume a >= 0.5. Then a^2 >= (1-a)^2.
    # Also 2a(1-a) = a - a^2.
    # Compare a^2 and a-a^2. a^2 >= a-a^2 <=> 2a^2 >= a <=> a >= 0.5.
    # So for a in [0.5, 1], max conv is a^2.
    # R(a) = (a^2 + (1-a)^2) / a^2 = 1 + (1-a)^2/a^2.
    # To maximize R(a), we want (1-a)/a to be large, so a small.
    # But a >= 0.5. So min a is 0.5.
    # R(0.5) = 1 + 0.5/0.5 = 2? No.
    # (1-0.5)^2 / 0.5^2 = 1. R(0.5) = 1 + 1 = 2?
    # Let's re-calculate.
    # a=0.5. h=[0.5, 0.5]. ||h||_2^2 = 0.5.
    # Conv: 0.25, 0.5, 0.25. Max = 0.5.
    # Ratio = 0.5 / 0.5 = 1.
    # My formula: R(a) = (a^2 + (1-a)^2) / a^2.
    # At a=0.5: (0.25+0.25)/0.25 = 0.5/0.25 = 2.
    # Why the discrepancy?
    # Ah, max conv at a=0.5 is 2a(1-a) = 0.5, not a^2=0.25.
    # My condition "a^2 >= a-a^2" holds for a>=0.5.
    # At a=0.5, a^2 = 0.25, a-a^2 = 0.25. They are equal.
    # So max conv is 0.25? No, conv is [0.25, 0.5, 0.25].
    # Wait. Conv of [0.5, 0.5] is [0.25, 0.5, 0.25].
    # Indices: 0, 1, 2.
    # h[0]=0.5, h[1]=0.5.
    # (h*h)[0] = 0.5*0.5 = 0.25.
    # (h*h)[1] = 0.5*0.5 + 0.5*0.5 = 0.5.
    # (h*h)[2] = 0.5*0.5 = 0.25.
    # So max is 0.5.
    # My formula for conv terms:
    # Term at 2i: a^2.
    # Term at 2j: (1-a)^2.
    # Term at i+j: 2a(1-a).
    # If i=0, j=1.
    # 2i=0, 2j=2, i+j=1.
    # Values: a^2, (1-a)^2, 2a(1-a).
    # At a=0.5: 0.25, 0.25, 0.5.
    # So max is 2a(1-a) when a=0.5.
    # For a > 0.5, say a=0.6.
    # a^2=0.36, (1-a)^2=0.16, 2a(1-a)=0.48.
    # Max is 0.48.
    # So for a in [0.5, 1], is 2a(1-a) always the max?
    # 2a(1-a) >= a^2 <=> 2-2a >= a <=> 2 >= 3a <=> a <= 2/3.
    # 2a(1-a) >= (1-a)^2 <=> 2a >= 1-a <=> 3a >= 1 <=> a >= 1/3.
    # So for a in [1/3, 2/3], 2a(1-a) is the max.
    # For a in [0.5, 2/3], max conv is 2a(1-a).
    # R(a) = (a^2 + (1-a)^2) / (2a(1-a)).
    # Let's maximize this for a in [0.5, 2/3].
    # Let f(a) = (a^2 + (1-a)^2) / (2a - 2a^2).
    # f'(a) = 0?
    # Symmetric around 0.5. So max at endpoints?
    # f(0.5) = 0.5 / 0.5 = 1.
    # f(2/3) = (4/9 + 1/9) / (2*2/3*1/3) = (5/9) / (4/9) = 5/4 = 1.25.
    # So at a=2/3, ratio is 1.25.
    # h = [2/3, 1/3].
    # ||h||_2^2 = 4/9 + 1/9 = 5/9.
    # Conv: 4/9, 4/9, 1/9. Max = 4/9.
    # Ratio = (5/9) / (4/9) = 1.25.
    #
    # Can we do better with 3 points?
    # Let's try h = [a, b, a] with a+b+2a=1? No, sum=1.
    # Let's just run a local optimizer on the full vector.
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N//2] = 1.0
    best_score = compute_ratio(best_h)
    
    def update_best(h, score):
        nonlocal best_h, best_score
        if score > best_score:
            best_score = score
            best_h = h.copy()

    # Try the 2-point solution [2/3, 1/3]
    h_2pt = np.zeros(N)
    h_2pt[N//2] = 2.0/3.0
    h_2pt[N//2 + 1] = 1.0/3.0
    sc = compute_ratio(h_2pt)
    update_best(h_2pt, sc)
    
    # Try [1/3, 2/3]
    h_2pt2 = np.zeros(N)
    h_2pt2[N//2] = 1.0/3.0
    h_2pt2[N//2 + 1] = 2.0/3.0
    sc = compute_ratio(h_2pt2)
    update_best(h_2pt2, sc)
    
    # Try 3-point distributions
    # h = [a, b, a] centered?
    # Let's do a random search with local refinement.
    
    np.random.seed(42)
    
    def local_optimize(h_init):
        h = h_init.copy()
        h = h / np.sum(h)
        
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                # Try moving mass from i to j
                for j in range(N):
                    if i == j:
                        continue
                    # Move small amount from i to j
                    # Simple coordinate ascent
                    # Let h[i] -> h[i] - delta, h[j] -> h[j] + delta
                    # Find delta that maximizes ratio
                    # This is expensive. Instead, try fixed small deltas.
                    pass
            
            # Simpler: Perturb and normalize
            h_new = h.copy()
            idx = np.random.randint(N)
            h_new[idx] += 0.01
            h_new = np.maximum(h_new, 0)
            h_new = h_new / np.sum(h_new)
            sc_new = compute_ratio(h_new)
            sc_old = compute_ratio(h)
            if sc_new > sc_old:
                h = h_new
                improved = True
            
            if not improved:
                break
                
        return h, compute_ratio(h)

    # Start from best candidate
    h_start = best_h.copy()
    h_opt, sc_opt = local_optimize(h_start)
    update_best(h_opt, sc_opt)
    
    # Try multiple random starts
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        h_rand = np.random.dirichlet(np.ones(N))
        h_opt, sc_opt = local_optimize(h_rand)
        update_best(h_opt, sc_opt)
        
    return list(best_h)
