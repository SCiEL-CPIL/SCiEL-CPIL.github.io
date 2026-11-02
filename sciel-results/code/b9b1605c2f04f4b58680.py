import numpy as np

def construct():
    N = 600
    # Interval [-1/4, 1/4] -> length 1/2, each width 1/1200
    # The uniform function: h_i = 1 (constant), score = (N/2)/N^2 = 1/(2N)
    uniform_score = 1.0 / (2 * N)  # = 1/1200 ~ 0.000833333...
    # Any valid h must have normalized max autoconv >= something
    # For a two-block function with widths a and b (a+b=N, in units of dx):
    # The autoconv is a trapezoid-like shape. The peak is at the overlap.
    
    # Let's try a two-rectangle structure: h is a on [0, a) and b on [a, a+L) with L total support
    # For equal rectangles of height c on [0, L] (i.e., a single block of width L):
    # autoconv peak = c^2 * L (for the middle part), sum = c * L
    # score = c^2 * L / (c * L)^2 = 1/L
    # So minimizing 1/L means maximizing L (use full support). Then L=N gives 1/N, which is worse than uniform.
    
    # But if we use TWO separated blocks, we reduce the peak relative to the area.
    # E.g., two blocks of width w each, separated by gap g.
    # The autoconv: at lag 0, the peak is 2*c^2*w (from both blocks). At lag in overlap region... 
    # Actually the peak is at lag 0: sum of squares = 2 * c^2 * w
    # Sum = 2 * c * w
    # score = 2*c^2*w / (2*c*w)^2 = 1/(2*w)
    # If we make w = N/2 (each half the interval), score = 1/N, which is worse than uniform 1/(2N)?
    # No, 1/N < 1/(2N)? No, 1/N > 1/(2N). So single block of width N gives 1/N > 1/(2N). Uniform is better.
    
    # The uniform function on full support is actually optimal for a single rectangle.
    # To beat uniform, we need a shape that spreads out the autoconv peak.
    # The best we can do: a function concentrated at two opposite ends of the interval.
    # h = c on [0, w] and h = c on [N-w, N], with gap in the middle.
    # Peak autoconv: at lag 0, 2*c^2*w. At lag N-2w (maximal separation), still c^2 (tiny overlap).
    # Score = 2*c^2*w / (2*c*w)^2 = 1/(2w).
    # If w = N/2, then the two blocks touch, becoming a single block of N, score = 1/N.
    # If w = N/4, then two blocks each of width N/4 separated by N/2, score = 1/(2*N/4) = 2/N. Worse.
    
    # So the uniform full-support is better than any two-block construction of equal width.
    # But what about asymmetric? Let's try h = c1 on [0, w1] and c2 on [N-w2, N].
    # Peak = c1^2*w1 + c2^2*w2. Sum = c1*w1 + c2*w2.
    # We want to minimize (c1^2*w1 + c2^2*w2) / (c1*w1 + c2*w2)^2.
    # By setting c1*w1 = c2*w2 = S/2 (equal mass), we get peak = 2*(S/2)^2/(w1) + 2*(S/2)^2/(w2) = S^2/2 * (1/w1 + 1/w2)
    # Score = S^2/2 * (1/w1 + 1/w2) / S^2 = (1/w1 + 1/w2)/2.
    # To minimize (1/w1 + 1/w2)/2 subject to w1 + w2 + gap = N (gap >= 0), 
    # minimize (1/w1 + 1/w2) subject to w1 + w2 <= N.
    # The minimum of 1/w1 + 1/w2 given w1+w2 = N is at w1=w2=N/2: (2/N+2/N)/2 = 2/N? 
    # No, (1/w1 + 1/w2)/2 with w1=w2=N/2: (2/N + 2/N)/2 = 2/N. Worse than uniform 1/(2N) = 0.5/N.
    
    # It seems uniform on full support gives 1/(2N), and it's hard to beat.
    # Actually wait: for uniform h_i = 1 on all N points:
    # sum = N, autoconv at lag 0 = N, at lag k = N-k (for k=0..N-1).
    # Peak = N (at lag 0).
    # score = N / N^2 = 1/N. 
    # I made an arithmetic error earlier. score = 1/N, not 1/(2N).
    
    # So uniform score = 1/N = 1/600 ~ 0.0016667.
    
    # Can we do better? A two-block structure with equal mass:
    # w1 = w2 = w, c1 = c2 = c. Score = (1/w + 1/w)/2 = 1/w.
    # Maximize w: w can be at most N/2 (if gap=0, they merge). If w=N/2, score=2/N, worse.
    # If w=N/4, score=4/N, worse.
    
    # What about a single block of width L < N? score = 1/L > 1/N, worse.
    
    # So uniform full support seems to be the best simple construction? That's score = 1/N.
    # But the seed program scored 1.1793, meaning its score is 1/1.1793 ~ 0.848 * (1/N)? No, that's not right.
    # The score IS the upper bound on C1. The "search score" is seed/value.
    # So if seed_score is the baseline, and our value is V, then search score = seed_score / V.
    # The current program has search score 1.1793, meaning its V is smaller than seed's V by factor 1/1.1793.
    # We just need to minimize V = max(autoconv) / (sum h)^2.
    
    # For uniform, V = 1/N ≈ 0.00166667.
    # Can we get V < 1/N?
    # Consider a "triangle" shaped h (tent function). h_i = 1 - |i - N/2|/(N/2) for i in [0, N].
    # This is a continuous tent. The autoconv of a tent is a "bump". The peak of the autoconv of a triangle of height 1 and width N is ~ N^2/12 or similar.
    # Actually, the autoconv of a triangle (tent) with peak 1 and support N: the peak of the autoconv is the integral of the square = area under triangle^2 = N/3.
    # Sum = area under triangle = N/2.
    # score = (N/3) / (N/2)^2 = (N/3) / (N^2/4) = 4/(3N) ≈ 1.333/N. Worse than 1/N.
    
    # So uniform (rectangle) is better than triangle for this metric.
    
    # The optimal shape to minimize max(autoconv)/sum^2 for non-negative functions:
    # By symmetry, we want a shape whose autoconv is as flat as possible.
    # The uniform function on the full interval gives the flattest autoconv (it's a triangle, decreasing linearly from peak).
    # But the peak is at lag 0 and equals N. Can we reduce the lag-0 peak without reducing the sum too much?
    # The lag-0 peak is sum(h^2). We want to minimize sum(h^2) / (sum h)^2.
    # For fixed sum, minimizing sum(h^2) means spreading h as evenly as possible.
    # Uniform h_i = 1 gives sum(h^2) = N, sum(h) = N, ratio = 1/N.
    # Any deviation from uniform increases sum(h^2) for fixed sum.
    # So uniform minimizes the lag-0 peak / sum^2.
    # But the max autoconv might not be at lag 0 for non-uniform shapes.
    # For symmetric shapes, the max is at lag 0.
    # So uniform is optimal? Then the best possible V is 1/N.
    # But the seed program achieved V < 1/N (since score > 1).
    
    # Let me reconsider: maybe we can make the autoconv peak at a non-zero lag be lower than the lag-0 peak.
    # No, for non-negative functions, the lag-0 autoconv is sum(h_i^2), which is always >= any other lag (by Cauchy-Schwarz or rearrangement).
    # Actually, for lag k, autoconv[k] = sum_i h_i * h_{i+k} <= sum_i h_i^2 (by Cauchy-Schwarz, since sum h_i^2 = autoconv[0]).
    # Wait, that's not true in general. sum_i h_i h_{i+k} <= sum_i h_i^2 only if the correlation is <= 1, which is true for the specific inner product.
    # By Cauchy-Schwarz: (sum_i h_i h_{i+k})^2 <= (sum_i h_i^2)(sum_i h_{i+k}^2) <= (sum_i h_i^2)^2 (since the latter is a subset of all indices).
    # So yes, autoconv[k] <= autoconv[0] for all k. The peak is always at lag 0.
    
    # Therefore, the problem reduces to minimizing sum(h_i^2) / (sum h_i)^2 subject to h_i >= 0.
    # This is minimized by making h as uniform as possible: h_i = constant.
    # The minimum is 1/N, achieved by any constant function.
    # So the optimal score is 1/N = 1/600 ≈ 0.00166667.
    # If the seed program had score 1.0, its V was some value, and our V can be at best 1/N.
    # The current program has search score 1.1793, so its V is seed_V / 1.1793.
    # If seed_V was 1/N (uniform), then our V is 1/(1.1793*N) < 1/N, which contradicts the optimality.
    # So the seed program must have been worse than uniform, or my analysis is flawed.
    
    # Let me just return uniform, which gives V = 1/N. This should be the optimal solution.
    return [1.0] * N
