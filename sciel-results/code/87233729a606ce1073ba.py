"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct step heights that minimize the autocorrelation metric.
    
    The metric is: value = abs(2 * N * max(convolve(h, h)) / sum(h)^2)
    
    For the seed (all ones), convolve gives [1, 2, 3, ..., 400, ..., 3, 2, 1],
    max = 400, sum = 400, so value = abs(2 * 400 * 400 / 400^2) = 2.0.
    
    We want to minimize this. Key insight: if we make h have both positive and
    negative values, the autoconvolution's peak (at lag 0) is sum(h^2), but
    sum(h) can be made small, potentially making the ratio large. However,
    we want the RATIO small.
    
    Let's think differently. The metric is:
    value = abs(2 * N * max_k(c_k) / (sum h)^2)
    where c = convolve(h, h).
    
    Note that c_0 = sum(h_i^2) by Parseval/definition of full convolution at lag 0.
    Actually for full convolution, c_k = sum_i h_i * h_{i+k} for valid k.
    The maximum over all signed entries of the full convolution.
    
    For a signal with zero mean (sum h = 0), the denominator is 0, which is invalid.
    So we need sum(h) != 0.
    
    Strategy: We want max(c_k) to be small relative to (sum h)^2.
    
    If h is a window function like a raised cosine or similar, the autocorrelation
    will have a main lobe and sidelobes. The peak is at lag 0 = sum(h^2).
    
    value = 2*N*sum(h^2) / (sum h)^2 if the max is at lag 0.
    
    By Cauchy-Schwarz, (sum h)^2 <= N * sum(h^2), so sum(h^2)/(sum h)^2 >= 1/N.
    Thus value >= 2*N*(1/N) = 2. Equality when all h_i are equal.
    
    So the minimum possible value is 2, achieved by constant sequences!
    Wait, but the task says "lower is better" and seed gives 2.0.
    
    Let me re-read: "The evaluator computes exactly this value, which is an upper bound
    on the constant C3, and lower is better"
    
    And "The seed np.ones(400) gives 2.0."
    
    So if 2 is the theoretical minimum, we can't do better than 2? But the task implies
    we can improve. Let me re-examine.
    
    Oh wait, the max is taken over SIGNED entries. So if the convolution has negative
    values, the max (largest signed value) might not be at lag 0. If we can make
    the lag-0 value (which is sum(h^2), always positive) NOT be the maximum by having
    some other lag have a larger positive value... but that's impossible since lag 0
    is the sum of squares.
    
    Actually no: for real sequences, c_0 = sum(h_i^2) >= 0, and by Cauchy-Schwarz
    |c_k| <= sqrt(sum(h_i^2) * sum(h_{i+k}^2)) <= sum(h_i^2) = c_0.
    
    So c_0 is always the maximum in absolute value, and since c_0 > 0, it's the max
    signed value too (unless all other values are also positive and larger, which
    can't happen).
    
    Wait, that's not right. c_k can be positive for k != 0. But |c_k| <= c_0 by
    Cauchy-Schwarz. So c_k <= c_0 for all k. Thus max(c_k) = c_0 = sum(h^2).
    
    So value = 2*N*sum(h^2) / (sum h)^2.
    
    And by Cauchy-Schwarz, this is >= 2, with equality iff h is constant.
    
    So the minimum is exactly 2.0, and we can't do better!
    
    But wait, the problem says "an upper bound on the constant C3" and asks to minimize.
    If 2 is the lower bound, then the best we can do is 2.0, which is what the seed
    already achieves.
    
    Let me double-check with a different sequence. Say h = [1, -1, 1, -1, ...]
    sum(h) = 0, invalid.
    
    Say h = [1, 0, 1, 0, ...] (alternating 1 and 0)
    sum(h) = 200, sum(h^2) = 200
    value = 2*400*200 / 200^2 = 160000/40000 = 4.0. Worse.
    
    Say h = [1, 1, ..., 1, -1, -1, ..., -1] with 200 ones and 200 minus ones.
    sum(h) = 0, invalid.
    
    Say 201 ones and 199 minus ones.
    sum(h) = 2, sum(h^2) = 400
    value = 2*400*400 / 4 = 80000/4 = 20000. Much worse.
    
    So indeed, the constant sequence minimizes the value at 2.0.
    
    Therefore, the optimal solution is just np.ones(400) or any constant sequence.
    The score is seed/value = 2/2 = 1.0, and we can't beat that.
    
    But the task says "above 1.0 is an improvement" which implies it's possible.
    Let me re-read the metric more carefully.
    
    "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    
    np.convolve(h, h) with mode='full' gives length 2N-1 = 799.
    The entries are c_k for k = 0, 1, ..., 798.
    c_k = sum_{i} h[i] * h[i+k] where the indices are valid.
    
    For k=0: c_0 = sum h[i]^2
    For k>0: c_k = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    For k<0: not present in full convolution; full convolution has c_k for k from 0 to 2N-2
    where c_k = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i] ... 
    
    Actually, np.convolve(a, b, 'full') gives result[k] = sum_i a[i]*b[k-i] for
    k = 0, ..., len(a)+len(b)-2.
    
    For a=b=h, result[k] = sum_i h[i]*h[k-i] where both indices are in [0, N-1].
    
    So result[0] = h[0]*h[0] = h[0]^2? No wait.
    
    result[k] = sum_{i} h[i] * h[k-i] where 0<=i<N and 0<=k-i<N.
    
    For k=0: only i=0, so result[0] = h[0]*h[0] = h[0]^2.
    
    Hmm, that doesn't seem right for autocorrelation. Let me think again.
    
    Actually np.convolve([1,2], [1,2], 'full') = [1, 4, 4, 4, 1]? No.
    [1,2] * [1,2] = [1, 2+2, 4] = [1, 4, 4] for 'full'? 
    Convolution: [1,2] convolved with [1,2]:
    result[0] = 1*1 = 1
    result[1] = 1*2 + 2*1 = 4
    result[2] = 2*2 = 4
    So [1, 4, 4].
    
    But autocorrelation of [1,2] should be [1, 2, 1] if we're talking about
    sum h[i]h[i+k].
    
    So np.convolve is NOT autocorrelation! It's convolution.
    
    For autocorrelation, we'd use np.correlate(h, h, 'full') or similar.
    
    But the task says np.convolve(h, h). So it's convolution, not correlation.
    
    For convolution: c_k = sum_i h[i] * h[k-i]
    
    For a constant sequence h[i] = 1 for all i:
    c_k = number of i such that 0<=i<N and 0<=k-i<N
    = number of i in [max(0, k-N+1), min(N-1, k)]
    
    For k=0: i=0, so c_0 = 1
    For k=1: i=0,1, so c_1 = 2
    ...
    For k=N-1=399: c_399 = N = 400
    For k=N=400: c_400 = N-1 = 399
    ...
    For k=2N-2=798: c_798 = 1
    
    So the max is 400 at k=399 (and symmetrically).
    sum(h) = 400
    value = abs(2*400*400 / 400^2) = 2.0. Confirmed.
    
    Now, can we get value < 2?
    
    value = 2*N*max(c_k) / (sum h)^2
    
    We want this small. Let's denote S = sum(h), M = max(c_k).
    
    For the constant sequence, S=N, M=N, value=2.
    
    Can we make M/S^2 smaller?
    
    Note that sum_k c_k = (sum h)^2 = S^2. This is because
    sum_k sum_i h[i]h[k-i] = sum_i sum_k h[i]h[k-i] = sum_i h[i] * sum_j h[j] = S^2.
    
    So the sum of all c_k is S^2. There are 2N-1 terms.
    So the average c_k is S^2/(2N-1).
    
    M >= S^2/(2N-1), so value >= 2*N*S^2/(2N-1)/S^2 = 2N/(2N-1) ≈ 1.
    
    So the lower bound is about 1, not 2!
    
    To achieve this, we'd want all c_k to be equal, which means the convolution
    is a constant. This is related to Barker sequences or CAZAC sequences.
    
    For a sequence whose convolution with itself is constant (a "perfect" autocorrelation
    in the convolution sense), we'd get value close to 2N/(2N-1).
    
    Zadoff-Chu sequences have this property in the correlation sense. Let me think
    about whether we can use them.
    
    A Zadoff-Chu sequence z_k = exp(-j*pi*k*(k+1)/N) for k=0,...,N-1.
    Its circular autocorrelation is perfect. But we need linear convolution.
    
    Let me try a different approach: use a window that spreads the energy
    more evenly across lags.
    
    Actually, let's try h[i] = exp(j*2*pi*f*i/N) for some frequency f, then take
    the real part. Or use a complex sequence and take real part.
    
    Let me try h[i] = cos(2*pi*i/M) for some M.
    
    For a cosine, the convolution will have structure. Let me just try several
    candidate sequences and compute the value.
    
    Given the constraint of 120 seconds, I can do a small grid search.
    """
    # Try several candidate sequences
    candidates = []
    
    # 1. Constant (seed)
    candidates.append(np.ones(N))
    
    # 2. Cosine with various frequencies
    for freq in [1, 2, 3, 5, 7, 10, 20, 50, 100]:
        t = np.arange(N)
        candidates.append(np.cos(2 * np.pi * freq * t / N))
    
    # 3. Sine
    for freq in [1, 2, 3, 5, 7, 10, 20, 50, 100]:
        t = np.arange(N)
        candidates.append(np.sin(2 * np.pi * freq * t / N))
    
    # 4. Zadoff-Chu real part
    for root in [1, 2, 3, 5, 7, 11, 13]:
        t = np.arange(N)
        zc = np.exp(-1j * np.pi * root * t * (t + 1) / N)
        candidates.append(np.real(zc))
        candidates.append(np.imag(zc))
    
    # 5. Raised cosine window
    for width in [0.5, 0.8, 1.0]:
        t = np.linspace(-1, 1, N)
        candidates.append(np.cos(np.pi * t / width) ** 2)
    
    # 6. Gaussian-like
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        t = np.linspace(-1, 1, N)
        candidates.append(np.exp(-t**2 / (2*sigma**2)))
    
    # Evaluate all candidates
    best_h = None
    best_val = float('inf')
    
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / s**2)
        if val < best_val:
            best_val = val
            best_h = h
    
    return best_h if best_h is not None else np.ones(N)
# EVOLVE-BLOCK-END
