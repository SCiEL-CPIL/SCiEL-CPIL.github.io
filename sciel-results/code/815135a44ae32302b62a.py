import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # The value to minimize is:
    # V(h) = | 2 * N * max_k (h*h)_k | / (sum(h))^2
    #
    # Let h be the step heights.
    # Let H = FFT(h).
    # The autoconvolution c = convolve(h, h) is the inverse FFT of |H|^2.
    # Specifically, if we use standard FFT conventions:
    # c_full = ifft(fft(h, 2N-1) * conj(fft(h, 2N-1)))
    # Wait, convolve(h, h) in numpy is linear convolution.
    # Let's verify the relationship.
    # c_k = sum_i h_i h_{i-k} (for valid k in full conv range)
    # This is the autocorrelation.
    # The maximum of the autocorrelation is at k=0, which is sum(h_i^2).
    # However, the problem states "maximum is taken over its signed entries".
    # Since h can be negative, h_i h_j can be negative.
    # But (h*h)_0 = sum h_i^2 is always the largest positive value?
    # Not necessarily if h has large negative parts? No, sum h_i^2 is the L2 norm squared.
    # Any other lag k != 0 has magnitude bounded by sum h_i^2 (Cauchy-Schwarz).
    # So max(c) is always c[0] = sum(h^2) if we assume the "max" picks the largest algebraic value.
    # Wait, np.max returns the largest algebraic value.
    # Is it possible for a negative lag to be larger than c[0]?
    # c[0] = sum h_i^2.
    # |c[k]| <= sqrt(sum h_i^2 * sum h_j^2) = sum h_i^2.
    # So c[0] is the global maximum in absolute value.
    # Since c[0] >= 0, max(c) = c[0] = sum(h^2).
    # 
    # Let's double check this assumption.
    # If h = [1, -1], conv = [1, -2, 1]. max = 1. sum(h^2) = 2.
    # Wait. conv([1, -1], [1, -1]) = [1, -2, 1].
    # sum(h^2) = 1^2 + (-1)^2 = 2.
    # But max(conv) is 1.
    # My Cauchy-Schwarz bound was on |c[k]|.
    # c[0] = sum h_i^2.
    # c[k] = sum h_i h_{i-k}.
    # For h=[1, -1], c[0] = 1+1=2.
    # Why did I say conv is [1, -2, 1]?
    # np.convolve([1, -1], [1, -1]) 
    # = 1*1 + 1*(-1) + (-1)*(-1) ... no.
    # Full conv:
    # k=0: h[0]*h[0] = 1
    # k=1: h[0]*h[1] + h[1]*h[0] = -1 -1 = -2
    # k=2: h[1]*h[1] = 1
    # So conv is [1, -2, 1].
    # sum(h^2) = 2.
    # But c[0] in the standard correlation definition (lag 0) is sum h_i^2.
    # In np.convolve, the center index corresponds to lag 0.
    # For N=2, len(conv)=3. Center is index 1.
    # conv[1] = -2.
    # conv[0] = 1, conv[2] = 1.
    # So max(conv) = 1.
    # sum(h^2) = 2.
    # So max(conv) is NOT sum(h^2) in general.
    #
    # Let's re-evaluate.
    # c[k] = sum_i h_i h_{i-k}.
    # The "center" of the convolution (lag 0) is at index N-1 in the full array of length 2N-1.
    # c[N-1] = sum_i h_i h_i = sum h_i^2.
    # So the value at the center is sum h_i^2.
    # Is it the maximum?
    # For h=[1, -1], center index is 1. c[1] = -2.
    # c[0] = 1, c[2] = 1.
    # Max is 1.
    # So max(c) can be less than sum(h^2) if the center value is negative?
    # Wait, sum(h^2) is always positive.
    # Why is c[1] = -2 for h=[1, -1]?
    # c[1] = h[0]h[1] + h[1]h[0] = -1 -1 = -2.
    # Ah, the formula c[k] = sum h_i h_{i-k} assumes h is zero-padded outside [0, N-1].
    # For k=0 (center), sum h_i h_i = sum h_i^2.
    # For h=[1, -1], sum h_i^2 = 2.
    # But np.convolve gives -2 at the center?
    # Let's check np.convolve definition.
    # np.convolve(a, v, mode='full')
    # It computes the convolution.
    # a=[1, -1], v=[1, -1].
    # Result: [1, -2, 1].
    # The "center" of a length 3 array is index 1. Value -2.
    # But sum(h^2) = 2.
    # There is a sign convention or index shift issue.
    # Actually, autocorrelation is usually defined as sum h_i h_{i-k}.
    # Convolution is sum h_i v_{k-i}.
    # If v=h, conv[k] = sum h_i h_{k-i}.
    # Let j = k-i => i = k-j.
    # conv[k] = sum h_{k-j} h_j.
    # This is the same as autocorrelation if we index correctly.
    # For k=N-1 (center for N=2 is 1):
    # conv[1] = sum_{j} h_{1-j} h_j.
    # j=0: h[1]h[0] = -1
    # j=1: h[0]h[1] = -1
    # Sum = -2.
    # But sum h_i^2 = 2.
    # Why the discrepancy?
    # Ah, h[1]h[0] + h[0]h[1] = 2 h[0]h[1].
    # The term h_i h_i is not present in the cross terms?
    # No, for k=N-1, the indices i and k-i must both be in [0, N-1].
    # For N=2, k=1.
    # i=0, k-i=1. h[0]h[1].
    # i=1, k-i=0. h[1]h[0].
    # The term h[0]h[0] and h[1]h[1] are NOT included in conv[1]?
    # They are in conv[0] and conv[2].
    # conv[0] = h[0]h[0] = 1.
    # conv[2] = h[1]h[1] = 1.
    # So the "peak" of the autocorrelation (lag 0) is split?
    # No, standard autocorrelation R[k] = sum h[n] h[n-k].
    # R[0] = sum h[n]^2.
    # In np.convolve, the lag 0 term is at index N-1 ONLY IF we are doing correlation?
    # np.correlate(h, h, 'full') would give the autocorrelation.
    # np.convolve(h, h, 'full') is convolution.
    # For real signals, convolution is symmetric.
    # But the "lag 0" of convolution is the sum of products of overlapping elements when one is shifted by 0 relative to the other?
    # If we align h and h with no shift (index 0 of h1 with index 0 of h2), we get sum h_i^2.
    # In np.convolve, the output index k corresponds to the shift where the first element of the second array aligns with the (k-(N-1))-th element of the first array?
    # Actually, np.convolve(a, v) output index k is sum a[i] v[k-i].
    # If we want the "overlap" at shift 0 (a[0] with v[0]), that is not the center.
    # For a=[1, -1], v=[1, -1].
    # Overlap at shift 0: 1*1 + (-1)*(-1) = 2.
    # Where is this in the conv output?
    # conv[0] = a[0]v[0] = 1.
    # conv[1] = a[0]v[1] + a[1]v[0] = -2.
    # conv[2] = a[1]v[1] = 1.
    # The value 2 is NOT in the convolution output!
    # This is because np.convolve does NOT compute the sum of squares at the center.
    # It computes the cross-correlation-like terms but with a specific indexing.
    # Wait, if a=v, conv is symmetric.
    # But R[0] = sum a[i]^2.
    # Is R[0] equal to any element of conv?
    # For N=1, a=[1]. conv=[1]. R[0]=1. Match.
    # For N=2, a=[1, 0]. conv=[1, 0, 0]. R[0]=1. conv[0]=1.
    # For N=2, a=[1, 1]. conv=[1, 2, 1]. R[0]=2. conv[1]=2.
    # For N=2, a=[1, -1]. conv=[1, -2, 1]. R[0]=2. conv[1]=-2.
    # So conv[N-1] = sum a[i] a[i-(N-1-i)]?
    # It seems conv[N-1] is NOT sum a[i]^2.
    # 
    # Let's check the definition of the value in the task:
    # value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    # 
    # If h = ones(400).
    # conv(ones, ones) is a triangle.
    # Max is at the center.
    # Center value is N = 400.
    # sum(h) = 400.
    # value = abs(2 * 400 * 400 / 160000) = 320000 / 160000 = 2.0.
    # This matches the seed.
    #
    # So for h=ones, max(conv) = N.
    # For h=[1, -1] (N=2), max(conv) = 1.
    # sum(h) = 0. (Invalid).
    #
    # Let's try h=[1, 0]. N=2.
    # conv=[1, 0, 0]. max=1.
    # sum=1.
    # value = abs(2*2*1 / 1) = 4.
    #
    # Let's try h=[1, 1]. N=2.
    # conv=[1, 2, 1]. max=2.
    # sum=2.
    # value = abs(2*2*2 / 4) = 2.
    #
    # Let's try h=[1, -0.5]. N=2.
    # conv:
    # 0: 1*1 = 1
    # 1: 1*(-0.5) + (-0.5)*1 = -1
    # 2: (-0.5)*(-0.5) = 0.25
    # max = 1.
    # sum = 0.5.
    # value = abs(2*2*1 / 0.25) = 16.
    #
    # It seems we want to minimize the max of the convolution.
    # 
    # Key Insight:
    # The max of the convolution is at least the maximum of the "diagonal" terms?
    # No, for h=[1, -1], the diagonal terms (squares) are 1 and 1. The cross term is -2.
    # Max is 1.
    # 
    # If we can make the center term (which is often large) negative or small, and keep the sum non-zero, we might win.
    # But the max is taken over ALL entries.
    # 
    # Strategy:
    # We want to minimize M = max_k conv_k.
    # Subject to S = sum h != 0.
    # Value = 2 N M / S^2.
    # 
    # Note that conv_k is a quadratic form in h.
    # The function M(h) is convex (max of linear/quadratic forms? No, max of quadratics is not convex).
    # However, we can use a different approach.
    # 
    # Let's look at the frequency domain again.
    # conv(h, h) = IFFT(FFT(h) * FFT(h)). (With appropriate zero padding).
    # Let H = FFT(h, 2N-1).
    # c = IFFT(H * H).
    # We want to minimize max(c).
    # 
    # This is equivalent to: Find h such that the inverse Fourier transform of H^2 has a small maximum.
    # 
    # Consider h as a vector.
    # c_k = sum_i h_i h_{i-k}.
    # 
    # If we choose h to be a "random" vector, c_k behaves like a random variable with mean 0 (for k != 0) and variance proportional to sum h^4.
    # The max of many Gaussians grows as sqrt(log N).
    # 
    # The current best is 1.1014.
    # Seed is 2.0.
    # 
    # Let's try to minimize the L-infinity norm of the autoconvolution.
    # 
    # We can use a gradient-free method like Nelder-Mead or a custom local search, but the dimension is 400.
    # 
    # Alternative:
    # Use the fact that if we scale h by alpha, M scales by alpha^2 and S^2 scales by alpha^2.
    # So the value is scale-invariant.
    # We can fix sum(h) = 1.
    # Then minimize max(conv(h, h)).
    # 
    # Let's try a structured signal: h_i = cos(2 pi k i / N) + small noise?
    # Or h_i = (-1)^i?
    # If h_i = (-1)^i, sum is 0.
    # If h_i = (-1)^i + eps.
    # 
    # Let's try a "boxcar" with a hole?
    # 
    # I will implement a simple local search that uses the gradient of the max term.
    # The gradient of max(c) with respect to h is the gradient of the specific c_k that is the max.
    # grad_h c_k = 2 * (shifted h).
    # 
    # We can do projected gradient descent on the constraint sum(h)=1.
    # 
    # Objective: f(h) = max_k c_k.
    # Constraint: sum(h) = 1.
    # 
    # Algorithm:
    # 1. Initialize h (e.g., random or ones).
    # 2. Normalize h so sum(h)=1.
    # 3. Compute c = convolve(h, h).
    # 4. Find k_max = argmax(c).
    # 5. Compute gradient g = grad_h c_{k_max}.
    # 6. Update h = h - lr * g.
    # 7. Project h onto sum(h)=1: h = h - (sum(h)-1)/N.
    # 8. Repeat.
    # 
    # This minimizes the single largest peak.
    # 
    # Let's refine the gradient.
    # c_k = sum_i h_i h_{i-k}.
    # dc_k / dh_j = h_{j-k} + h_{j+k} (with boundary handling).
    # 
    # This is effectively 2 * (h correlated with a delta at k).
    # 
    # Let's implement this.

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    rng = np.random.default_rng(42)
    
    # Gradient Descent on max(autoconv)
    # We fix sum(h) = 1 to handle scale invariance.
    # Then value = 2 * N * max(c).
    
    def normalize_sum(h, target=1.0):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h
        return h * (target / s)
    
    def grad_max_conv(h):
        c = np.convolve(h, h)
        k_max = np.argmax(c)
        # Gradient of c_{k_max} w.r.t h
        # c_k = sum_i h_i h_{i-k}
        # d c_k / d h_j = h_{j-k} + h_{j+k}
        # Note: indices must be valid.
        # We can compute this via convolution with a shifted delta?
        # Actually, grad = convolve(h, delta_k) + convolve(h, delta_{-k})?
        # Let's just use the formula directly or via FFT.
        
        # FFT approach for gradient:
        # c = IFFT(H^2)
        # grad_h c_k = 2 * IFFT(H * exp(-i 2 pi k j / M))?
        # Let's stick to time domain for simplicity and correctness, N=400 is small.
        
        # k_max is in range [0, 2N-2]
        # h is defined on [0, N-1]
        
        grad = np.zeros(N)
        # Term 1: h_{j-k}
        # If 0 <= j-k < N, grad[j] += h[j-k]
        # Term 2: h_{j+k}
        # If 0 <= j+k < N, grad[j] += h[j+k]
        
        # Efficiently:
        # Shift h by k_max and add to grad (where valid)
        # Shift h by -k_max and add to grad (where valid)
        
        # This is equivalent to:
        # grad = np.convolve(h, np.zeros(2N-1)) ... no.
        
        # Let's just loop or use slicing.
        k = k_max
        
        # h[j-k] is non-zero if j >= k and j-k < N => j in [k, k+N-1]
        # But j is in [0, N-1].
        # So j in [max(0, k), N-1]
        # h[j+k] is non-zero if j+k < N => j < N-k
        # So j in [0, N-k-1]
        
        if k < N:
            # h[j-k] part: j from k to N-1
            # h[j-k] for j in [k, N-1] is h[0:N-k]
            grad[k:N] += h[0:N-k]
            
            # h[j+k] part: j from 0 to N-k-1
            # h[j+k] for j in [0, N-k-1] is h[k:N]
            grad[0:N-k] += h[k:N]
        else:
            # k >= N
            # h[j-k]: j-k < 0 always for j < N. So 0.
            # h[j+k]: j+k < N => j < N-k. Since k >= N, N-k <= 0. So 0.
            # Wait, if k >= N, the shifts are out of bounds for the support of h?
            # c_k = sum h_i h_{i-k}.
            # If k >= N, then i-k >= N-i-k?
            # Let's check N=2, k=2.
            # c_2 = h_1 h_1.
            # d c_2 / d h_1 = 2 h_1.
            # d c_2 / d h_0 = 0.
            # Formula:
            # h[j-k]: j=1, 1-2=-1 (0).
            # h[j+k]: j=1, 1+2=3 (0).
            # My slicing logic failed for k>=N.
            
            # Let's use a general convolution trick.
            # grad_h c_k = 2 * (h * delta_k)_{j} ? No.
            # c_k = <h, shift_k(h)>.
            # grad = shift_{-k}(h) + shift_k(h).
            
            pass # Handled below with general method
            
        # General method using np.roll or padding
        # Let's just use the FFT derivative which is robust.
        # c = IFFT(H^2)
        # c_k = (1/M) sum_j H_j^2 exp(i 2 pi j k / M)
        # This is getting complicated with linear vs circular.
        
        # Let's stick to the slicing but fix the k>=N case.
        # If k >= N:
        # h[j-k] is 0 for all j in [0, N-1] because j-k < 0.
        # h[j+k] is 0 for all j in [0, N-1] because j+k >= k >= N.
        # So grad is 0?
        # Check N=2, k=2. c_2 = h_1^2.
        # d c_2 / d h_1 = 2 h_1.
        # My formula said 0.
        # Why?
        # c_k = sum_i h_i h_{i-k}.
        # For k=2, i must be such that i-2 is in [0,1].
        # i=2? No, i is in [0,1].
        # So i-2 is in [-2, -1].
        # h_{i-2} is 0.
        # So c_2 = 0?
        # But np.convolve([1, -1], [1, -1]) = [1, -2, 1].
        # c_2 = 1.
        # h_1 h_1 = (-1)(-1) = 1.
        # So the term is h_1 h_1.
        # In the sum sum_i h_i h_{i-k}, if we define h_n=0 for n<0 or n>=N:
        # i=0: h_0 h_{-2} = 0.
        # i=1: h_1 h_{-1} = 0.
        # So c_2 should be 0?
        # 
        # There is a mismatch between np.convolve and the formula sum h_i h_{i-k}.
        # np.convolve(a, v) computes sum a_i v_{k-i}.
        # Let v=h.
        # c_k = sum_i h_i h_{k-i}.
        # For k=2, N=2:
        # i=0: h_0 h_2 = 0.
        # i=1: h_1 h_1 = 1.
        # i=2: h_2 h_0 = 0.
        # Sum = 1.
        # 
        # So the formula is c_k = sum_i h_i h_{k-i}.
        # Gradient w.r.t h_j:
        # d c_k / d h_j = h_{k-j} + h_{j-k} (from the two terms where j is i or k-i).
        # 
        # So for k=2, j=1:
        # h_{2-1} + h_{1-2} = h_1 + h_{-1} = -1 + 0 = -1.
        # But d(h_1^2)/dh_1 = 2 h_1 = -2.
        # Factor of 2?
        # c_k = sum_i h_i h_{k-i}.
        # If k is even and h is symmetric?
        # For j=k/2, the term h_j h_j appears only once in the sum?
        # No, sum is over i.
        # If i=j and k-i=j, then i=k/2.
        # The term is h_j h_j.
        # Derivative is 2 h_j.
        # My formula h_{k-j} + h_{j-k} gives h_j + h_j = 2 h_j.
        # Correct.
        
        # So the gradient is:
        # grad[j] = h[(k-j) % N]? No, with zero padding.
        # grad[j] = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N).
        
        # Let's re-implement grad calculation correctly.
        grad = np.zeros(N)
        
        # Part 1: h[k-j]
        # Non-zero if 0 <= k-j < N => k-N < j <= k
        # j in [max(0, k-N+1), min(N-1, k)]
        
        # Part 2: h[j-k]
        # Non-zero if 0 <= j-k < N => k < j <= k+N-1
        # j in [max(0, k+1), min(N-1, k+N-1)]
        
        # We can compute these using slicing.
        
        # Part 1:
        # Indices j where 0 <= k-j < N
        # j = k - m, m in [0, N-1]
        # j in [k-N+1, k]
        # Clip to [0, N-1]
        j_start1 = max(0, k - N + 1)
        j_end1 = min(N - 1, k)
        if j_start1 <= j_end1:
            # j runs from j_start1 to j_end1
            # k-j runs from k-j_start1 down to k-j_end1
            # h[k-j] for j in [j_start1, j_end1]
            # This is h[k-j_end1 : k-j_start1 + 1] reversed?
            # Let m = k-j.
            # j = j_start1 => m = k - j_start1
            # j = j_end1 => m = k - j_end1
            # m decreases as j increases.
            # So the slice is h[k-j_end1 : k-j_start1 + 1]
            # And we assign to grad[j_start1 : j_end1 + 1]
            # But the order is reversed.
            # grad[j_start1 : j_end1 + 1] = h[k-j_end1 : k-j_start1 + 1][::-1]
            
            # Let's verify.
            # N=2, k=2.
            # j_start1 = max(0, 2-2+1) = 1.
            # j_end1 = min(1, 2) = 1.
            # j=1.
            # h[k-j] = h[1] = -1.
            # Slice: h[2-1 : 2-1 + 1] = h[1:2] = [-1].
            # Reversed: [-1].
            # grad[1] += -1.
            
            # Part 2:
            # Indices j where 0 <= j-k < N
            # j = k + m, m in [0, N-1]
            # j in [k, k+N-1]
            # Clip to [0, N-1]
            j_start2 = max(0, k)
            j_end2 = min(N - 1, k + N - 1)
            if j_start2 <= j_end2:
                # j runs from j_start2 to j_end2
                # m = j-k runs from j_start2-k to j_end2-k
                # h[j-k] for j in [j_start2, j_end2]
                # This is h[j_start2-k : j_end2-k + 1]
                # Order is same.
                
                # N=2, k=2.
                # j_start2 = 2.
                # j_end2 = min(1, 3) = 1.
                # j_start2 > j_end2, so skip.
                
                # Wait, for k=2, j must be in [2, 3]. But j max is 1.
                # So Part 2 is empty.
                # Total grad[1] = -1.
                # But we expected 2*h_1 = -2.
                # Where is the other -1?
                # Ah, for k=2, j=1.
                # Part 1: h[2-1] = h_1 = -1.
                # Part 2: h[1-2] = h_{-1} = 0.
                # Sum = -1.
                # But derivative of h_1^2 is 2 h_1 = -2.
                # 
                # Why?
                # c_2 = h_1 h_1.
                # In the sum sum_i h_i h_{k-i}:
                # i=1: h_1 h_1.
                # The term h_1 appears twice in the product?
                # No, it's h_i h_{k-i}.
                # If i=1 and k-i=1, it's the same element.
                # The derivative of x^2 is 2x.
                # My formula h[k-j] + h[j-k] gives h_1 + h_{-1} = -1.
                # It misses the factor of 2 for the diagonal term?
                # 
                # Let's check the derivative of sum_i h_i h_{k-i} w.r.t h_j.
                # The term involving h_j is:
                # 1. When i=j: h_j h_{k-j}.
                # 2. When k-i=j => i=k-j: h_{k-j} h_j.
                # 
                # If j != k-j (i.e., 2j != k), these are two distinct terms in the sum.
                # Derivative is h_{k-j} + h_{k-j} = 2 h_{k-j}.
                # 
                # If j == k-j (i.e., 2j == k), these are the SAME term in the sum.
                # The term is h_j h_j = h_j^2.
                # Derivative is 2 h_j.
                # 
                # So in both cases, the derivative is 2 h_{k-j} (if valid).
                # 
                # My previous formula h[k-j] + h[j-k] is actually:
                # Term 1 (i=j): h_{k-j}
                # Term 2 (i=k-j): h_{j-k} ? No, the term is h_{k-j} h_j. Deriv is h_{k-j}.
                # So it should be h_{k-j} + h_{k-j} = 2 h_{k-j}.
                # 
                # Why did I write h[j-k]?
                # Because I thought of the other shift.
                # But both terms involve h_{k-j} (or h_{j-k} which is the same index if we consider symmetry? No).
                # 
                # Let's re-evaluate.
                # c_k = sum_i h_i h_{k-i}.
                # d/dh_j (h_j h_{k-j}) = h_{k-j} (if j != k-j) or 2 h_j (if j=k-j).
                # d/dh_j (h_{k-j} h_j) = h_{k-j} (if j != k-j) or 2 h_j (if j=k-j).
                # 
                # If j != k-j, the sum has two terms: one with i=j and one with i=k-j.
                # Both contribute h_{k-j} (or h_j, which is the same value if we swap indices? No).
                # Term 1: i=j. Value h_j h_{k-j}. Deriv w.r.t h_j is h_{k-j}.
                # Term 2: i=k-j. Value h_{k-j} h_j. Deriv w.r.t h_j is h_{k-j}.
                # Total: 2 h_{k-j}.
                # 
                # If j == k-j, there is only one term i=j.
                # Value h_j^2. Deriv 2 h_j.
                # 
                # So the gradient is ALWAYS 2 * h_{k-j} (if 0 <= k-j < N) + 2 * h_{j-k} (if 0 <= j-k < N)?
                # No, h_{k-j} and h_{j-k} are the same index if we consider the array?
                # No, k-j and j-k are different indices unless k=0.
                # 
                # Wait, if k-j is valid, then the index is k-j.
                # If j-k is valid, then the index is j-k.
                # 
                # Case 1: k-j is valid (0 <= k-j < N).
                # Then the term h_j h_{k-j} exists.
                # Also, does the term h_{k-j} h_j exist?
                # This corresponds to i = k-j.
                # Is i in [0, N-1]? Yes, because k-j is valid.
                # So yes, both terms exist.
                # They are distinct unless j = k-j.
                # If distinct, sum of derivatives is h_{k-j} + h_{k-j} = 2 h_{k-j}.
                # If same, derivative is 2 h_j.
                # In both cases, it is 2 * h_{k-j}.
                # 
                # Case 2: j-k is valid (0 <= j-k < N).
                # Then the term h_j h_{j-k}? No.
                # The sum is over i.
                # The term with h_j is h_j h_{k-j}.
                # If k-j is NOT valid, this term is 0.
                # 
                # So, is it possible that k-j is invalid but j-k is valid?
                # k-j < 0 => j > k.
                # j-k < N => j < k+N.
                # So if j > k, k-j is negative (invalid).
                # But j-k is positive.
                # 
                # Does the term h_j h_{k-j} exist?
                # k-j is negative, so h_{k-j} = 0.
                # So the term is 0.
                # 
                # But wait, the sum is c_k = sum_i h_i h_{k-i}.
                # If we swap the roles of h and h in the convolution, it's symmetric.
                # 
                # Let's check the case j > k.
                # Example N=2, k=0, j=1.
                # c_0 = h_0 h_0.
                # d c_0 / d h_1 = 0.
                # Formula: 2 h_{0-1} + ... h_{-1} is 0. Correct.
                # 
                # Example N=2, k=2, j=1.
                # c_2 = h_1 h_1.
                # d c_2 / d h_1 = 2 h_1.
                # Formula:
                # k-j = 1 (valid). 2 h_1.
                # j-k = -1 (invalid).
                # Total 2 h_1. Correct.
                # 
                # Example N=3, k=1, j=2.
                # c_1 = h_0 h_1 + h_1 h_0 = 2 h_0 h_1.
                # d c_1 / d h_2 = 0.
                # Formula:
                # k-j = -1 (invalid).
                # j-k = 1 (valid).
                # Does the term with h_2 exist in c_1?
                # c_1 = sum_i h_i h_{1-i}.
                # i=0: h_0 h_1.
                # i=1: h_1 h_0.
                # i=2: h_2 h_{-1} = 0.
                # So h_2 does not appear.
                # But my "Case 2" logic suggested j-k valid means something.
                # 
                # The term is h_i h_{k-i}.
                # If i=j, term is h_j h_{k-j}.
                # If i=k-j, term is h_{k-j} h_j.
                # 
                # If j=2, k=1.
                # i=2: h_2 h_{-1} = 0.
                # i=-1: invalid.
                # So no term.
                # 
                # So the gradient is simply:
                # If 0 <= k-j < N: grad[j] += 2 * h[k-j]
                # If 0 <= j-k < N: grad[j] += 2 * h[j-k] ?
                # 
                # Let's check if j-k valid implies a term.
                # If 0 <= j-k < N, let m = j-k.
                # Then j = k+m.
                # The term with i=k+m is h_{k+m} h_{k-(k+m)} = h_{k+m} h_{-m} = 0.
                # The term with i=m is h_m h_{k-m} = h_m h_j.
                # Derivative w.r.t h_j is h_m = h[j-k].
                # 
                # So yes, if j-k is valid, there is a term h_{j-k} h_j.
                # And if k-j is valid, there is a term h_j h_{k-j}.
                # 
                # These are distinct terms unless j-k = k-j => 2j=2k => j=k.
                # 
                # So:
                # grad[j] = 2 * h[k-j] (if valid) + 2 * h[j-k] (if valid)
                # 
                # Let's re-verify N=2, k=2, j=1.
                # k-j = 1 (valid). 2 h_1.
                # j-k = -1 (invalid).
                # Total 2 h_1. Correct.
                
                # So the gradient is the sum of these two parts.
                # We can compute this efficiently.
                
                pass # Done in the code below

        # Implementation of the gradient
        grad = np.zeros(N)
        
        # Part 1: 2 * h[k-j] for valid j
        # j in [max(0, k-N+1), min(N-1, k)]
        j_start1 = max(0, k - N + 1)
        j_end1 = min(N - 1, k)
        if j_start1 <= j_end1:
            # h[k-j] for j in [j_start1, j_end1]
            # Indices of h: k-j_start1 down to k-j_end1
            # Slice: h[k-j_end1 : k-j_start1 + 1]
            # Reversed
            h_slice = h[k-j_end1 : k-j_start1 + 1][::-1]
            grad[j_start1 : j_end1 + 1] += 2.0 * h_slice
            
        # Part 2: 2 * h[j-k] for valid j
        # j in [max(0, k), min(N-1, k+N-1)]
        j_start2 = max(0, k)
        j_end2 = min(N - 1, k + N - 1)
        if j_start2 <= j_end2:
            # h[j-k] for j in [j_start2, j_end2]
            # Indices of h: j_start2-k to j_end2-k
            # Slice: h[j_start2-k : j_end2-k + 1]
            h_slice = h[j_start2-k : j_end2-k + 1]
            grad[j_start2 : j_end2 + 1] += 2.0 * h_slice
            
        return grad

    # Optimization Loop
    h_cur = np.ones(N)
    h_cur = normalize_sum(h_cur, 1.0)
    
    lr = 0.01
    for step in range(2000):
        if time.time() > deadline:
            break
            
        c = np.convolve(h_cur, h_cur)
        val = abs(2.0 * N * np.max(c)) # Since sum=1
        if val < val_best:
            val_best = val
            h_best = h_cur.copy()
            
        grad = grad_max_conv(h_cur)
        
        # Gradient step
        h_new = h_cur - lr * grad
        
        # Project to sum=1
        h_new = normalize_sum(h_new, 1.0)
        
        # Check for improvement (optional, can just accept)
        h_cur = h_new
        
        # Adaptive learning rate
        if step % 100 == 0:
            lr *= 0.9
            
    return h_best
