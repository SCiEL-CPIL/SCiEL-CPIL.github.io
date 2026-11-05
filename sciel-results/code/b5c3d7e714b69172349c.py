import numpy as np
import time

N = 400

def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _compute_metric_l3(acf, n):
    """Compute L3 norm of autocorrelation sidelobes."""
    if n <= 1:
        return 0.0
    return float(np.sum(np.abs(acf[1:n]) ** 3))

def _compute_gradient(h, acf, n):
    """
    Compute gradient of L3 metric w.r.t h.
    M = sum |R_k|^3
    dM/dh_i = sum_k 3 |R_k|^2 sign(R_k) (h_{i-k} + h_{i+k})
    """
    # Weights for each lag k
    # R_k is acf[k]
    # Note: acf from irfft is real.
    # R_k for k=1..n-1
    R_sidelobes = acf[1:n]
    W = 3.0 * np.abs(R_sidelobes) ** 2 * np.sign(R_sidelobes)
    
    # Construct symmetric kernel K of length n
    # K[k] = W[k-1] for k=1..n-1
    # K[n-k] = W[k-1] for k=1..n-1
    # K[0] = 0
    K = np.zeros(n)
    # W has length n-1
    # Indices in K: 1 to n-1
    K[1:n] = W
    K[1:n] = W[::-1] # Wait, K[n-1] should correspond to lag 1?
    # Let's check indices.
    # dR_k/dh_i involves h_{i-k} and h_{i+k}.
    # Sum over k.
    # This is a convolution of h with a kernel that has W at +/- k.
    # Kernel length n.
    # K[0] = 0
    # K[k] = W[k-1] for k=1..n-1 (this is h_{i-k} term? No)
    # Let's use FFT directly.
    # G = IFFT(FFT(h) * FFT(K))
    # K must be the kernel for the linear operator.
    # L(h)_i = sum_k W_k (h_{i-k} + h_{i+k})
    # This is equivalent to convolving h with K where K has W_k at index k and -k (mod n)?
    # Since we are using circular convolution via FFT, we need K to represent the periodic extension.
    # K[j] = W_j if j > 0, W_{n-j} if j < 0?
    # W is defined for k=1..n-1.
    # K[1] = W[0] (lag 1)
    # K[2] = W[1] (lag 2)
    # ...
    # K[n-1] = W[n-2] (lag n-1)
    # And K[n-1] is also at index -1 (mod n).
    # So K is symmetric: K[k] = W[k-1] and K[n-k] = W[k-1].
    
    K = np.zeros(n)
    K[1:n] = W
    K[1:n] = W[::-1] # This sets K[1]=W[-1], K[2]=W[-2]... which is wrong.
    # W[0] is lag 1.
    # K[1] should be W[0].
    # K[n-1] should be W[0].
    # K[2] should be W[1].
    # K[n-2] should be W[1].
    
    K[1:n] = W
    K[1:n] = W[::-1] # No, this overwrites.
    
    # Correct assignment:
    K = np.zeros(n)
    # Left side: indices 1 to n-1 correspond to lags 1 to n-1?
    # In circular convolution, index k corresponds to shift k.
    # We want shift +k and -k.
    # Shift +k is index k.
    # Shift -k is index n-k.
    # So K[k] = W[k-1] for k=1..n-1.
    # And K[n-k] = W[k-1] for k=1..n-1.
    
    for k in range(1, n):
        val = W[k-1]
        K[k] += val
        K[n-k] += val
        
    # Now compute G
    H = np.fft.rfft(h, n=n)
    Kf = np.fft.rfft(K, n=n)
    G = np.fft.irfft(H * Kf, n=n)
    return G

def _exact_delta_m(h, i, acf, n):
    """
    Calculate exact change in L3 metric if h[i] is flipped.
    """
    R = acf[1:n]
    old_M = np.sum(np.abs(R) ** 3)
    
    # Delta R_k = -2 h[i] (h[i-k] + h[i+k])
    # We need to sum |R_k + Delta R_k|^3 - |R_k|^3
    # To save time, we only compute for k where Delta R_k != 0.
    # Delta R_k is non-zero if i-k or i+k is in [0, n-1].
    # For k in 1..n-1:
    # i-k in [0, n-1] => k <= i
    # i+k in [0, n-1] => k <= n-1-i
    # So k is in [1, max(i, n-1-i)].
    # Actually, it's the union.
    # Max k is n-1.
    # Min k is 1.
    # So we iterate k=1 to n-1.
    
    delta_M = 0.0
    hi = h[i]
    for k in range(1, n):
        # Get neighbors
        left_idx = i - k
        right_idx = i + k
        
        neighbor_sum = 0.0
        if 0 <= left_idx < n:
            neighbor_sum += h[left_idx]
        if 0 <= right_idx < n:
            neighbor_sum += h[right_idx]
            
        if neighbor_sum == 0:
            continue
            
        dR = -2.0 * hi * neighbor_sum
        new_R = R[k-1] + dR
        delta_M += np.abs(new_R) ** 3 - np.abs(R[k-1]) ** 3
        
    return delta_M

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initialize
    # Try Legendre sequence
    p = 401 # Prime > 400
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 1
        
    h_leg = np.zeros(p)
    for i in range(1, p):
        val = pow(i, (p-1)//2, p)
        if val == 1:
            h_leg[i] = 1.0
        elif val == p-1:
            h_leg[i] = -1.0
        else:
            h_leg[i] = 1.0
    h = h_leg[:N].copy()
    
    acf = _compute_acf(h)
    current_M = _compute_metric_l3(acf, N)
    
    best_h = h.copy()
    best_M = current_M
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < deadline:
        # Compute Gradient
        G = _compute_gradient(h, acf, N)
        
        # Find best flip based on gradient
        # Score = h[i] * G[i]
        # We want to maximize this (to decrease M)
        scores = h * G
        best_i = int(np.argmax(scores))
        best_score = scores[best_i]
        
        if best_score <= 0:
            # No improving move found via gradient
            # Try random moves or cool down
            if T > T_min:
                # Sample a few random moves
                found_improvement = False
                for _ in range(10):
                    i = int(rng.integers(N))
                    dM = _exact_delta_m(h, i, acf, N)
                    if dM < 0:
                        # Accept
                        h[i] = -h[i]
                        acf = _compute_acf(h) # Recompute for simplicity and accuracy
                        current_M = _compute_metric_l3(acf, N)
                        if current_M < best_M:
                            best_M = current_M
                            best_h = h.copy()
                        found_improvement = True
                        break
                    else:
                        # SA Accept
                        if np.random.random() < np.exp(-dM / T):
                            h[i] = -h[i]
                            acf = _compute_acf(h)
                            current_M = _compute_metric_l3(acf, N)
                            if current_M < best_M:
                                best_M = current_M
                                best_h = h.copy()
                            found_improvement = True
                            break
                if not found_improvement:
                    T *= cooling
            else:
                # Cooldown complete, break or continue with random?
                # Let's just break if T is very low and no improvement
                break
        else:
            # Apply best flip
            i = best_i
            dM = _exact_delta_m(h, i, acf, N)
            if dM < 0:
                h[i] = -h[i]
                acf = _compute_acf(h)
                current_M = _compute_metric_l3(acf, N)
                if current_M < best_M:
                    best_M = current_M
                    best_h = h.copy()
                # Reset T if we found a good improvement?
                # T = max(T, 1.0) # Optional
            else:
                # Gradient said good, but exact says bad?
                # Accept with SA
                if np.random.random() < np.exp(-dM / T):
                    h[i] = -h[i]
                    acf = _compute_acf(h)
                    current_M = _compute_metric_l3(acf, N)
                    if current_M < best_M:
                        best_M = current_M
                        best_h = h.copy()
                T *= cooling

    return best_h
