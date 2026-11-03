"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals
    n_fft = 2 * N

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        # Linear autoconv: indices 0..N-1 are lags 0..N-1, indices N..2N-1 are lags -(N-1)..-1
        # We need all valid lags. For circular conv of size 2N with signal in [0,N):
        # The linear convolution result is in indices 0..2N-2 (lag 0 to 2N-2) 
        # but wrapped. Actually for signal length N padded to 2N:
        # autoconv_circ[0] = sum h[i]^2 (lag 0)
        # autoconv_circ[k] for k=1..N-1 = sum h[i]h[i-k] (positive lags)
        # autoconv_circ[N] = sum h[i]h[i-N] = 0 (since h is zero for i>=N)
        # autoconv_circ[N+k] = sum h[i]h[i-(N+k)] ... this is the negative lags wrapped
        
        # More carefully: h_padded has non-zeros in [0, N-1].
        # Circular conv (h_padded * h_padded)[k] = sum_j h_padded[j] h_padded[(k-j) mod 2N]
        # For k in [0, N-1]: (k-j) mod 2N is in [0, 2N-1]. If j in [0,N-1] and k-j in [0,N-1], 
        #   then k-j >= 0 so no wrap. This gives sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j]h[k-j] which is correct for lag k.
        #   Wait, if k-j < 0, then (k-j) mod 2N = 2N + k - j. Since j <= N-1, 2N+k-j >= 2N+k-(N-1) = N+k+1 > N-1.
        #   So h_padded[(k-j) mod 2N] = 0 for those terms. 
        #   So for k in [0, N-1], autoconv_circ[k] = sum_{j=0}^{N-1} h[j] h[k-j] where h[k-j]=0 if k-j<0 or k-j>=N.
        #   This is the standard linear autoconv for lag k.
        # For k in [N, 2N-1]: Let k' = k - N. Then (k-j) mod 2N = (N+k'-j) mod 2N.
        #   If j <= k', N+k'-j >= N, so mod 2N gives N+k'-j (if < 2N). Since j>=0, N+k'-j <= N+k' <= 2N-1.
        #   If j > k', N+k'-j < N, so mod 2N gives N+k'-j.
        #   Wait, (N+k'-j) mod 2N: if N+k'-j >= 0, it's just N+k'-j. Since j <= N-1 and k' >= 0, N+k'-j >= N-(N-1) = 1 > 0.
        #   So (k-j) mod 2N = N+k'-j.
        #   autoconv_circ[k] = sum_j h[j] h[N+k'-j].
        #   Since h is non-zero only for indices in [0, N-1], h[N+k'-j] is non-zero only if 0 <= N+k'-j <= N-1,
        #   i.e., N+k'-(N-1) <= j <= N+k', i.e., k'+1 <= j <= N+k'.
        #   But j is also in [0, N-1]. So j in [max(0, k'+1), min(N-1, N+k')].
        #   Since k' >= 0, k'+1 >= 1. And N+k' >= N, so min(N-1, N+k') = N-1.
        #   So j in [k'+1, N-1].
        #   autoconv_circ[k] = sum_{j=k'+1}^{N-1} h[j] h[N+k'-j].
        #   Let i = N-1-j. Then j = N-1-i, i from 0 to N-1-k'-1.
        #   h[N+k'-(N-1-i)] = h[k'+1+i].
        #   So sum_{i=0}^{N-2-k'} h[N-1-i] h[k'+1+i].
        #   This is the autoconv for lag -(k'+1) = -(k-N+1).
        #   Actually, standard linear autoconv for lag -m is sum_j h[j] h[j+m].
        #   Here m = k'-1+1 = k'. No, lag is -(k-N). Let lag = k-N = k'. 
        #   Standard: R[-m] = sum_j h[j] h[j+m].
        #   Our expression: sum_{j=k'+1}^{N-1} h[j] h[N+k'-j].
        #   Let j' = j - k'. Then j' from 1 to N-1-k'.
        #   h[j'+k'] h[N-k'+k'-j'] = h[j'+k'] h[N-j']. This doesn't look like standard form.
        
        # Simpler approach: Just compute linear autoconv via FFT properly or use the fact that
        # for real signals, autoconv is symmetric. R[k] = R[-k].
        # R[0] = autoconv_circ[0]
        # R[k] for k=1..N-1 = autoconv_circ[k]
        # R[-k] = R[k]
        # So max is max(autoconv_circ[0], max(autoconv_circ[1:N]))
        # But we must be careful: does autoconv_circ[N+k] correspond to R[-k-1] or R[-k]?
        # From above, autoconv_circ[N+k'] corresponds to lag -(k'+1).
        # So lags represented are: 0, 1, ..., N-1, -(N-1), ..., -1.
        # Specifically:
        # autoconv_circ[0] -> lag 0
        # autoconv_circ[1] -> lag 1
        # ...
        # autoconv_circ[N-1] -> lag N-1
        # autoconv_circ[N] -> lag -(N-1)  (since k'=0, lag = -(0+1) = -1? No, let's recheck)
        #   k = N, k' = 0. Lag = -(0+1) = -1.
        # autoconv_circ[N+1] -> lag = -(1+1) = -2.
        # ...
        # autoconv_circ[2N-1] -> k' = N-1. Lag = -(N-1+1) = -N.
        #   But max lag is N-1. So autoconv_circ[2N-1] should be 0.
        #   Let's check: j in [k'+1, N-1] = [N, N-1], empty sum. So 0. Correct.
        
        # So the valid lags are 0, 1, ..., N-1 and -1, -2, ..., -(N-1).
        # Due to symmetry R[k] = R[-k], we just need max of R[0..N-1].
        # R[0] = autoconv_circ[0]
        # R[k] for k=1..N-1 = autoconv_circ[k]
        
        max_autoconv = np.max(autoconv_circ[:N]) * dx**2
        return max_autoconv / l2_sq

    def compute_grad(h):
        """Compute gradient of autoconv_max_ratio w.r.t. h."""
        h = np.asarray(h, dtype=np.float64)
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        l2_sq = np.sum(h**2) * dx
        max_autoconv = np.max(autoconv_circ[:N]) * dx**2
        max_idx = np.argmax(autoconv_circ[:N])
        
        # Gradient of autoconv_circ[k] w.r.t. h[j] for j in [0, N-1]
        # autoconv_circ[k] = sum_m h_padded[m] h_padded[(k-m) mod n_fft]
        # d/dh[j] = h_padded[(k-j) mod n_fft] + (if j appears in second term, but symmetry handles it)
        # Actually, since it's a square, d/dh[j] of sum_m h[m]h[k-m] is h[k-j] + h[j-k] (if distinct) or 2h[k-j] if k=2j?
        # No, circular convolution: C = h ⊛ h. C[k] = sum_m h[m] h[(k-m) mod n].
        # dC[k]/dh[j] = h[(k-j) mod n] + h[j] * (coefficient from second term?)
        # Let's expand: C[k] = h[j]h[(k-j) mod n] + sum_{m != j} h[m]h[(k-m) mod n].
        # dC[k]/dh[j] = h[(k-j) mod n] + (if (k-m) mod n == j for some m, then h[m]).
        # (k-m) mod n = j => m = (k-j) mod n.
        # So if m = (k-j) mod n is in [0, n-1], then there is a term h[(k-j) mod n].
        # So dC[k]/dh[j] = h[(k-j) mod n] + h[(k-j) mod n] = 2 h[(k-j) mod n]?
        # Wait, if j = (k-j) mod n, i.e., 2j = k mod n, then the two terms are the same.
        # But generally, for j in [0, N-1] and k in [0, N-1]:
        # Term 1: m=j, gives h[(k-j) mod n_fft].
        # Term 2: (k-m) mod n_fft = j => m = (k-j) mod n_fft.
        #   If m is in [0, N-1], then h[m] contributes.
        #   Note m = (k-j) mod n_fft. Since k,j in [0, N-1], k-j in [-(N-1), N-1].
        #   If k>=j, m = k-j in [0, N-1].
        #   If k<j, m = n_fft + k - j = 2N + k - j in [N+1, 2N-1].
        #   In the second case, m is not in [0, N-1], so h[m] = 0 (since h_padded is zero there).
        #   So Term 2 is non-zero only if k>=j, and then it is h[k-j].
        #   Term 1: h[(k-j) mod n_fft].
        #   If k>=j, (k-j) mod n_fft = k-j. So Term 1 = h[k-j].
        #   If k<j, (k-j) mod n_fft = 2N+k-j. h[2N+k-j] = 0.
        #   So if k>=j: dC[k]/dh[j] = h[k-j] + h[k-j] = 2 h[k-j].
        #   If k<j: dC[k]/dh[j] = 0 + 0 = 0? 
        #   Wait, let's re-evaluate.
        #   C[k] = sum_{m=0}^{n_fft-1} h[m] h[(k-m) mod n_fft].
        #   Only m in [0, N-1] have h[m] != 0.
        #   So C[k] = sum_{m=0}^{N-1} h[m] h[(k-m) mod n_fft].
        #   dC[k]/dh[j] = h[(k-j) mod n_fft] + sum_{m=0, m!=j}^{N-1} h[m] * d/dh[j] h[(k-m) mod n_fft].
        #   The second sum is non-zero only if (k-m) mod n_fft = j for some m.
        #   As derived, this happens if m = (k-j) mod n_fft is in [0, N-1].
        #   So dC[k]/dh[j] = h[(k-j) mod n_fft] + (h[(k-j) mod n_fft] if (k-j) mod n_fft < N else 0).
        #   Let m2 = (k-j) mod n_fft.
        #   If m2 < N: dC[k]/dh[j] = 2 * h[m2].
        #   If m2 >= N: dC[k]/dh[j] = h[m2] (which is 0) = 0.
        #   
        #   So for k in [0, N-1] and j in [0, N-1]:
        #   m2 = (k-j) mod (2N).
        #   If m2 < N: grad[j] = 2 * h[m2].
        #   Else: grad[j] = 0.
        
        grad_autoconv = np.zeros(N)
        k = max_idx
        for j in range(N):
            m2 = (k - j) % n_fft
            if m2 < N:
                grad_autoconv[j] = 2.0 * h[m2]
            else:
                grad_autoconv[j] = 0.0
        
        # Scale by dx^2 (since autoconv_vals = autoconv_circ * dx^2)
        grad_autoconv *= dx**2
        
        # Gradient of l2_sq = sum(h[i]^2) * dx
        grad_l2 = 2.0 * h * dx
        
        # Score S = A / L, where A = max_autoconv, L = l2_sq
        # dS/dh[j] = (grad_A[j] * L - A * grad_L[j]) / L^2
        grad_score = (grad_autoconv * l2_sq - max_autoconv * grad_l2) / (l2_sq ** 2)
        
        return grad_score, max_autoconv, l2_sq

    def fourier_to_space(coeffs):
        """Convert Fourier coefficients to spatial domain step heights.
        coeffs[0] is DC, coeffs[1..] are amplitudes for cos(k*x) and sin(k*x) or complex?
        Let's use real coefficients: h[x] = c0 + sum_{k=1}^{K} (a_k cos(2*pi*k*x/L) + b_k sin(2*pi*k*x/L))
        Here x is in [-1/4, 1/4], L = 1/2.
        Actually, simpler: h[n] = c0 + sum_{k=1}^{K} (a_k cos(2*pi*k*n/N) + b_k sin(2*pi*k*n/N))
        where n is 0..N-1. This corresponds to frequencies on the discrete grid.
        """
        n = np.arange(N)
        h = np.zeros(N)
        h += coeffs[0]
        for k in range(1, len(coeffs)//2 + 1):
            if 2*k < len(coeffs):
                a_k = coeffs[2*k-1]
                b_k = coeffs[2*k]
                h += a_k * np.cos(2*np.pi*k*n/N) + b_k * np.sin(2*np.pi*k*n/N)
        return np.maximum(h, 0)

    def space_to_fourier(h):
        """Convert spatial domain to Fourier coefficients (least squares projection)."""
        n = np.arange(N)
        K = 20  # number of harmonics
        num_coeffs = 1 + 2*K
        A = np.zeros((N, num_coeffs))
        A[:, 0] = 1.0
        for k in range(1, K+1):
            A[:, 2*k-1] = np.cos(2*np.pi*k*n/N)
            A[:, 2*k] = np.sin(2*np.pi*k*n/N)
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        return coeffs

    def objective_fourier(coeffs):
        """Objective function for Fourier coefficient optimization."""
        h = fourier_to_space(coeffs)
        if np.sum(h) < 1e-10:
            return 1e10
        return autoconv_max_ratio(h)

    best_score = float('inf')
    best_h = None

    centers = np.arange(N) - N // 2

    # 1. Generate initial candidates in spatial domain
    candidates = []
    
    # Gaussian peaks
    for sigma in [5, 10, 20, 50, 100, 200]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    # Triangular peaks
    for width in [20, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)
    
    # Box functions
    for half_width in [20, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(centers) <= half_width
        h[mask] = 1.0
        candidates.append(h)
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Evaluate spatial candidates
    for h in candidates:
        if time.time() > deadline:
            break
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = autoconv_max_ratio(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # 2. Fourier-domain optimization
    K = 20
    num_coeffs = 1 + 2*K
    
    # Initialize Fourier coefficients from best spatial candidate
    if best_h is not None:
        init_coeffs = space_to_fourier(best_h)
    else:
        init_coeffs = np.zeros(num_coeffs)
        init_coeffs[0] = 1.0

    # Optimize in Fourier domain using Nelder-Mead
    # We need to handle the non-negativity constraint. 
    # The projection to non-negative in space might make the objective non-smooth in Fourier space.
    # However, Nelder-Mead is derivative-free and can handle this.
    
    def fourier_objective_with_time_check(coeffs):
        if time.time() > deadline - 5:
            return 1e10
        h = fourier_to_space(coeffs)
        if np.sum(h) < 1e-10:
            return 1e10
        return autoconv_max_ratio(h)

    try:
        result = minimize(fourier_objective_with_time_check, init_coeffs, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
        
        if result.fun < best_score:
            best_score = result.fun
            best_h = fourier_to_space(result.x)
    except Exception as e:
        pass

    # 3. Refine in spatial domain using gradient descent
    if best_h is not None and time.time() < deadline - 10:
        h = best_h.copy()
        current_score = best_score
        lr = 0.01
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            grad, A, L = compute_grad(h)
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0)
            new_score = autoconv_max_ratio(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
                lr *= 1.01  # slightly increase lr if improving
            else:
                lr *= 0.95
                if lr < 1e-8:
                    break
        if current_score < best_score:
            best_score = current_score
            best_h = h

    # 4. Further refine with coordinate descent on a subset of coordinates if time permits
    if best_h is not None and time.time() < deadline - 5:
        h = best_h.copy()
        current_score = best_score
        step_size = 0.05
        # Random coordinate descent
        np.random.seed(42)
        for it in range(500):
            if time.time() > deadline - 2:
                break
            idx = np.random.randint(N)
            for direction in [1, -1]:
                h_trial = h.copy()
                h_trial[idx] = max(0, h[idx] + direction * step_size)
                trial_score = autoconv_max_ratio(h_trial)
                if trial_score < current_score:
                    h = h_trial
                    current_score = trial_score
                    break
        if current_score < best_score:
            best_score = current_score
            best_h = h

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
