"""Spectral optimization for Erdos minimum overlap."""
import numpy as np
import time

N = 100
DEADLINE = 150  # seconds, with safety margin


def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h) over all shifts.
    Overlap at shift k is sum_i h[i] * (1 - h[(i-k) % N]).
    We want to minimize the maximum of these overlaps.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    one_minus_h = 1.0 - h
    
    # Compute circular cross-correlation: for each shift k,
    # overlap[k] = sum_i h[i] * one_minus_h[(i-k) % n]
    # This is the circular cross-correlation of h and one_minus_h.
    
    # Use FFT for efficient computation
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    
    # Circular cross-correlation: ifft(conj(H) * O) gives sum_i h[i] * one_minus_h[(i+k)%n]
    # We need sum_i h[i] * one_minus_h[(i-k)%n] which is ifft(H * conj(O))
    # Actually let's be careful:
    # corr[k] = sum_i h[i] * one_minus_h[(i-k) % n]
    # This is the circular cross-correlation.
    # Using FFT: corr = ifft(conj(FFT(h)) * FFT(one_minus_h))
    # Let's verify: ifft(conj(H)*O)[k] = sum_i h[i] * one_minus_h[(i-k)%n]? 
    # Actually, the circular cross-correlation r_xy[k] = sum_n x[n] * conj(y[n-k])
    # For real signals: r_xy[k] = sum_n x[n] * y[n-k]
    # So r_h_omh[k] = sum_i h[i] * one_minus_h[i-k]
    # That's exactly what we want.
    
    corr = np.real(np.fft.ifft(np.conj(H) * O))
    
    return np.max(corr)


def construct():
    """
    Construct step heights h on [0,1] with sum = N/2, minimizing max overlap with shifted 1-h.
    
    Strategy: Start from constant 0.5 baseline, add sinusoidal perturbation.
    h[i] = 0.5 + A * sin(2*pi*f*i/N + phase)
    
    The overlap at shift k:
    sum_i h[i] * (1-h[i-k]) = sum_i (0.5 + A*sin(theta_i)) * (0.5 - A*sin(theta_{i-k}))
    = sum_i [0.25 - 0.5*A*sin(theta_{i-k}) + 0.5*A*sin(theta_i) - A^2*sin(theta_i)*sin(theta_{i-k})]
    
    For a single frequency f: theta_i = 2*pi*f*i/N + phase
    The linear terms sum to zero over a full period (for integer f).
    The A^2 term: -A^2 * sum_i sin(theta_i)*sin(theta_{i-k})
    = -A^2 * (N/2) * cos(2*pi*f*k/N)  [using trig identity and orthogonality]
    
    So overlap[k] = N/4 - A^2 * (N/2) * cos(2*pi*f*k/N)
    
    Wait, let me redo this more carefully.
    
    h[i] = 0.5 + A*sin(2*pi*f*i/N + phi)
    1-h[i] = 0.5 - A*sin(2*pi*f*i/N + phi)
    
    overlap[k] = sum_{i=0}^{N-1} h[i] * (1-h[i-k])
    = sum_i [0.5 + A*sin(2*pi*f*i/N + phi)] * [0.5 - A*sin(2*pi*f*(i-k)/N + phi)]
    
    = sum_i [0.25 - 0.5*A*sin(2*pi*f*(i-k)/N + phi) + 0.5*A*sin(2*pi*f*i/N + phi) - A^2*sin(2*pi*f*i/N+phi)*sin(2*pi*f*(i-k)/N+phi)]
    
    For integer f, sum_i sin(2*pi*f*i/N + phi) = 0 and sum_i sin(2*pi*f*(i-k)/N + phi) = 0.
    
    So overlap[k] = N*0.25 - A^2 * sum_i sin(2*pi*f*i/N+phi)*sin(2*pi*f*(i-k)/N+phi)
    
    Using product-to-sum:
    sin(a)*sin(b) = (cos(a-b) - cos(a+b))/2
    
    a = 2*pi*f*i/N + phi, b = 2*pi*f*(i-k)/N + phi
    a-b = 2*pi*f*k/N
    a+b = 2*pi*f*(2i-k)/N + 2*phi
    
    sum_i sin(a)*sin(b) = sum_i [cos(2*pi*f*k/N) - cos(2*pi*f*(2i-k)/N + 2*phi)]/2
    
    = (N/2)*cos(2*pi*f*k/N) - (1/2)*sum_i cos(2*pi*f*(2i-k)/N + 2*phi)
    
    For integer f, sum_i cos(2*pi*f*(2i-k)/N + 2*phi) = sum_i cos(4*pi*f*i/N - 2*pi*f*k/N + 2*phi)
    
    If 2*f is not a multiple of N (i.e., f not a multiple of N/2), this sum is 0.
    If 2*f is a multiple of N, then cos(4*pi*f*i/N + const) = cos(const) for all i, so sum = N*cos(const).
    
    For simplicity, assume f is such that 2f is not a multiple of N (e.g., f=1,2,...,N/2-1 when N is even).
    
    Then: overlap[k] = N/4 - A^2 * (N/2) * cos(2*pi*f*k/N)
    
    The maximum over k of overlap[k] = N/4 - A^2*(N/2)*min_k(cos(2*pi*f*k/N))
    
    min_k cos(2*pi*f*k/N) = -1 (when k = N/(2f), if that's an integer)
    
    So max overlap = N/4 + A^2*(N/2)
    
    Hmm, that means increasing A increases the max overlap! That's bad.
    
    Wait, I think I made an error. Let me reconsider.
    
    Actually, the overlap we want to minimize is the maximum over shifts. The constant baseline gives N/4 = 25 for N=100.
    
    With the sinusoidal perturbation, we get overlap[k] = N/4 - A^2*(N/2)*cos(2*pi*f*k/N).
    
    The maximum is at cos = -1, giving N/4 + A^2*N/2. This is worse than baseline.
    
    So a single sinusoid doesn't help. We need a different approach.
    
    Let me think about this differently. The problem is to minimize max_k sum_i h[i]*(1-h[i-k]).
    
    Note that sum_i h[i]*(1-h[i-k]) = sum_i h[i] - sum_i h[i]*h[i-k] = N/2 - autocorr_h[k].
    
    So we want to minimize max_k (N/2 - autocorr_h[k]) = N/2 - min_k autocorr_h[k].
    
    Equivalently, maximize min_k autocorr_h[k].
    
    The autocorrelation at lag 0 is sum_i h[i]^2. By Cauchy-Schwarz, this is >= (sum h[i])^2/N = (N/2)^2/N = N/4.
    
    So autocorr[0] >= N/4, and we want min_k autocorr[k] to be as large as possible.
    
    For the constant h=0.5, autocorr[k] = N/4 for all k, so min = N/4, and max overlap = N/2 - N/4 = N/4 = 25.
    
    To improve, we need to make the autocorrelation flatter, with the minimum being higher.
    
    A binary sequence (0/1) with exactly N/2 ones has autocorr[0] = N/2, and for other lags, the autocorrelation can vary.
    
    Let me try a different structural approach: use a greedy or local search to find a binary-like sequence that has a flat autocorrelation profile.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - 5  # 5 second safety margin
    
    # Start with constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try binary sequences with exactly N/2 ones
    # Use local search to minimize max overlap (equivalently, maximize min autocorrelation)
    
    def _score_binary(seq):
        """Compute max overlap for a binary sequence."""
        h = np.array(seq, dtype=np.float64)
        return _compute_max_overlap(h)
    
    def _min_autocorr(seq):
        """Compute minimum autocorrelation for a binary sequence."""
        h = np.array(seq, dtype=np.float64)
        H = np.fft.fft(h)
        ac = np.real(np.fft.ifft(np.conj(H) * H))
        return np.min(ac)
    
    # Initialize with alternating pattern
    seq = np.zeros(N, dtype=np.int8)
    seq[::2] = 1  # 1,0,1,0,...
    
    current_score = _score_binary(seq)
    if current_score < best_score:
        best_score = current_score
        best_h = seq.astype(np.float64)
    
    # Local search: flip pairs of bits to maintain sum = N/2
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_flips = 0
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i+1, N):
                if time.time() > deadline:
                    break
                # Flip seq[i] and seq[j]
                if seq[i] == seq[j]:
                    continue  # Flipping same values doesn't change sum
                # Try flipping
                seq[i] = 1 - seq[i]
                seq[j] = 1 - seq[j]
                new_score = _score_binary(seq)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = seq.astype(np.float64).copy()
                    improved = True
                    n_flips += 1
                else:
                    seq[i] = 1 - seq[i]
                    seq[j] = 1 - seq[j]
                if n_flips > 10:
                    break
            if time.time() > deadline:
                break
    
    # If we're still within time, try a few random restarts
    if time.time() < deadline:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline:
                break
            # Random binary sequence with N/2 ones
            seq = np.zeros(N, dtype=np.int8)
            ones = np.random.choice(N, N//2, replace=False)
            seq[ones] = 1
            
            current_score = _score_binary(seq)
            if current_score < best_score:
                best_score = current_score
                best_h = seq.astype(np.float64).copy()
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(N):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, N):
                        if time.time() > deadline:
                            break
                        if seq[i] == seq[j]:
                            continue
                        seq[i] = 1 - seq[i]
                        seq[j] = 1 - seq[j]
                        new_score = _score_binary(seq)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_h = seq.astype(np.float64).copy()
                            improved = True
                        else:
                            seq[i] = 1 - seq[i]
                            seq[j] = 1 - seq[j]
    
    return best_h
