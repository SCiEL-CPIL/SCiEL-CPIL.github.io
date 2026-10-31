import numpy as np

N = 600

def construct():
    # Try a few Gaussian widths and pick the best one
    # Metric assumption: max(S_k for k != 0)
    # We want to minimize this.
    
    best_f = None
    best_score = float('inf')
    
    # Candidate widths (sigma)
    sigmas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    centers = np.arange(N)
    center = N // 2
    
    for sigma in sigmas:
        f = np.exp(-0.5 * ((centers - center) / sigma) ** 2)
        f_sum = np.sum(f)
        if f_sum == 0:
            continue
        f = f / f_sum
        
        # Compute autoconvolution
        # Use FFT for speed
        F = np.fft.fft(f, n=2*N)
        C = np.fft.ifft(F * F).real
        # C is circular convolution, but since f is zero-padded to 2N, it's linear
        # The peak is at N (shifted)
        # We need to check k != 0
        # In the FFT result, index 0 is lag 0.
        # Due to zero padding, lags 0 to N-1 are in 0 to N-1?
        # Standard linear convolution via FFT:
        # f is length N, zero padded to 2N.
        # Convolution length 2N-1.
        # Peak at lag 0 is at index 0?
        # np.fft.ifft returns circular conv of length 2N.
        # The linear conv is the first 2N-1 elements?
        # Actually, for real signals, it's easier to use scipy.signal.fftconvolve or just np.convolve if N is small.
        # N=600 is small. np.convolve is fast enough.
        
        c = np.convolve(f, f, mode='full')
        # c[0] is lag -(N-1)? No.
        # np.convolve(f, f, 'full') length 2N-1.
        # Index N-1 corresponds to lag 0.
        
        c0 = c[N-1]
        
        # We want max c[k] for k != N-1
        # Mask the center
        c_masked = c.copy()
        c_masked[N-1] = 0
        
        max_sidelobe = np.max(c_masked)
        
        # If the metric is max_sidelobe, we minimize it.
        # If the metric is max_sidelobe / sum(f)^2, since sum(f)=1, it's just max_sidelobe.
        
        if max_sidelobe < best_score:
            best_score = max_sidelobe
            best_f = f
            
    return list(best_f)
