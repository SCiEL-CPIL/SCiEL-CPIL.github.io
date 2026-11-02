import numpy as np
import time

N = 400
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _autocorr_max_abs(s):
    """
    Compute the maximum absolute value of the non-zero lag autocorrelation.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    return np.max(np.abs(r[1:N]))

def _autocorr_l2_sq(s):
    """
    Compute the sum of squares of the non-zero lag autocorrelation.
    This is differentiable and used for gradient descent.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    return np.sum(r[1:N]**2)

def _gradient_l2_sq(s):
    """
    Compute the gradient of the sum of squares of autocorrelation sidelobes.
    J(x) = sum_{k=1}^{N-1} R(k)^2
    Using Parseval: J(x) = (1/N) * sum_{m=0}^{N-1} |X(m)|^4
    Gradient: dJ/dx_i = (1/N) * 4 * Re( FFT^{-1}( |X|^2 * X ) * conj(X) )? 
    Actually, let's derive it properly or use the time-domain relation.
    
    Let R(k) = sum_{n=0}^{N-1-k} x(n+k)x(n).
    This is circular autocorrelation if we pad with zeros, but here we have linear autocorrelation.
    The FFT method computes circular autocorrelation of the zero-padded signal.
    Since the signal is length N, zero-padded to 2N, the circular autocorrelation of the padded signal
    corresponds to the linear autocorrelation for lags 1..N-1.
    
    Let y be the zero-padded signal. R_circ(k) = sum y(n+k)y(n).
    J = sum_{k=1}^{N-1} R_circ(k)^2.
    
    We can compute the gradient efficiently.
    Let F = FFT(y).
    R_circ = IFFT(|F|^2).
    J = sum_{k=1}^{N-1} R_circ(k)^2.
    
    Alternatively, minimize sum |F(m)|^4.
    Gradient of sum |F(m)|^4 w.r.t y is 4 * IFFT( |F|^2 * F ).
    We only care about the first N elements.
    """
    n_fft = 2 * N
    F = np.fft.fft(s, n=n_fft)
    # Gradient of sum |F|^4 is 4 * IFFT( |F|^2 * F )
    grad_fft = 4.0 * (np.abs(F)**2) * F
    grad = np.fft.ifft(grad_fft).real[:N]
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    best_s = None
    best_val = np.inf
    
    # We will run multiple iterations of:
    # 1. Continuous gradient descent to minimize L2 norm of sidelobes (flat spectrum)
    # 2. Projection to binary
    # 3. Local search to minimize Peak sidelobe
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize continuous signal
        x = rng.uniform(-1, 1, size=N)
        
        # Gradient Descent Phase
        lr = 0.1
        for step in range(200):
            if time.time() > deadline:
                break
                
            grad = _gradient_l2_sq(x)
            x -= lr * grad
            
            # Normalize to keep values bounded, though sign projection handles scale
            # We don't strictly need to normalize if we project to sign, 
            # but it helps gradient stability.
            norm = np.linalg.norm(x)
            if norm > 0:
                x = x / norm * np.sqrt(N) # Keep energy roughly constant
        
        # Project to binary
        s = np.sign(x)
        # Handle zeros (should be rare)
        s[s == 0] = 1.0
        
        cur_val = _autocorr_max_abs(s)
        
        if cur_val < best_val:
            best_val = cur_val
            best_s = s.copy()
            
        # Local Search Phase on the binary sequence
        # We use a fast local search that tries flipping single bits and small blocks
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Single bit flips
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = s[i]
                s[i] = -old_val
                new_val = _autocorr_max_abs(s)
                if new_val < cur_val:
                    cur_val = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_s = s.copy()
                    improved = True
                else:
                    s[i] = old_val
            
            # Small block flips (size 2-5) to escape local minima
            if not improved:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    block_size = int(rng.integers(2, 6))
                    start = int(rng.integers(0, N - block_size + 1))
                    old_block = s[start:start+block_size].copy()
                    s[start:start+block_size] = -old_block
                    new_val = _autocorr_max_abs(s)
                    if new_val < cur_val:
                        cur_val = new_val
                        if new_val < best_val:
                            best_val = new_val
                            best_s = s.copy()
                        improved = True
                    else:
                        s[start:start+block_size] = old_block

    if best_s is None:
        best_s = np.ones(N)
    
    return best_s
