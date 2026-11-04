import numpy as np

N = 400

def construct():
    # Start with a rectangular pulse (all ones)
    # This is a common initialization for low-PAPR / low-sidelobe designs
    x = np.ones(N)
    
    # Normalize sum to 1
    x = x / np.sum(x)
    
    M = 2 * N
    best_x = x.copy()
    best_score = np.inf
    
    # Relaxation factor
    # 0.5 is a safe start. 0.8 might be faster.
    lambda_relax = 0.8
    
    for iteration in range(2000):
        # Compute FFT with zero padding
        x_pad = np.zeros(M)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        P = np.abs(X)**2
        
        # Calculate current score
        R = np.fft.ifft(P).real
        # Extract sidelobes
        # R[0] is lag 0
        # R[1:N] are lags 1..N-1
        # R[M-N+1:M] are lags -(N-1)..-1
        R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
        
        sum_x = np.sum(x)
        if abs(sum_x) < 1e-9:
            sum_x = 1e-9
        score = np.max(np.abs(R_sidelobes)) / (sum_x**2)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Spectral Flattening
        P_mean = np.mean(P)
        mag = np.sqrt(P)
        mag_target = np.sqrt(P_mean)
        
        # Relaxation
        mag_new = (1 - lambda_relax) * mag + lambda_relax * mag_target
        
        # Preserve phase
        phase = np.angle(X)
        X_new = mag_new * np.exp(1j * phase)
        
        # Enforce Hermitian symmetry for real-valued x
        # M is even (2*N)
        half = M // 2
        X_new[0] = np.real(X_new[0])
        X_new[half] = np.real(X_new[half])
        
        # Mirror the first half to the second half
        # Indices 1 to half-1
        # Corresponding indices M-1 down to M-(half-1)
        # X_new[1:half] -> X_new[M-1:M-half:-1]
        # In numpy: X_new[half+1:M] = conj(X_new[1:half])[::-1]
        
        left_part = X_new[1:half]
        right_part = np.conj(left_part)[::-1]
        X_new[half+1:M] = right_part
        
        # Inverse FFT
        x_new = np.fft.ifft(X_new).real[:N]
        
        # Normalize to sum to 1
        sum_new = np.sum(x_new)
        if abs(sum_new) < 1e-9:
            # Fallback if sum becomes zero
            x_new = x_new + 1e-6 * np.ones(N)
            sum_new = np.sum(x_new)
        x_new = x_new / sum_new
        
        x = x_new
        
    return best_x
