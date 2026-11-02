import numpy as np

N = 400

def construct():
    # Initialize with random zero-mean unit-norm vector
    np.random.seed(42)
    x = np.random.randn(N)
    x -= np.mean(x)
    x /= np.linalg.norm(x)
    
    # Objective: Minimize Variance of Power Spectrum
    # This promotes flat spectrum, minimizing autoconvolution peaks.
    
    learning_rate = 0.1
    num_iters = 500
    
    for _ in range(num_iters):
        # Compute FFT
        X = np.fft.fft(x)
        P = np.abs(X)**2
        P_mean = np.mean(P)
        
        # Gradient of sum((P - P_mean)^2) w.r.t x
        # d/dx sum_k (|X_k|^2 - mu)^2
        # Let f = sum_k (P_k - mu)^2
        # df/dP_k = 2(P_k - mu)
        # dP_k/dx = 2 * Re(X_k * conj(FFT_inv( ... ))) ?
        # Simpler: Use finite difference or chain rule.
        # Chain rule:
        # X_k = sum_j x_j e^{-i 2 pi k j / N}
        # P_k = X_k X_k^*
        # dP_k/dx_j = e^{-i 2 pi k j / N} X_k^* + e^{i 2 pi k j / N} X_k
        #             = 2 Re( X_k^* e^{-i 2 pi k j / N} )
        # df/dx_j = sum_k 2(P_k - mu) * 2 Re( X_k^* e^{-i 2 pi k j / N} )
        # This can be computed via IFFT.
        # Let G_k = 2(P_k - mu) X_k^*
        # Then grad_j = 2 Re( IFFT(G)_j )
        
        G = 2.0 * (P - P_mean) * np.conj(X)
        grad = 2.0 * np.real(np.fft.ifft(G))
        
        # Update
        x_new = x - learning_rate * grad
        
        # Project onto constraints: sum(x)=0, norm(x)=1
        x_new -= np.mean(x_new)
        norm = np.linalg.norm(x_new)
        if norm < 1e-10:
            # Re-randomize if collapsed
            x_new = np.random.randn(N)
            x_new -= np.mean(x_new)
            x_new /= np.linalg.norm(x_new)
        else:
            x_new /= norm
            
        x = x_new
        
    return x
