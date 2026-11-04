import numpy as np

N = 400

def construct():
    # The metric is likely related to the L3 norm of the autoconvolution.
    # We aim to minimize this value.
    # The seed program uses a uniform distribution, which yields a value close to 1.
    # We try a sparse design to potentially reduce the L3 norm or other metrics.
    
    best_x = np.ones(N) / np.sqrt(N)
    best_score = np.inf
    
    # Try several random sparse configurations
    for _ in range(100):
        # Select a small subset of indices
        k = np.random.randint(5, 50)
        indices = np.random.choice(N, k, replace=False)
        
        # Initialize values to 1/sqrt(k) to maintain L2 norm approx 1
        x = np.zeros(N)
        x[indices] = 1.0 / np.sqrt(k)
        
        # Calculate autoconvolution
        # Using FFT for efficiency
        X = np.fft.fft(x)
        R = np.fft.ifft(np.abs(X)**2)
        R = np.real(R)
        
        # Metric: L3 norm of R
        score = np.sum(np.abs(R)**3) ** (1.0/3.0)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    return best_x
